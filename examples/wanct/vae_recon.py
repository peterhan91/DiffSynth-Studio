import torch
from torchvision import transforms
from diffsynth import ModelManager, WanVideoPipeline
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
import os
import traceback # Added for detailed error logging
import random # Added for random sampling

# --- Configuration --- 
# !!! UPDATE THESE PATHS !!!
VAE_MODEL_PATH = "/home/than/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth" # CHOOSE YOUR VAE
EXAMPLE_VIDEO_PATH = "examples/wanct/output/train_3_a_1_resampled_axis0.mp4"
OUTPUT_DIR_FOR_COMPARISON_IMAGES = "vae_reconstruction_test_output"

# Device to run the VAE on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE input image size (consistent with train_wan_t2v.py defaults, adjust if necessary)
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Number of frames to sample from the video for testing
NUM_TEST_FRAMES = 5
# --- End Configuration ---

def load_vae(vae_path, device):
    print(f"Loading VAE from: {vae_path}")
    if not os.path.exists(vae_path):
        print(f"ERROR: VAE model path does not exist: {vae_path}")
        return None

    model_manager_vae = ModelManager(torch_dtype=torch.bfloat16, device=device)
    vae = None
    
    try:
        print(f"Loading VAE model into ModelManager from: {vae_path}")
        model_manager_vae.load_models([vae_path])
        print(f"ModelManager loaded models. Attributes: {dir(model_manager_vae)}")

        print("Attempting to create WanVideoPipeline from ModelManager...")
        # We only loaded a VAE, so other components required by WanVideoPipeline might be missing.
        # The from_model_manager might fail or might return a pipeline with only the VAE populated.
        # For this script, we only need the VAE component.
        try:
            # We pass device=None to from_model_manager, as the VAE will be moved to the correct device later.
            # Components are expected to be on CPU by default by from_model_manager
            # We need to ensure model_manager_vae was initialized with device='cpu' or components are on cpu.
            # Let's re-initialize model_manager for CPU to be safe for pipeline creation, then move VAE to target device.
            
            mm_for_pipeline = ModelManager(torch_dtype=torch.bfloat16, device="cpu") # Create on CPU for pipeline
            mm_for_pipeline.load_models([vae_path])
            print("Creating WanVideoPipeline...")
            pipeline = WanVideoPipeline.from_model_manager(mm_for_pipeline, device="cpu") #Keep pipeline on CPU for now
            print("WanVideoPipeline created.")
            
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                vae = pipeline.vae
                print(f"VAE component retrieved from WanVideoPipeline. Type: {type(vae)}")
                if not (isinstance(vae, torch.nn.Module) and hasattr(vae, 'encode') and hasattr(vae, 'decode')):
                    print("ERROR: pipeline.vae is not a valid VAE module (missing encode/decode or not nn.Module).")
                    vae = None # Invalidate if not a proper VAE
            else:
                print("ERROR: WanVideoPipeline does not have a 'vae' attribute or it is None.")
                # Fallback or further inspection of `pipeline` attributes could be added here if needed
                print(f"Attributes of pipeline: {dir(pipeline)}")

        except Exception as e_pipeline:
            print(f"Error creating WanVideoPipeline or accessing VAE from it: {e_pipeline}")
            traceback.print_exc()
            print("Falling back to direct attribute access on ModelManager as a last resort.")
            # Fallback to direct access if pipeline creation fails (e.g. if other components are strictly required by WanVideoPipeline)
            if hasattr(model_manager_vae, 'wan_video_vae'):
                candidate = getattr(model_manager_vae, 'wan_video_vae')
                if isinstance(candidate, torch.nn.Module) and hasattr(candidate, 'encode') and hasattr(candidate, 'decode'):
                    vae = candidate
                    print(f"Fallback: Successfully identified VAE using attribute: 'wan_video_vae'")

        if vae is None:
            print(f"ERROR: Could not identify or access the VAE model from {vae_path}.")
            return None
        
        vae.to(device) # Move the final VAE to the target device
        vae.eval()
        print(f"VAE loaded successfully ({type(vae).__name__}) and set to eval mode on {device}.")
        return vae
    except Exception as e:
        print(f"Outer error during VAE loading process: {e}")
        traceback.print_exc()
        return None

def load_and_preprocess_frames(video_path, num_frames, target_height, target_width):
    print(f"Loading frames from: {video_path}")
    if not os.path.exists(video_path):
        print(f"ERROR: Example video path does not exist: {video_path}")
        return None, None

    original_frames_pil = []
    try:
        with imageio.get_reader(video_path) as reader:
            try:
                total_frames_in_video = reader.count_frames()
                if isinstance(total_frames_in_video, float) and np.isinf(total_frames_in_video):
                    # Handle cases where count_frames returns inf (e.g. streams or problematic files)
                    print("Warning: Could not determine total number of frames. Will attempt to read sequentially.")
                    total_frames_in_video = -1 # Sentinel for sequential reading
            except Exception as e_count:
                print(f"Warning: Error calling reader.count_frames(): {e_count}. Will attempt to read sequentially.")
                total_frames_in_video = -1 # Sentinel for sequential reading

            frame_indices_to_load = []
            if total_frames_in_video != -1 and total_frames_in_video >= num_frames:
                print(f"Video has {total_frames_in_video} frames. Randomly sampling {num_frames} frames.")
                frame_indices_to_load = sorted(random.sample(range(total_frames_in_video), num_frames))
            elif total_frames_in_video != -1 and total_frames_in_video > 0: # Less than num_frames but known count
                print(f"Video has {total_frames_in_video} frames (less than requested {num_frames}). Loading all available frames.")
                frame_indices_to_load = list(range(total_frames_in_video))
            else: # Unknown frame count or error, load sequentially up to num_frames
                print(f"Attempting to load first {num_frames} frames sequentially.")
                # This loop will attempt to read `num_frames` or until an error/end of video
                for i in range(num_frames):
                    try:
                        frame_np = reader.get_data(i)
                        original_frames_pil.append(Image.fromarray(frame_np).convert('RGB'))
                    except IndexError:
                        print(f"Reached end of video after {i} frames during sequential read.")
                        break
                    except Exception as e_get_seq:
                        print(f"Error getting frame {i} sequentially: {e_get_seq}. Stopping frame loading.")
                        break
                frame_indices_to_load = None # Signal that frames are already loaded

            if frame_indices_to_load is not None:
                for idx in frame_indices_to_load:
                    try:
                        frame_np = reader.get_data(idx)
                        original_frames_pil.append(Image.fromarray(frame_np).convert('RGB'))
                    except IndexError:
                        print(f"Error: Frame index {idx} out of bounds. This shouldn't happen if count_frames was accurate.")
                        break # Stop if an index is bad
                    except Exception as e_get_idx:
                        print(f"Error getting frame at index {idx}: {e_get_idx}. Skipping frame.")
                        continue
            
            print(f"Loaded {len(original_frames_pil)} frames.")

        if not original_frames_pil:
            print("No frames were loaded from the video.")
            return None, None

        preprocess = transforms.Compose([
            transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_tensor = torch.stack([preprocess(frame) for frame in original_frames_pil])
        print(f"Preprocessed tensor shape: {input_tensor.shape}")
        return original_frames_pil, input_tensor
    except Exception as e:
        print(f"Error reading or preprocessing video: {e}")
        return None, None

def reconstruct_frames(vae, input_tensor, device):
    print("Reconstructing frames...")
    if vae is None or input_tensor is None:
        print("VAE or input tensor not available for reconstruction.")
        return None
    
    # input_tensor initial shape: (NUM_FRAMES, C, H, W)
    input_tensor = input_tensor.to(device=device, dtype=torch.bfloat16)
    print(f"Input tensor original - shape: {input_tensor.shape}, device: {input_tensor.device}, dtype: {input_tensor.dtype}")

    # Permute to (C, NUM_FRAMES, H, W) for VAE
    video_for_vae = input_tensor.permute(1, 0, 2, 3)
    print(f"Permuted video for VAE input list - shape: {video_for_vae.shape}")

    with torch.no_grad():
        try:
            # Pass as a list: [ (C, T, H, W) ], and device argument is required.
            # Using tiled=False for simplicity, to use single_encode/decode path.
            encoded_output = vae.encode([video_for_vae], device=device, tiled=False)
            # encoded_output should be (1, C_latent, T_latent, H_latent, W_latent)
            print(f"Encoded output (latents) from VAE - shape: {encoded_output.shape}, device: {encoded_output.device}, dtype: {encoded_output.dtype}")
            latents = encoded_output # In this case, encoded_output is the latents for the single video in the list

            # Pass latents to decode. device argument is required.
            decoded_tensor_batched = vae.decode(latents, device=device, tiled=False)
            # decoded_tensor_batched should be (1, C_out, T_out, H_out, W_out)
            print(f"Decoded tensor (batched) from VAE - shape: {decoded_tensor_batched.shape}")

            if decoded_tensor_batched.shape[0] != 1:
                raise ValueError(f"Expected batch size of 1 from VAE decode, got {decoded_tensor_batched.shape[0]}")
            
            # Squeeze the batch dimension: (1, C, T, H, W) -> (C, T, H, W)
            decoded_tensor_cthw = decoded_tensor_batched.squeeze(0)
            print(f"Decoded tensor (squeezed) - shape: {decoded_tensor_cthw.shape}")

            # Permute back to (T, C, H, W)
            reconstructed_tensor = decoded_tensor_cthw.permute(1, 0, 2, 3)
            print(f"Reconstructed tensor (final) - shape: {reconstructed_tensor.shape}")
            
            return reconstructed_tensor
        except Exception as e:
            print(f"Error during VAE encode/decode: {e}")
            traceback.print_exc()
            return None

def tensor_to_pil_images(tensor_batch):
    tensor_batch = (tensor_batch.float().cpu() + 1) / 2
    tensor_batch = torch.clamp(tensor_batch, 0, 1)
    pil_images = []
    for tensor_img in tensor_batch:
        pil_images.append(transforms.ToPILImage()(tensor_img))
    return pil_images

def save_comparison_images(original_pils, reconstructed_pils, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving comparison images to: {output_dir}")
    
    # Attempt to load a default font, fallback to basic if not found
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for i, (orig_img, recon_img) in enumerate(zip(original_pils, reconstructed_pils)):
        if orig_img.size != recon_img.size:
             recon_img = recon_img.resize(orig_img.size)

        combined_width = orig_img.width + recon_img.width
        combined_height = orig_img.height
        
        combined_image = Image.new('RGB', (combined_width, combined_height + 40), (255, 255, 255)) # Add space for labels
        combined_image.paste(orig_img, (0, 40))
        combined_image.paste(recon_img, (orig_img.width, 40))
        
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 10), "Original", fill=(0,0,0), font=font)
        draw.text((orig_img.width + 10, 10), "Reconstructed", fill=(0,0,0), font=font)
        
        output_path = os.path.join(output_dir, f"comparison_frame_{i+1:03d}.png")
        combined_image.save(output_path)
    print(f"Saved {len(original_pils)} comparison images.")

def main():
    print("Starting VAE Reconstruction Test...")
    print(f"Using device: {device}")

    # --- VAE Model Path ---
    actual_vae_path = VAE_MODEL_PATH
    print(f"Attempting to use local VAE path: {actual_vae_path}")
    if not os.path.exists(actual_vae_path):
        print(f"ERROR: Local VAE path {actual_vae_path} not found. Please check VAE_MODEL_PATH.")
        return
    # --- End VAE Model Path ---

    vae = load_vae(actual_vae_path, device)
    if vae is None:
        print("Exiting due to VAE loading failure.")
        return

    original_frames_pil, input_tensor = load_and_preprocess_frames(
        EXAMPLE_VIDEO_PATH, NUM_TEST_FRAMES, IMG_HEIGHT, IMG_WIDTH
    )
    if original_frames_pil is None or input_tensor is None:
        print("Exiting due to frame loading or preprocessing failure.")
        return

    reconstructed_tensor = reconstruct_frames(vae, input_tensor, device)
    if reconstructed_tensor is None:
        print("Exiting due to reconstruction failure.")
        return

    reconstructed_frames_pil = tensor_to_pil_images(reconstructed_tensor)
    
    save_comparison_images(original_frames_pil, reconstructed_frames_pil, OUTPUT_DIR_FOR_COMPARISON_IMAGES)
    
    print("VAE Reconstruction Test Finished.")
    print("Please check the images in the '{OUTPUT_DIR_FOR_COMPARISON_IMAGES}' directory to assess reconstruction quality.")
    print("If reconstructions are poor (blurry, loss of detail), the VAE likely needs finetuning for your CT data.")

if __name__ == "__main__":
    main() 