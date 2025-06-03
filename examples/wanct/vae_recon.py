import torch
from torchvision import transforms
from diffsynth import ModelManager, WanVideoPipeline
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
import os
import traceback # Added for detailed error logging
import random # Added for random sampling
import nibabel as nib # Added for NIfTI file support
import torch.nn.functional as F # Added for trilinear interpolation

# --- Configuration --- 
# !!! UPDATE THESE PATHS AND SETTINGS !!!
VAE_MODEL_PATH = "/home/than/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth" # CHOOSE YOUR VAE

# --- Input Data: Choose NIFTI or VIDEO ---
INPUT_TYPE = "NIFTI" # "NIFTI" or "VIDEO"

# For NIFTI input
NIFTI_FILE_PATHS = [
    # Add paths to your NIfTI files here, e.g.:
    # "/path/to/your/scan1.nii.gz",
    # "/path/to/your/scan2.nii.gz",
]
SLICING_AXIS = 0 # Axis to slice along for the *final processed* NIfTI volume (0 for Depth/Z, 1 for Height/Y, 2 for Width/X).
# After preprocessing, volume is (Depth, Height, Width) corresponding to (Z,Y,X) resampled axes.

# NIFTI Preprocessing Parameters (adapted from nifty_mp4.py)
TARGET_X_SPACING = 0.75
TARGET_Y_SPACING = 0.75
TARGET_Z_SPACING = 1.5
NORMALIZED_BACKGROUND_PADDING_VALUE = -1.0 # Used for padding if data is in [-1,1] range

# Target shape for NIfTI volumes after resampling (Depth, Height, Width).
# Set to None to disable cropping/padding to a fixed shape.
# Example: TARGET_SHAPE_DHW = (128, 256, 256)
TARGET_SHAPE_DHW = None # (e.g., (160, 256, 256))

# For VIDEO input (original script's setting)
EXAMPLE_VIDEO_PATH = "examples/wanct/output/train_3_a_1_resampled_axis0.mp4" # Used if INPUT_TYPE is "VIDEO"

OUTPUT_DIR_FOR_COMPARISON_IMAGES = "vae_reconstruction_test_output"

# Device to run the VAE on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE input image size (consistent with train_wan_t2v.py defaults, adjust if necessary)
# This is the size slices will be resized to before VAE.
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Number of frames/slices to sample from the video/NIfTI volume for testing
NUM_TEST_FRAMES_OR_SLICES = 5
# --- End Configuration ---

# --- Helper functions from nifty_mp4.py (adapted) ---
def resize_volume_trilinear(volume_tensor, current_spacing, target_spacing):
    """
    Resize the 3D volume tensor to match the target spacing using trilinear interpolation.
    volume_tensor: input tensor of shape (1, 1, D, H, W) or (D,H,W)
    current_spacing: tuple of (spacing_dim0, spacing_dim1, spacing_dim2) for D, H, W respectively
    target_spacing: tuple of (target_spacing_dim0, target_spacing_dim1, target_spacing_dim2) for D, H, W respectively
    Returns: resampled numpy array of shape (D_new, H_new, W_new)
    """
    if volume_tensor.ndim == 3: # (D, H, W)
        # Add batch and channel dimensions for F.interpolate
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
    elif volume_tensor.ndim != 5 or volume_tensor.shape[0] != 1 or volume_tensor.shape[1] != 1:
        raise ValueError("volume_tensor must be (D,H,W) or (1,1,D,H,W)")

    original_shape_d_h_w = volume_tensor.shape[2:] # (D, H, W)
    
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(3)
    ]
    new_shape_d_h_w = [
        int(original_shape_d_h_w[i] * scaling_factors[i]) for i in range(3)
    ]
    # Ensure new shape dimensions are at least 1 to avoid errors with F.interpolate
    new_shape_d_h_w = [max(1, s) for s in new_shape_d_h_w]

    resampled_tensor = F.interpolate(
        volume_tensor.float(), # Ensure float for interpolation
        size=new_shape_d_h_w, 
        mode='trilinear', 
        align_corners=False # Usually False for medical imaging
    )
    return resampled_tensor.squeeze().cpu().numpy() # Back to (D_new, H_new, W_new) numpy array

def crop_or_pad_volume_to_target_shape(volume, target_shape_d_h_w, pad_value):
    """
    Crops or pads a 3D volume to a target shape (Depth, Height, Width).
    Assumes volume is a NumPy array with shape (D, H, W).
    Cropping is centered. Padding uses the specified pad_value.
    """
    current_shape = volume.shape
    target_d, target_h, target_w = target_shape_d_h_w

    # Calculate padding/cropping for depth (axis 0)
    d_diff = target_d - current_shape[0]
    d_pad_before = d_diff // 2 if d_diff > 0 else 0
    d_pad_after = d_diff - d_pad_before if d_diff > 0 else 0
    d_crop_start = -d_diff // 2 if d_diff < 0 else 0
    d_crop_end = current_shape[0] - (-d_diff - d_crop_start) if d_diff < 0 else current_shape[0]

    # Calculate padding/cropping for height (axis 1)
    h_diff = target_h - current_shape[1]
    h_pad_before = h_diff // 2 if h_diff > 0 else 0
    h_pad_after = h_diff - h_pad_before if h_diff > 0 else 0
    h_crop_start = -h_diff // 2 if h_diff < 0 else 0
    h_crop_end = current_shape[1] - (-h_diff - h_crop_start) if h_diff < 0 else current_shape[1]

    # Calculate padding/cropping for width (axis 2)
    w_diff = target_w - current_shape[2]
    w_pad_before = w_diff // 2 if w_diff > 0 else 0
    w_pad_after = w_diff - w_pad_before if w_diff > 0 else 0
    w_crop_start = -w_diff // 2 if w_diff < 0 else 0
    w_crop_end = current_shape[2] - (-w_diff - w_crop_start) if w_diff < 0 else current_shape[2]

    # Apply cropping first
    cropped_volume = volume[d_crop_start:d_crop_end, h_crop_start:h_crop_end, w_crop_start:w_crop_end]

    # Apply padding if needed
    if d_diff > 0 or h_diff > 0 or w_diff > 0:
        padded_volume = np.pad(
            cropped_volume, 
            ((d_pad_before, d_pad_after), (h_pad_before, h_pad_after), (w_pad_before, w_pad_after)),
            mode='constant', 
            constant_values=pad_value
        )
        return padded_volume
    else:
        return cropped_volume
# --- End Helper functions ---

class NiftiSliceDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, num_slices_to_sample, 
                 img_height_for_vae, img_width_for_vae, 
                 slicing_axis_for_final_volume=0, 
                 target_shape_dhw=None, 
                 base_transform=None):
        """
        Args:
            file_paths (list of str): List of paths to NIfTI files.
            num_slices_to_sample (int): Number of slices to sample from each NIfTI volume.
            img_height_for_vae (int): Target height for slices fed to VAE (after PIL resize).
            img_width_for_vae (int): Target width for slices fed to VAE (after PIL resize).
            slicing_axis_for_final_volume (int): Axis of the *final processed volume* to slice along (0:Depth, 1:Height, 2:Width).
            target_shape_dhw (tuple or None): Target (D,H,W) for volume after resampling. If None, no cropping/padding.
            base_transform (callable, optional): Transformations for VAE input (ToTensor, Normalize).
        """
        self.file_paths = file_paths
        self.num_slices_to_sample = num_slices_to_sample
        self.img_height_for_vae = img_height_for_vae
        self.img_width_for_vae = img_width_for_vae
        self.slicing_axis_for_final_volume = slicing_axis_for_final_volume
        self.target_shape_dhw = target_shape_dhw
        self.base_transform = base_transform

        if self.base_transform is None:
            self.base_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nifti_path = self.file_paths[idx]
        
        try:
            img_nifti = nib.load(nifti_path)
            data = img_nifti.get_fdata().astype(np.float32)

            # 1. Apply slope and intercept if available
            scl_slope, scl_inter = img_nifti.header.get_slope_inter()
            if scl_slope is None: scl_slope = 1.0
            if scl_inter is None: scl_inter = 0.0
            if scl_slope != 0: # Avoid issues with slope being exactly 0
                 data = data * scl_slope + scl_inter
            else: # if slope is 0, it's likely an issue or a very specific case. Use intercept only.
                 data = np.full_like(data, scl_inter)


            # 2. HU Clipping
            hu_min, hu_max = -1000, 1000
            data_clipped = np.clip(data, hu_min, hu_max)

            # 3. Normalize HU-clipped data to [-1.0, 1.0]
            # (data - hu_min) / (hu_max - hu_min) maps to [0,1]
            # then * 2.0 - 1.0 maps to [-1,1]
            if (hu_max - hu_min) > 1e-5: # Avoid division by zero
                volume_norm_minus1_to_1 = ((data_clipped - hu_min) / (hu_max - hu_min)) * 2.0 - 1.0
            else: # If range is too small, set to mid-value (0 in [-1,1] scale) or padding value
                volume_norm_minus1_to_1 = np.full_like(data_clipped, 0.0)


            # 4. Transpose to (Depth, Height, Width) convention for resampling (e.g., Z, Y, X)
            # NIfTI data is often (X, Y, Z). get_zooms() gives (x_zoom, y_zoom, z_zoom).
            # We transpose to (Z, Y, X) to match target_spacing (TARGET_Z_SPACING, TARGET_Y_SPACING, TARGET_X_SPACING)
            volume_transposed = volume_norm_minus1_to_1.transpose(2, 1, 0) # From (X,Y,Z) to (Z,Y,X)
            
            # Original spacings corresponding to (X,Y,Z)
            x_zoom, y_zoom, z_zoom = img_nifti.header.get_zooms()
            # Spacings for the transposed (Z,Y,X) volume
            current_spacing_for_resample = (z_zoom, y_zoom, x_zoom)
            target_spacing_for_resample = (TARGET_Z_SPACING, TARGET_Y_SPACING, TARGET_X_SPACING)

            # 5. Resample to target spacing
            volume_tensor_for_resample = torch.from_numpy(volume_transposed.copy()) # Ensure it's a new tensor for unsqueeze
            resampled_volume = resize_volume_trilinear(
                volume_tensor_for_resample, 
                current_spacing_for_resample, 
                target_spacing_for_resample
            ) # Output is numpy array, still in [-1,1] range

            # 6. Crop or Pad to Target Shape (if specified)
            final_volume_processed_minus1_to_1 = resampled_volume
            if self.target_shape_dhw is not None:
                final_volume_processed_minus1_to_1 = crop_or_pad_volume_to_target_shape(
                    resampled_volume, 
                    self.target_shape_dhw, 
                    NORMALIZED_BACKGROUND_PADDING_VALUE # Padding value is -1.0
                )
            
            # 7. Convert final processed volume from [-1,1] to [0,1] for PIL conversion and subsequent transforms
            final_volume_0_to_1 = (final_volume_processed_minus1_to_1 + 1.0) / 2.0

        except Exception as e:
            print(f"Error preprocessing NIfTI file {nifti_path}: {e}")
            traceback.print_exc()
            return None, None 

        # Check dimensions after processing
        if final_volume_0_to_1.ndim != 3:
            print(f"Error: Processed NIfTI data for {nifti_path} is not 3D (shape: {final_volume_0_to_1.shape}). Skipping.")
            return None, None

        num_available_slices = final_volume_0_to_1.shape[self.slicing_axis_for_final_volume]

        if num_available_slices == 0:
            print(f"Error: Processed NIfTI {nifti_path} has 0 slices along axis {self.slicing_axis_for_final_volume}. Skipping.")
            return None, None

        slice_indices = []
        if num_available_slices >= self.num_slices_to_sample:
            # Ensure indices are spread out if possible, or just random sample
            # For simplicity, using random.sample. For more even spread, use np.linspace then int.
            slice_indices = sorted(random.sample(range(num_available_slices), self.num_slices_to_sample))
        else:
            slice_indices = sorted(random.choices(range(num_available_slices), k=self.num_slices_to_sample))
            print(f"Warning: Requested {self.num_slices_to_sample} slices from processed {nifti_path} (axis {self.slicing_axis_for_final_volume} has {num_available_slices}). Sampling with replacement.")

        processed_tensor_slices = []
        display_pil_slices = []

        for slice_idx in slice_indices:
            if self.slicing_axis_for_final_volume == 0: # Depth (Z)
                slice_np_0_to_1 = final_volume_0_to_1[slice_idx, :, :]
            elif self.slicing_axis_for_final_volume == 1: # Height (Y)
                slice_np_0_to_1 = final_volume_0_to_1[:, slice_idx, :]
            elif self.slicing_axis_for_final_volume == 2: # Width (X)
                slice_np_0_to_1 = final_volume_0_to_1[:, :, slice_idx]
            else:
                raise ValueError(f"Invalid slicing_axis_for_final_volume: {self.slicing_axis_for_final_volume}")

            # Slice is in [0,1] float. Convert to uint8 [0,255] for PIL
            slice_uint8 = (slice_np_0_to_1 * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(slice_uint8, mode='L')
            pil_img_rgb = pil_img.convert('RGB')

            # Resize for VAE input and display
            pil_resized_rgb = pil_img_rgb.resize((self.img_width_for_vae, self.img_height_for_vae), Image.BILINEAR)
            
            display_pil_slices.append(pil_resized_rgb)

            # Apply base transformations (ToTensor, Normalize to [-1,1]) for VAE
            tensor_slice = self.base_transform(pil_resized_rgb) # Expects PIL, outputs tensor normalized to [-1,1]
            processed_tensor_slices.append(tensor_slice)
        
        if not processed_tensor_slices:
            print(f"Error: No slices processed for {nifti_path}. This is unexpected.")
            return None, None
            
        input_tensor = torch.stack(processed_tensor_slices) # (num_slices, C, H, W)
        return input_tensor, display_pil_slices


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

    if INPUT_TYPE == "NIFTI":
        if not NIFTI_FILE_PATHS:
            print("ERROR: INPUT_TYPE is NIFTI, but NIFTI_FILE_PATHS is empty. Please provide NIfTI file paths.")
            return
        
        print(f"Processing NIfTI files from: {NIFTI_FILE_PATHS}")
        print(f"NIfTI settings: Target Spacing (Z,Y,X): ({TARGET_Z_SPACING},{TARGET_Y_SPACING},{TARGET_X_SPACING}), Target Shape (D,H,W): {TARGET_SHAPE_DHW}, Slicing Axis for final volume: {SLICING_AXIS}")

        dataset = NiftiSliceDataset(
            file_paths=NIFTI_FILE_PATHS,
            num_slices_to_sample=NUM_TEST_FRAMES_OR_SLICES,
            img_height_for_vae=IMG_HEIGHT,
            img_width_for_vae=IMG_WIDTH,
            slicing_axis_for_final_volume=SLICING_AXIS,
            target_shape_dhw=TARGET_SHAPE_DHW
            # base_transform is defaulted in the class
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 for easier debugging

        for i, data_batch in enumerate(dataloader):
            # Check if dataset __getitem__ returned None due to error
            if data_batch[0] is None or data_batch[1] is None: 
                original_nifti_path = dataset.file_paths[i] # Get path using index if dataloader preserves order
                print(f"Skipping NIfTI file (originally {original_nifti_path} at index {i}) due to loading/preprocessing error in dataset.")
                continue
            
            input_tensor_item, original_frames_pil_item_list = data_batch
            
            input_tensor = input_tensor_item.squeeze(0) 
            original_frames_pil = original_frames_pil_item_list[0] 

            current_nifti_path = dataset.file_paths[i] # Get path based on current index in dataloader
            nifti_filename_base = os.path.basename(current_nifti_path)
            nifti_filename_stem = os.path.splitext(os.path.splitext(nifti_filename_base)[0])[0] 
            
            output_dir_for_file = os.path.join(OUTPUT_DIR_FOR_COMPARISON_IMAGES, f"nifti_{nifti_filename_stem}")
            print(f"Processing NIfTI file: {current_nifti_path} (Outputting to: {output_dir_for_file})")

            reconstructed_tensor = reconstruct_frames(vae, input_tensor, device)
            if reconstructed_tensor is None:
                print(f"Reconstruction failed for {current_nifti_path}. Skipping.")
                continue

            reconstructed_frames_pil = tensor_to_pil_images(reconstructed_tensor)
            save_comparison_images(original_frames_pil, reconstructed_frames_pil, output_dir_for_file)
            print(f"Finished processing {current_nifti_path}. Check results in {output_dir_for_file}")

    elif INPUT_TYPE == "VIDEO":
        print(f"Processing Video file: {EXAMPLE_VIDEO_PATH}")
        original_frames_pil, input_tensor = load_and_preprocess_frames(
            EXAMPLE_VIDEO_PATH, NUM_TEST_FRAMES_OR_SLICES, IMG_HEIGHT, IMG_WIDTH
        )
        if original_frames_pil is None or input_tensor is None:
            print("Exiting due to frame loading or preprocessing failure.")
            return

        output_dir_for_video = os.path.join(OUTPUT_DIR_FOR_COMPARISON_IMAGES, "video_reconstruction")
        reconstructed_tensor = reconstruct_frames(vae, input_tensor, device)
        if reconstructed_tensor is None:
            print("Exiting due to reconstruction failure.")
            return

        reconstructed_frames_pil = tensor_to_pil_images(reconstructed_tensor)
        save_comparison_images(original_frames_pil, reconstructed_frames_pil, output_dir_for_video)
        print(f"Finished processing video. Check results in {output_dir_for_video}")

    else:
        print(f"ERROR: Unknown INPUT_TYPE: {INPUT_TYPE}. Choose 'NIFTI' or 'VIDEO'.")
        return
    
    print("VAE Reconstruction Test Finished for all inputs.")
    print(f"Please check the images in the subdirectories of '{OUTPUT_DIR_FOR_COMPARISON_IMAGES}' to assess reconstruction quality.")

if __name__ == "__main__":
    main() 