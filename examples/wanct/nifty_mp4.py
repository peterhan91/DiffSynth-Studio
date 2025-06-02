import os
import argparse
import nibabel as nib
import numpy as np
import imageio
import concurrent.futures
import pandas as pd
import torch
import torch.nn.functional as F
import ast
from PIL import Image

# Fixed target spacing as per process_example_test.py
TARGET_X_SPACING = 0.75
TARGET_Y_SPACING = 0.75
TARGET_Z_SPACING = 1.5

# Padding value for normalized CT data (air/background)
NORMALIZED_BACKGROUND_PADDING_VALUE = -1.0 

def crop_or_pad_volume_to_target_shape(volume, target_shape_d_h_w):
    """
    Crops or pads a 3D volume to a target shape (Depth, Height, Width).
    Assumes volume is a NumPy array with shape (D, H, W).
    Cropping is centered. Padding uses NORMALIZED_BACKGROUND_PADDING_VALUE.
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
            constant_values=NORMALIZED_BACKGROUND_PADDING_VALUE
        )
        return padded_volume
    else:
        return cropped_volume

def resize_volume_trilinear(volume_tensor, current_spacing, target_spacing):
    """
    Resize the 3D volume tensor to match the target spacing using trilinear interpolation.
    volume_tensor: input tensor of shape (1, 1, D, H, W) or (D,H,W)
    current_spacing: tuple of (z_spacing_current, y_spacing_current, x_spacing_current)
    target_spacing: tuple of (z_spacing_target, y_spacing_target, x_spacing_target)
    Returns: resampled numpy array of shape (D_new, H_new, W_new)
    """
    if volume_tensor.ndim == 3: # (D, H, W)
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
        volume_tensor.float(), 
        size=new_shape_d_h_w, 
        mode='trilinear', 
        align_corners=False
    )
    return resampled_tensor.squeeze().cpu().numpy()

def process_nifti_file_wrapper(args_tuple):
    """
    Wrapper function to unpack arguments for ProcessPoolExecutor.map
    """
    nifti_path, output_path, fps, video_slice_axis_from_resampled, df_metadata, file_base_name, target_shape_for_video = args_tuple
    process_nifti_file(nifti_path, output_path, fps, video_slice_axis_from_resampled, df_metadata, file_base_name, target_shape_for_video)

def process_nifti_file(nifti_path, output_path, fps, video_slice_axis_from_resampled, df_metadata, file_base_name, target_shape_for_video):
    """
    Processes a single NIfTI file, resamples it, crops/pads, and saves it as an MP4 video.
    video_slice_axis_from_resampled: axis of the *cropped/padded* volume to slice for video (0 for Z, 1 for Y, 2 for X if resampled is Z,Y,X)
    target_shape_for_video: tuple (D,H,W) for the final volume shape before slicing, or None.
    """
    try:
        print(f"Processing {nifti_path}...")
        img_nib = nib.load(nifti_path)
        img_data_original_orientation = img_nib.get_fdata()

        if img_data_original_orientation.ndim != 3:
            print(f"Skipping {nifti_path}: Expected 3D data, got {img_data_original_orientation.ndim}D.")
            return

        row = df_metadata[df_metadata['VolumeName'] == file_base_name + ".nii.gz"]
        if row.empty:
            row = df_metadata[df_metadata['VolumeName'] == file_base_name]
        if row.empty:
            print(f"Skipping {nifti_path}: Metadata not found for {file_base_name}")
            return

        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        try:
            xy_spacing_str = str(row["XYSpacing"].iloc[0])
            if "[" in xy_spacing_str:
                xy_spacing = float(ast.literal_eval(xy_spacing_str)[0])
            else:
                xy_spacing = float(xy_spacing_str)
            z_spacing = float(row["ZSpacing"].iloc[0])
        except Exception as e:
            print(f"Skipping {nifti_path}: Error parsing spacing metadata for {file_base_name}: {e}")
            return
        
        processed_data = slope * img_data_original_orientation + intercept
        hu_min, hu_max = -1000, 1000
        processed_data = np.clip(processed_data, hu_min, hu_max)
        processed_data = (processed_data / 1000.0).astype(np.float32)

        data_transposed_for_resample = processed_data.transpose(2, 1, 0) 
        current_spacing_tuple = (z_spacing, xy_spacing, xy_spacing)
        target_spacing_tuple = (TARGET_Z_SPACING, TARGET_Y_SPACING, TARGET_X_SPACING)
        
        volume_tensor = torch.from_numpy(data_transposed_for_resample.copy()).unsqueeze(0).unsqueeze(0)
        resampled_volume_np = resize_volume_trilinear(volume_tensor, current_spacing_tuple, target_spacing_tuple)

        # --- Crop or Pad to Target Shape for Video ---
        if target_shape_for_video is not None:
            print(f"Cropping/padding resampled volume from {resampled_volume_np.shape} to {target_shape_for_video} for {nifti_path}")
            final_volume_for_video = crop_or_pad_volume_to_target_shape(resampled_volume_np, target_shape_for_video)
        else:
            final_volume_for_video = resampled_volume_np
        print(f"Final volume shape for video: {final_volume_for_video.shape} for {nifti_path}")

        # --- Video Creation from Final Volume ---
        num_slices_in_video_dim = final_volume_for_video.shape[video_slice_axis_from_resampled]
        if num_slices_in_video_dim == 0:
            print(f"Skipping {nifti_path}: No slices in final volume along axis {video_slice_axis_from_resampled}.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
        
        actual_frames_written = 0
        for i in range(num_slices_in_video_dim):
            if video_slice_axis_from_resampled == 0:
                slice_2d = final_volume_for_video[i, :, :]
            elif video_slice_axis_from_resampled == 1:
                slice_2d = final_volume_for_video[:, i, :]
            elif video_slice_axis_from_resampled == 2:
                slice_2d = final_volume_for_video[:, :, i]
            else:
                print(f"Skipping {nifti_path}: Invalid video_slice_axis_from_resampled {video_slice_axis_from_resampled}.")
                video_writer.close()
                return

            # Debug: print the shape
            print(f"Slice {i} shape: {slice_2d.shape}")

            slice_uint8 = ((slice_2d + 1.0) / 2.0 * 255.0).astype(np.uint8)
            print(f"slice_uint8 shape before stacking: {slice_uint8.shape}, dtype: {slice_uint8.dtype}")
            if slice_uint8.ndim != 2:
                print(f"Skipping slice {i} due to unexpected shape after conversion: {slice_uint8.shape}")
                continue
            slice_rgb = np.stack((slice_uint8,) * 3, axis=-1)
            slice_rgb = np.ascontiguousarray(slice_rgb, dtype=np.uint8)
            img_rgb = Image.fromarray(slice_rgb, mode='RGB')
            video_writer.append_data(np.array(img_rgb))
            actual_frames_written += 1
        
        video_writer.close()
        print(f"Successfully converted {nifti_path} to {output_path} with {actual_frames_written} frames.")

    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")
        import traceback
        traceback.print_exc() 

def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI CT scans to MP4 videos with 3D resampling, HU normalization, and optional fixed shaping.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing NIfTI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save MP4 videos.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to CSV metadata file.")
    parser.add_argument("--fps", type=int, default=10, help="FPS for output video.")
    parser.add_argument("--video_slice_axis", type=int, default=0, choices=[0, 1, 2], help="Axis to slice for video. Default: 0 (Z-axis).")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes.")
    parser.add_argument("--target_depth", type=int, help="Target depth (D) for the final volume.")
    parser.add_argument("--target_height", type=int, help="Target height (H) for the final volume.")
    parser.add_argument("--target_width", type=int, help="Target width (W) for the final volume.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return
    if not os.path.isfile(args.metadata_csv):
        print(f"Error: Metadata CSV file '{args.metadata_csv}' not found.")
        return

    # Target shape logic
    if all(v is not None for v in [args.target_depth, args.target_height, args.target_width]):
        target_shape = (args.target_depth, args.target_height, args.target_width)
        print(f"Will crop/pad final volumes to target shape: {target_shape}")
    elif any(v is not None for v in [args.target_depth, args.target_height, args.target_width]):
        print("Error: All target shape arguments must be specified together.")
        return
    else:
        target_shape = None

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df_metadata = pd.read_csv(args.metadata_csv)
    except Exception as e:
        print(f"Error reading metadata CSV: {e}")
        return

    # Gather NIfTI files
    tasks = []
    for fname in os.listdir(args.input_dir):
        if fname.endswith((".nii", ".nii.gz")):
            base = fname
            if fname.endswith('.nii.gz'):
                base = fname[:-7]
            elif fname.endswith('.nii'):
                base = fname[:-4]
            out_name = f"{base}_resampled_axis{args.video_slice_axis}.mp4"
            if target_shape:
                out_name = f"{base}_shape{'_'.join(map(str, target_shape))}_axis{args.video_slice_axis}.mp4"
            tasks.append((
                os.path.join(args.input_dir, fname),
                os.path.join(args.output_dir, out_name),
                args.fps,
                args.video_slice_axis,
                df_metadata,
                base,
                target_shape
            ))

    if not tasks:
        print("No NIfTI files found in the input directory.")
        return

    print(f"Found {len(tasks)} NIfTI files to process using up to {args.num_workers} workers.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(executor.map(process_nifti_file_wrapper, tasks))

    print("All NIfTI files processed.")

if __name__ == "__main__":
    main() 