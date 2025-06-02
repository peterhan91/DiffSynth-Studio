from huggingface_hub import hf_hub_download
from datasets import load_dataset

def download_ct_rate_data(split='train', start_at=0, hf_token=None):
    repo_id = 'ibrahimhamamci/CT-RATE'

    data_labels = load_dataset(repo_id, 'labels', split=split, trust_remote_code=True)

    volume_names = data_labels['VolumeName']
    if start_at > 0:
        volume_names = volume_names[start_at:]

    for name in volume_names: # Removed tqdm wrapper
        parts = name.split('_')
        # Construct the subfolder path as per the notebook's logic
        # e.g., train_1_a_1.nii.gz -> dataset/train/train_1/train_1_a/
        folder_name = f"{parts[0]}_{parts[1]}" # e.g., train_1
        subfolder_name = f"{folder_name}_{parts[2]}" # e.g., train_1_a
        
        # This is the subfolder path within the Hugging Face repository structure
        hf_subfolder_path = f"dataset/{split}/{folder_name}/{subfolder_name}"
        
        # hf_hub_download will create the necessary subdirectories under 'data' based on hf_subfolder_path
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                token=hf_token,
                subfolder=hf_subfolder_path, 
                filename=name,
                cache_dir='./cache', 
                local_dir='data', 
                # local_dir_use_symlinks=False, # Deprecated and ignored
                resume_download=True,
            )
        except Exception as e:
            print(f"Error downloading {name}: {e}")



if __name__ == '__main__':
    download_ct_rate_data(split='train', start_at=0, hf_token=None)
    download_ct_rate_data(split='valid', start_at=0, hf_token=None)