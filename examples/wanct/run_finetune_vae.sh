#!/bin/bash

# Script to run VAE finetuning

# --- Configuration ---
DATASET_PATH="/path/to/your/ct_mp4_data" # FIXME: Update this path
VAE_PATH="/path/to/your/Wan2.1_VAE.pth"   # FIXME: Update this path
OUTPUT_PATH="./ct_vae_finetuned_$(date +%Y%m%d_%H%M%S)"

NUM_TRAIN_FRAMES=16
FRAME_INTERVAL=1
IMG_HEIGHT=256
IMG_WIDTH=256

LEARNING_RATE=1e-5
BETA_KLD=3.0e-6            # From WAN paper, Section 4.1.2
L1_LOSS_WEIGHT=3.0         # From WAN paper (reconstruction loss), Section 4.1.2. Set to 0.0 to disable.
LPIPS_LOSS_WEIGHT=3.0      # From WAN paper (perceptual loss), Section 4.1.2. Set to 0.0 to disable.

BATCH_SIZE_PER_GPU=2
MAX_EPOCHS=50              # Increased epochs for potentially better finetuning
DATALOADER_NUM_WORKERS=4

ACCELERATOR="gpu"
DEVICES="auto"             # Use 'auto' or specify (e.g., "0,1" for 2 GPUs, or 1 for a single GPU)
PRECISION_MODE="bf16-mixed"
STRATEGY="auto"            # 'ddp' if using multiple GPUs, 'auto' otherwise
SEED=42

# --- Activate Conda Environment (if needed) ---
# echo "Activating conda environment..."
# source /path/to/your/conda/etc/profile.d/conda.sh # FIXME: Update this path if you use conda
# conda activate your_env_name # FIXME: Update your conda environment name

# --- Run Training ---
echo "Starting VAE finetuning..."

python examples/wanct/finetune_ct_vae.py \
    --dataset_path "${DATASET_PATH}" \
    --vae_path "${VAE_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --num_train_frames ${NUM_TRAIN_FRAMES} \
    --frame_interval ${FRAME_INTERVAL} \
    --img_height ${IMG_HEIGHT} \
    --img_width ${IMG_WIDTH} \
    --learning_rate ${LEARNING_RATE} \
    --beta_kld ${BETA_KLD} \
    --l1_loss_weight ${L1_LOSS_WEIGHT} \
    --lpips_loss_weight ${LPIPS_LOSS_WEIGHT} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --max_epochs ${MAX_EPOCHS} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision_mode "${PRECISION_MODE}" \
    --strategy "${STRATEGY}" \
    --seed ${SEED}

echo "Finetuning finished. Checkpoints and logs are in ${OUTPUT_PATH}" 