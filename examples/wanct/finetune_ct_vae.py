import torch, os, imageio, argparse, random, glob
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd # Not strictly needed for CTVideoDataset but good to have if metadata is used later
from diffsynth import WanVideoPipeline, ModelManager, WanVideoVAE # Make sure WanVideoVAE is the correct class for your VAE model
import torchvision
from PIL import Image
import numpy as np
import torch.nn.functional as F # Added for L1 loss
import lpips # Added for LPIPS loss

# Default configuration values
DEFAULT_IMG_HEIGHT = 256  # Starting with a smaller default for VAE finetuning efficiency
DEFAULT_IMG_WIDTH = 256
DEFAULT_NUM_FRAMES = 16  # Number of frames per training clip
DEFAULT_BETA_KLD = 3.0e-6 # From WAN paper (Sec 4.1.2)
DEFAULT_L1_LOSS_WEIGHT = 0.0 # Default to 0 to make it optional, paper uses 3.0 for L1
DEFAULT_LPIPS_LOSS_WEIGHT = 0.0 # Default to 0 to make it optional, paper uses 3.0 for LPIPS

class CTVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, num_frames=DEFAULT_NUM_FRAMES, height=DEFAULT_IMG_HEIGHT, width=DEFAULT_IMG_WIDTH, frame_interval=1):
        self.base_path = base_path
        self.video_files = sorted(glob.glob(os.path.join(base_path, "**", "*.mp4"), recursive=True)) # Sort for consistency
        if not self.video_files:
            raise RuntimeError(f"No .mp4 files found in {base_path} or its subdirectories.")
        print(f"Found {len(self.video_files)} MP4 files in {base_path}.")

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_interval = frame_interval 

        self.transform = v2.Compose([
            v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames_pil = []
        try:
            with imageio.get_reader(video_path, 'ffmpeg') as reader: # Specify ffmpeg backend
                total_frames_in_video = 0
                try:
                    # imageio v3 uses 'fps', 'duration', 'nframes' in metadata
                    # imageio v2 used .count_frames()
                    if hasattr(reader, 'metadata'):
                        meta = reader.metadata()
                        if 'nframes' in meta:
                            total_frames_in_video = meta['nframes']
                        elif 'duration' in meta and 'fps' in meta and meta['fps'] > 0:
                             total_frames_in_video = int(meta['duration'] * meta['fps'])
                    if total_frames_in_video == 0 and hasattr(reader, 'count_frames'): # Fallback for older imageio
                         total_frames_in_video = reader.count_frames()
                    if isinstance(total_frames_in_video, float) and np.isinf(total_frames_in_video):
                         total_frames_in_video = 0 
                except Exception:
                    pass 

                if total_frames_in_video == 0 : # Still unknown or very short
                    print(f"Warning: Could not determine frame count or video too short for {video_path}. Attempting to read first {self.num_frames} frames.")
                    # Try to read sequentially
                    for i in range(self.num_frames):
                        try:
                            frame_np = reader.get_data(i)
                            frames_pil.append(Image.fromarray(frame_np).convert('RGB'))
                        except (IndexError, RuntimeError): # RuntimeError if stream ends
                            break # Stop if not enough frames
                    if len(frames_pil) < self.num_frames:
                        print(f"Warning: Video {video_path} has less than {self.num_frames} readable frames ({len(frames_pil)}). Skipping.")
                        return None
                else:
                    if total_frames_in_video < self.num_frames * self.frame_interval:
                        print(f"Warning: Video {video_path} has {total_frames_in_video} frames, less than required for {self.num_frames} with interval {self.frame_interval}. Skipping.")
                        return None
                    
                    max_start_frame = total_frames_in_video - (self.num_frames -1) * self.frame_interval -1
                    start_frame_idx = 0
                    if max_start_frame > 0 :
                        start_frame_idx = random.randint(0, max_start_frame)

                    for i in range(self.num_frames):
                        frame_idx_to_load = start_frame_idx + i * self.frame_interval
                        frame_np = reader.get_data(frame_idx_to_load)
                        frames_pil.append(Image.fromarray(frame_np).convert('RGB'))
            
            if not frames_pil or len(frames_pil) != self.num_frames:
                 print(f"Warning: Could not load desired {self.num_frames} frames from {video_path}. Loaded {len(frames_pil)}. Skipping.")
                 return None

            frames_tensor = torch.stack([self.transform(frame) for frame in frames_pil]) 
            video_tensor = frames_tensor.permute(1, 0, 2, 3) # C, T, H, W
            return {"video": video_tensor, "path": video_path}

        except Exception as e:
            print(f"Error processing video {video_path}: {e}. Skipping.")
            return None

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None 
    return torch.utils.data.dataloader.default_collate(batch)

class LightningVAEFinetune(pl.LightningModule):
    def __init__(
        self,
        vae_path: str,
        learning_rate: float = 1e-5,
        beta_kld: float = DEFAULT_BETA_KLD,
        l1_loss_weight: float = DEFAULT_L1_LOSS_WEIGHT,
        lpips_loss_weight: float = DEFAULT_LPIPS_LOSS_WEIGHT,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_manager = ModelManager(torch_dtype=torch.bfloat16) 
        model_manager.load_models_to_device([vae_path], "cpu")
        
        temp_pipe = WanVideoPipeline.from_model_manager(model_manager, device="cpu")
        if temp_pipe.vae is None or not isinstance(temp_pipe.vae, WanVideoVAE): # Ensure it's the expected type
            raise RuntimeError(f"Could not load WanVideoVAE from {vae_path} via WanVideoPipeline. Found: {type(temp_pipe.vae)}")
        self.vae = temp_pipe.vae
        self.vae.train() # Set VAE to training mode

        self.learning_rate = learning_rate
        self.beta_kld = beta_kld
        self.l1_loss_weight = l1_loss_weight
        self.lpips_loss_weight = lpips_loss_weight

        if self.lpips_loss_weight > 0:
            # LPIPS model will be automatically moved to the correct device by PyTorch Lightning
            # as it is an nn.Module assigned as an attribute.
            self.lpips_loss_fn = lpips.LPIPS(net='alex') 
        else:
            self.lpips_loss_fn = None

    def training_step(self, batch, batch_idx):
        if batch is None: 
            return None

        video_batch_cthw = batch["video"] 
        
        # Ensure correct dtype for VAE input (Lightning usually handles device)
        # self.vae.model is VideoVAE_ which contains parameters. Its dtype should be bfloat16.
        video_batch_cthw = video_batch_cthw.to(dtype=next(self.vae.model.parameters()).dtype)

        try:
            # VAE forward pass: self.vae.model is VideoVAE_
            # VideoVAE_.encoder expects (B, C, T, H, W)
            encoded_features = self.vae.model.encoder(video_batch_cthw) 
            intermediate_latents = self.vae.model.conv1(encoded_features)
            mu_unscaled, log_var = intermediate_latents.chunk(2, dim=1)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z_sampled_unscaled = mu_unscaled + eps * std
            
            # VideoVAE_.decode expects z_unscaled and scale factors
            reconstructed_video = self.vae.model.decode(z_sampled_unscaled, self.vae.scale)

        except Exception as e:
            self.print(f"Error in VAE forward pass during training_step: {e}")
            self.log("train_loss", 100.0, prog_bar=True, sync_dist=True)
            # Return a dummy loss tensor that requires grad to avoid issues with Lightning
            dummy_loss = torch.tensor(100.0, device=self.device, dtype=video_batch_cthw.dtype)
            dummy_loss.requires_grad_()
            return dummy_loss

        total_loss = 0.0
        
        # Reconstruction Loss (MSE is default, L1 is optional)
        reconstruction_loss_mse = F.mse_loss(reconstructed_video, video_batch_cthw)
        if self.l1_loss_weight > 0:
            reconstruction_loss_l1 = F.l1_loss(reconstructed_video, video_batch_cthw)
            total_loss += self.l1_loss_weight * reconstruction_loss_l1
            self.log("reconstruction_loss_l1", reconstruction_loss_l1, sync_dist=True, batch_size=video_batch_cthw.size(0))
        else: # If L1 is not used, MSE is the primary reconstruction loss component
            total_loss += reconstruction_loss_mse # Add MSE to total_loss if L1 is not active or if you want both

        self.log("reconstruction_loss_mse", reconstruction_loss_mse, sync_dist=True, batch_size=video_batch_cthw.size(0))

        # LPIPS Loss (optional)
        if self.lpips_loss_fn is not None and self.lpips_loss_weight > 0:
            # LPIPS expects input in range [-1, 1]. Our normalization in dataset should handle this.
            # Ensure inputs are on the same device as lpips_loss_fn
            lpips_loss_val = self.lpips_loss_fn(reconstructed_video, video_batch_cthw).mean()
            total_loss += self.lpips_loss_weight * lpips_loss_val
            self.log("lpips_loss", lpips_loss_val, sync_dist=True, batch_size=video_batch_cthw.size(0))
        
        kld_loss_terms = 1 + log_var - mu_unscaled.pow(2) - log_var.exp()
        kld_loss = -0.5 * torch.sum(kld_loss_terms, dim=list(range(1, mu_unscaled.ndim))) # Sum over all non-batch dims
        kld_loss = kld_loss.mean() # Average over batch

        total_loss += self.beta_kld * kld_loss

        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=video_batch_cthw.size(0))
        # self.log("reconstruction_loss", reconstruction_loss_mse, sync_dist=True, batch_size=video_batch_cthw.size(0)) # Now logging specific recon losses
        self.log("kld_loss", kld_loss, sync_dist=True, batch_size=video_batch_cthw.size(0))

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.learning_rate)
        # Consider adding a learning rate scheduler if needed
        return optimizer

def parse_args_finetune_vae():
    parser = argparse.ArgumentParser(description="Finetune a VAE on CT scan MP4 data.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing CT MP4 files.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the pre-trained VAE model.")
    parser.add_argument("--output_path", type=str, default="./vae_finetune_output", help="Path to save checkpoints and logs.")
    parser.add_argument("--num_train_frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames per training clip.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Interval between sampled frames.")
    parser.add_argument("--img_height", type=int, default=DEFAULT_IMG_HEIGHT, help="Target height for VAE input frames.")
    parser.add_argument("--img_width", type=int, default=DEFAULT_IMG_WIDTH, help="Target width for VAE input frames.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for VAE optimizer.")
    parser.add_argument("--beta_kld", type=float, default=DEFAULT_BETA_KLD, help=f"Weight for KL divergence term. Paper uses {DEFAULT_BETA_KLD}.")
    parser.add_argument("--l1_loss_weight", type=float, default=DEFAULT_L1_LOSS_WEIGHT, help="Weight for L1 reconstruction loss. Paper uses 3.0.")
    parser.add_argument("--lpips_loss_weight", type=float, default=DEFAULT_LPIPS_LOSS_WEIGHT, help="Weight for LPIPS perceptual loss. Paper uses 3.0.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size.") # Adjusted default
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "mps"], help="Accelerator.")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use (e.g., 'auto', 1, 4, '0,1').")
    parser.add_argument("--precision_mode", type=str, default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"], help="Training precision.")
    parser.add_argument("--strategy", type=str, default="auto", help="Distributed training strategy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    return args

def finetune_vae_entrypoint(args):
    pl.seed_everything(args.seed, workers=True)

    dataset = CTVideoDataset(
        base_path=args.dataset_path,
        num_frames=args.num_train_frames,
        height=args.img_height,
        width=args.img_width,
        frame_interval=args.frame_interval
    )
    
    if len(dataset) == 0:
        print("No videos loaded in dataset. Exiting.")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=True if args.dataloader_num_workers > 0 else False
    )

    model = LightningVAEFinetune(
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        beta_kld=args.beta_kld,
        l1_loss_weight=args.l1_loss_weight,
        lpips_loss_weight=args.lpips_loss_weight
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_path, "checkpoints"),
        filename="vae-ct-finetuned-{epoch:02d}-{train_loss:.4f}", # More precision in loss
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True
    )
    
    learning_rate_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')


    logger = pl.loggers.TensorBoardLogger(save_dir=args.output_path, name="logs_vae_finetune")
    
    devices_to_use = args.devices
    if args.accelerator == "gpu" and isinstance(devices_to_use, str) and devices_to_use.lower() != "auto":
        try:
            parsed_devices = [int(d.strip()) for d in devices_to_use.split(",")]
            devices_to_use = parsed_devices if len(parsed_devices) > 1 else parsed_devices[0]
        except ValueError:
            print(f"Warning: Could not parse devices '{args.devices}'. Using default/auto.")
            devices_to_use = "auto"
    elif args.accelerator == "gpu" and devices_to_use == "auto":
         devices_to_use = -1 # Means all available GPUs for pytorch lightning 'auto'

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices_to_use, 
        strategy=args.strategy if args.strategy != "auto" else "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto", # More robust DDP default
        max_epochs=args.max_epochs,
        precision=args.precision_mode,
        logger=logger,
        callbacks=[checkpoint_callback, learning_rate_monitor],
        default_root_dir=args.output_path,
    )

    print(f"Starting VAE finetuning with configuration: {vars(args)}")
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args_finetune_vae()
    finetune_vae_entrypoint(args) 