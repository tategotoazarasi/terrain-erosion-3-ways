#!/usr/bin/python3
"""
Step 2: Erosion Model Training (U-Net).

Task: Image-to-Image Translation (Heightmap -> Eroded Heightmap).
Model: U-Net with Residual blocks (optional) or standard DoubleConv.
Loss: L1 Loss (Better for preserving sharp ridges than MSE).

Usage:
    python3 step2_train_model.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
	"data_dir"     : "dataset_standard_fbm",
	"img_size"     : 512,
	"batch_size"   : 8,  # Adjust based on VRAM (8 fits comfortably on 16GB VRAM)
	"epochs"       : 50,
	"learning_rate": 1e-4,
	"num_workers"  : 4,
	"seed"         : 42,
	"save_path"    : "erosion_unet_model.pth",
	"vis_dir"      : "training_visuals"
}


# --- Utils ---
def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


# --- Dataset Definition ---
class TerrainDataset(Dataset):
	def __init__(self, root_dir, transform=True):
		self.input_dir = os.path.join(root_dir, "inputs")
		self.target_dir = os.path.join(root_dir, "targets")

		# Get all run IDs based on files present in inputs
		self.filenames = [f for f in os.listdir(self.input_dir) if f.endswith('.npy')]

		# Verify matching targets exist
		self.valid_filenames = []
		for f in self.filenames:
			if os.path.exists(os.path.join(self.target_dir, f)):
				self.valid_filenames.append(f)

		print(f"Dataset initialized: {len(self.valid_filenames)} pairs found.")
		self.transform = transform

	def __len__(self):
		return len(self.valid_filenames)

	def __getitem__(self, idx):
		fname = self.valid_filenames[idx]

		# Load NPY (already float32 from Step 1)
		# Add channel dimension: (H, W) -> (1, H, W)
		input_map = np.load(os.path.join(self.input_dir, fname))[np.newaxis, ...]
		target_map = np.load(os.path.join(self.target_dir, fname))[np.newaxis, ...]

		# Data Augmentation (Random Flips/Rotations)
		# Terrain erosion is rotation invariant (physics doesn't change with orientation)
		if self.transform:
			# Random Horizontal Flip
			if random.random() > 0.5:
				input_map = np.flip(input_map, axis=2).copy()
				target_map = np.flip(target_map, axis=2).copy()

			# Random Vertical Flip
			if random.random() > 0.5:
				input_map = np.flip(input_map, axis=1).copy()
				target_map = np.flip(target_map, axis=1).copy()

			# Random Rotation (0, 90, 180, 270)
			k = random.randint(0, 3)
			if k > 0:
				input_map = np.rot90(input_map, k, axes=(1, 2)).copy()
				target_map = np.rot90(target_map, k, axes=(1, 2)).copy()

		# Convert to Tensor
		return torch.from_numpy(input_map), torch.from_numpy(target_map)


# --- Model Architecture: U-Net ---
class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# Input is CHW. x2 is skip connection.
		# Handling padding if dimensions don't match exactly (though 512 is power of 2, so it's safe)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		if diffX > 0 or diffY > 0:
			x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
			                            diffY // 2, diffY - diffY // 2])

		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class ErosionUNet(nn.Module):
	def __init__(self, n_channels=1):
		super(ErosionUNet, self).__init__()
		self.n_channels = n_channels

		# Encoder
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 1024)

		# Decoder
		self.up1 = Up(1024 + 512, 512)
		self.up2 = Up(512 + 256, 256)
		self.up3 = Up(256 + 128, 128)
		self.up4 = Up(128 + 64, 64)

		# Output
		self.outc = nn.Conv2d(64, n_channels, kernel_size=1)
		self.sigmoid = nn.Sigmoid()  # Ensure output is in [0, 1] as inputs are normalized

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return self.sigmoid(logits)


# --- Visualization ---
def save_sample_prediction(model, val_loader, device, epoch, save_dir):
	"""Saves a visual comparison: Input | Ground Truth | Prediction"""
	model.eval()
	os.makedirs(save_dir, exist_ok=True)

	# Get one batch
	inputs, targets = next(iter(val_loader))
	inputs = inputs.to(device)

	with torch.no_grad():
		preds = model(inputs)

	# Take first image
	img_in = inputs[0, 0].cpu().numpy()
	img_target = targets[0, 0].cpu().numpy()
	img_pred = preds[0, 0].cpu().numpy()

	# Plot
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(img_in, cmap='terrain')
	axes[0].set_title("Input (Before)")
	axes[0].axis('off')

	axes[1].imshow(img_target, cmap='terrain')
	axes[1].set_title("Target (Simulated)")
	axes[1].axis('off')

	axes[2].imshow(img_pred, cmap='terrain')
	axes[2].set_title("Predicted (AI)")
	axes[2].axis('off')

	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
	plt.close()
	model.train()


# --- Main Training Loop ---
def main():
	set_seed(CONFIG["seed"])

	# 1. Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# 2. Data
	full_dataset = TerrainDataset(CONFIG["data_dir"], transform=True)

	# Split Train/Val (90/10)
	val_size = int(0.1 * len(full_dataset))
	train_size = len(full_dataset) - val_size
	train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

	# Disable augmentation for validation dataset?
	# It's hard to disable via random_split because it splits the dataset object.
	# But since erosion is invariant, validating on augmented data is acceptable.

	train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
	                          shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
	                        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

	print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

	# 3. Model
	model = ErosionUNet().to(device)

	# 4. Optimization
	# L1 Loss usually produces sharper results for terrain than MSE
	criterion = nn.L1Loss()
	optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
	scaler = GradScaler()  # For Mixed Precision

	# 5. Training Loop
	best_val_loss = float('inf')

	print("Starting training...")
	for epoch in range(1, CONFIG["epochs"] + 1):
		model.train()
		train_loss = 0.0

		loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", leave=False)

		for inputs, targets in loop:
			inputs = inputs.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)

			# Forward (with Autocast)
			with autocast():
				preds = model(inputs)
				loss = criterion(preds, targets)

			# Backward
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_loss += loss.item()
			loop.set_postfix(loss=loss.item())

		avg_train_loss = train_loss / len(train_loader)

		# Validation
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for inputs, targets in val_loader:
				inputs = inputs.to(device)
				targets = targets.to(device)
				preds = model(inputs)
				loss = criterion(preds, targets)
				val_loss += loss.item()

		avg_val_loss = val_loss / len(val_loader)

		# Scheduler Step
		scheduler.step()

		# Print stats
		print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

		# Save Snapshot
		if epoch % 5 == 0 or epoch == 1:
			save_sample_prediction(model, val_loader, device, epoch, CONFIG["vis_dir"])

		# Save Best Model
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(model.state_dict(), CONFIG["save_path"])
			print(f"  -> Model saved (Val Loss improved).")

	print("Training Complete.")
	print(f"Best Validation Loss: {best_val_loss}")
	print(f"Model saved to: {CONFIG['save_path']}")


if __name__ == "__main__":
	main()
