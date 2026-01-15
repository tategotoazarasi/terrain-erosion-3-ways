#!/usr/bin/python3
"""
Step 3 (Updated): Multi-Terrain AI Inference.

Changes:
1. Generates ALL 4 original terrain types (Plain, Standard, Ridge, Warped).
2. Removes water rendering from visualizations (pure terrain).
3. Packages Input/Output pairs into a TAR archive.

Note: Since the model was trained ONLY on Standard_FBM, apply it to
Ridge_Noise or Domain_Warping tests its "generalization" capability.
"""

import io
import os
import sys
import tarfile
import time

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib.colors import LightSource, LinearSegmentedColormap
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "erosion_unet_model.pth"
OUTPUT_TAR = "ai_erosion_multitype_results.tar"
NUM_SAMPLES = 64  # Divisible by 4 for balanced classes
IMG_SIZE = 512
DIMENSION = 512  # Needed for generation logic
SEED_SCALE_FACTOR = DIMENSION / 512.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# PART 1: Model Architecture (Matches Step 2)
# ==========================================

class DoubleConv(nn.Module):
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

	def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
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
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 1024)
		self.up1 = Up(1024 + 512, 512)
		self.up2 = Up(512 + 256, 256)
		self.up3 = Up(256 + 128, 128)
		self.up4 = Up(128 + 64, 64)
		self.outc = nn.Conv2d(64, n_channels, kernel_size=1)
		self.sigmoid = nn.Sigmoid()

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


# ==========================================
# PART 2: Original Generators (CuPy)
# ==========================================

def lerp(x, y, a): return (1.0 - a) * x + a * y


def normalize(x, bounds=(0, 1)):
	if not x.flags.c_contiguous: x = cp.ascontiguousarray(x)
	xp = cp.stack([x.min(), x.max()])
	fp = cp.asarray(bounds, dtype=x.dtype)
	return cp.interp(x, xp, fp).astype(x.dtype)


def fbm(shape, p, lower=-cp.inf, upper=cp.inf):
	"""Core noise function."""
	freqs = tuple(cp.fft.fftfreq(n, d=1.0 / n) for n in shape)
	freq_radial = cp.hypot(*cp.meshgrid(*freqs))
	envelope = (cp.power(freq_radial, p) * (freq_radial > lower) * (freq_radial < upper))
	envelope[0][0] = 0.0
	phase_noise = cp.exp(2j * cp.pi * cp.random.rand(*shape, dtype=cp.float32))
	complex_res = cp.fft.ifft2(cp.fft.fft2(phase_noise) * envelope)
	return normalize(cp.real(complex_res)).astype(cp.float16)


def sample(a, offset):
	"""Bilinear sampling with offset."""
	shape_tuple = a.shape
	shape_gpu = cp.array(shape_tuple)
	delta = cp.array((offset.real, offset.imag))
	grid_vectors = [cp.arange(n) for n in shape_tuple]
	coords = cp.array(cp.meshgrid(*grid_vectors)) - delta
	lower_coords = cp.floor(coords).astype(int)
	upper_coords = lower_coords + 1
	coord_offsets = (coords - lower_coords).astype(a.dtype)
	lower_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]
	upper_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]
	result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
	                   a[lower_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              lerp(a[upper_coords[1], lower_coords[0]],
	                   a[upper_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              coord_offsets[1])
	return result


# --- The 4 Generator Functions ---

def gen_plain_fbm(shape):
	return fbm(shape, -2, lower=2.0)


def gen_standard_fbm(shape):
	return fbm(shape, -2.0)


def gen_ridge_noise(shape):
	def noise_octave(s, f): return fbm(s, -1, lower=f, upper=(2 * f))

	values = cp.zeros(shape, dtype=cp.float16)
	for p in range(1, 12):
		a = 2 ** p
		values += cp.abs(noise_octave(shape, a) - 0.5) / a
	return (1.0 - normalize(values)) ** 2


def gen_domain_warping(shape):
	values = fbm(shape, -2, lower=2.0)
	noise_real = fbm(shape, -2, lower=1.5)
	noise_imag = fbm(shape, -2, lower=1.5)
	warp_scale = 150.0 * SEED_SCALE_FACTOR
	offsets = warp_scale * (noise_real + 1j * noise_imag)
	return sample(values, offsets).astype(cp.float16)


GENERATORS = [
	("Plain_FBM", gen_plain_fbm),
	("Standard_FBM", gen_standard_fbm),
	("Ridge_Noise", gen_ridge_noise),
	("Domain_Warping", gen_domain_warping)
]

# ==========================================
# PART 3: Visualization (No Water)
# ==========================================

_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
	(0.00, (0.15, 0.3, 0.15)),
	(0.25, (0.3, 0.45, 0.3)),
	(0.50, (0.5, 0.5, 0.35)),
	(0.80, (0.4, 0.36, 0.33)),
	(1.00, (1.0, 1.0, 1.0)),
])


def get_hillshaded_png_bytes(array_gpu):
	"""Pure terrain hillshade, no water masking."""
	# Move to CPU
	if hasattr(array_gpu, 'get'):
		arr_cpu = array_gpu.get().astype(np.float32)
	elif isinstance(array_gpu, torch.Tensor):
		arr_cpu = array_gpu.cpu().numpy().astype(np.float32)
	else:
		arr_cpu = array_gpu.astype(np.float32)

	# Normalize carefully
	arr_min, arr_max = arr_cpu.min(), arr_cpu.max()
	if arr_max > arr_min:
		arr_cpu = (arr_cpu - arr_min) / (arr_max - arr_min)

	# Hillshade
	ls = LightSource(azdeg=270, altdeg=30)
	# No blend_mode='overlay' with water mask anymore, just pure shading map
	rgb = ls.shade(arr_cpu, cmap=_TERRAIN_CMAP, vert_exag=10.0, blend_mode='soft')[:, :, :3]

	uint8_data = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

	with io.BytesIO() as output:
		Image.fromarray(uint8_data, mode='RGB').save(output, format="PNG", optimize=True)
		return output.getvalue()


# ==========================================
# PART 4: Main Logic
# ==========================================

def add_to_tar(tf, filename, data_bytes):
	info = tarfile.TarInfo(name=filename)
	info.size = len(data_bytes)
	info.mtime = time.time()
	with io.BytesIO(data_bytes) as f:
		tf.addfile(info, f)


def npy_to_bytes(arr):
	if hasattr(arr, 'get'): arr = arr.get()
	if isinstance(arr, torch.Tensor): arr = arr.cpu().numpy()
	with io.BytesIO() as f:
		np.save(f, arr)
		return f.getvalue()


def main():
	print("=== AI Terrain Erosion Inference (Multi-Type) ===")

	# 1. Setup
	try:
		cp.cuda.Device(0).use()
	except Exception as e:
		print(f"Error initializing CuPy: {e}");
		sys.exit(1)

	if not os.path.exists(MODEL_PATH):
		print(f"Error: Model file '{MODEL_PATH}' not found.");
		sys.exit(1)

	# 2. Load Model
	print(f"Loading model...")
	model = ErosionUNet().to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()

	# 3. Processing
	print(f"Generating {NUM_SAMPLES} samples ({NUM_SAMPLES // 4} per type)...")

	with tarfile.open(OUTPUT_TAR, "w") as tf:
		for i in tqdm(range(NUM_SAMPLES)):
			# Rotate through generators: 0, 1, 2, 3, 0, 1...
			gen_idx = i % len(GENERATORS)
			gen_name, gen_func = GENERATORS[gen_idx]

			sample_id = f"ai_{i:03d}_{gen_name}"

			# A. Generate (CuPy)
			input_gpu = gen_func((IMG_SIZE, IMG_SIZE))
			input_gpu = normalize(input_gpu)

			# B. Inference (PyTorch)
			input_tensor = torch.from_numpy(cp.asnumpy(input_gpu).astype(np.float32))
			input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

			with torch.no_grad():
				output_tensor = model(input_tensor)
				output_tensor = output_tensor.squeeze(0).squeeze(0)

			# C. Packaging
			# NPYs
			add_to_tar(tf, f"{sample_id}_input.npy", npy_to_bytes(input_gpu))
			add_to_tar(tf, f"{sample_id}_output.npy", npy_to_bytes(output_tensor))
			# PNGs (Hillshaded, No Water)
			add_to_tar(tf, f"{sample_id}_input_view.png", get_hillshaded_png_bytes(input_gpu))
			add_to_tar(tf, f"{sample_id}_output_view.png", get_hillshaded_png_bytes(output_tensor))

	print(f"\nDone! Results saved to: {OUTPUT_TAR}")


if __name__ == "__main__":
	main()
