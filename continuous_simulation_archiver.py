#!/usr/bin/python3

"""
Continuous Terrain Simulation 8K (Interrupt-Safe TAR Version).

KEY CHANGE: Uses .tar format instead of .zip.
Reason: ZIP files require a footer to be valid. If the script is killed
before the footer is written, the whole archive is corrupt.
TAR files are stream-based. If the script is killed, all files written
up to that point are perfectly valid.

Usage:
    python3 continuous_simulation_8k_safe.py [output_filename.tar]
"""

import io
import os
import signal
import sys
import tarfile
import time
from math import sqrt

import cupy as cp
import numpy as np
from PIL import Image

import util

# Global flag
KEEP_RUNNING = True

# Configuration
DIMENSION = 8192
SEED_SCALE_FACTOR = DIMENSION / 512.0


def signal_handler(sig, frame):
	global KEEP_RUNNING
	print('\n[!] Signal received. Stopping after current step...')
	KEEP_RUNNING = False


# --- TAR Helper Functions ---

def add_bytes_to_tar(tf, filename, data_bytes):
	"""
	Writes raw bytes to the TAR stream immediately.
	"""
	# Create a TarInfo object (header)
	info = tarfile.TarInfo(name=filename)
	info.size = len(data_bytes)
	info.mtime = time.time()

	# Write Header and Data
	with io.BytesIO(data_bytes) as f:
		tf.addfile(info, f)

	# Crucial: Flush Python's buffer to OS, and OS buffer to Disk
	# This ensures that if power is cut 1ms later, this file is saved.
	tf.fileobj.flush()
	os.fsync(tf.fileobj.fileno())


def write_grayscale_to_tar(tf, filename, array_gpu):
	normalized = util.normalize(array_gpu)
	cpu_data = cp.asnumpy(normalized)
	uint8_data = (cpu_data * 255).astype(np.uint8)

	with io.BytesIO() as output:
		Image.fromarray(uint8_data, mode='L').save(output, format="PNG", optimize=True)
		add_bytes_to_tar(tf, filename, output.getvalue())

	del cpu_data, uint8_data, normalized


def write_hillshade_to_tar(tf, filename, array_gpu):
	shaded_gpu = util.hillshaded(array_gpu)
	cpu_data = cp.asnumpy(shaded_gpu)
	uint8_data = (np.clip(cpu_data, 0.0, 1.0) * 255).astype(np.uint8)

	with io.BytesIO() as output:
		Image.fromarray(uint8_data, mode='RGB').save(output, format="PNG", optimize=True)
		add_bytes_to_tar(tf, filename, output.getvalue())

	del cpu_data, uint8_data, shaded_gpu


def write_npy_to_tar(tf, filename, array_gpu):
	cpu_data = cp.asnumpy(array_gpu)
	with io.BytesIO() as output:
		np.save(output, cpu_data)
		add_bytes_to_tar(tf, filename, output.getvalue())
	del cpu_data


def save_state_package(tf, run_id, stage, terrain_gpu):
	"""Saves the trio (NPY, Gray, Hillshade) to the TAR stream."""
	print(f"[{run_id}] Archiving '{stage}'...")
	# NPY first (Raw Data)
	write_npy_to_tar(tf, f"{run_id}_{stage}.npy", terrain_gpu)
	# Then visuals
	write_grayscale_to_tar(tf, f"{run_id}_{stage}_gray.png", terrain_gpu)
	write_hillshade_to_tar(tf, f"{run_id}_{stage}_hillshade.png", terrain_gpu)


# --- Generation Algorithms (Same as before) ---

def gen_plain_fbm(shape): return util.fbm(shape, -2, lower=2.0)


def gen_standard_fbm(shape): return util.fbm(shape, -2.0)


def gen_ridge_noise(shape):
	def noise_octave(s, f): return util.fbm(s, -1, lower=f, upper=(2 * f))

	values = cp.zeros(shape, dtype=cp.float16)
	for p in range(1, 12):
		a = 2 ** p
		values += cp.abs(noise_octave(shape, a) - 0.5) / a
	return (1.0 - util.normalize(values)) ** 2


def gen_domain_warping(shape):
	values = util.fbm(shape, -2, lower=2.0)
	noise_real = util.fbm(shape, -2, lower=1.5)
	noise_imag = util.fbm(shape, -2, lower=1.5)
	warp_scale = 150.0 * SEED_SCALE_FACTOR
	offsets = warp_scale * (noise_real + 1j * noise_imag)
	return util.sample(values, offsets).astype(cp.float16)


GENERATORS = [
	("Plain_FBM", gen_plain_fbm),
	("Standard_FBM", gen_standard_fbm),
	("Ridge_Noise", gen_ridge_noise),
	("Domain_Warping", gen_domain_warping)
]


# --- Simulation Logic ---

def apply_slippage(terrain, repose_slope, cell_width):
	delta = util.simple_gradient(terrain) / cell_width
	smoothed = util.gaussian_blur(terrain, sigma=1.5)
	result = cp.where(cp.abs(delta) > repose_slope, smoothed, terrain)
	return result.astype(cp.float16)


def run_simulation_cycle(tf, run_id, generator_name, generator_func):
	full_width = 200
	shape = (DIMENSION, DIMENSION)
	cell_width = full_width / DIMENSION
	cell_area = cell_width ** 2
	rain_rate = 0.0008 * cell_area
	evaporation_rate = 0.0005
	min_height_delta = 0.05
	repose_slope = 0.03
	gravity = 30.0
	sediment_capacity_constant = 50.0
	dissolving_rate = 0.25
	deposition_rate = 0.001
	iterations = int(sqrt(2) * DIMENSION)

	print(f"[{run_id}] Algo: {generator_name}")
	t0 = time.time()

	# 1. Generate & Save IMMEDIATE
	print(f"[{run_id}] Gen & Save 'Before'...")
	terrain = generator_func(shape)

	# Write 'Before' state to TAR immediately.
	# Even if sim crashes later, this data is safe and readable.
	save_state_package(tf, f"{run_id}_{generator_name}", "01_before", terrain)

	# 2. Simulate
	print(f"[{run_id}] Simulating...")

	sediment = cp.zeros_like(terrain, dtype=cp.float16)
	water = cp.zeros_like(terrain, dtype=cp.float16)
	velocity = cp.zeros_like(terrain, dtype=cp.float16)

	sim_start = time.time()
	for i in range(iterations):
		if not KEEP_RUNNING: return

		water += (cp.random.rand(*shape, dtype=cp.float32) * rain_rate).astype(cp.float16)
		gradient = util.simple_gradient(terrain)
		random_gradient = cp.exp(2j * cp.pi * cp.random.rand(*shape, dtype=cp.float32)).astype(cp.complex64)
		gradient = cp.where(cp.abs(gradient) < 1e-10, random_gradient, gradient)
		gradient /= cp.abs(gradient)
		neighbor_height = util.sample(terrain, -gradient).astype(cp.float16)
		height_delta = terrain - neighbor_height
		sediment_capacity = ((cp.maximum(height_delta,
		                                 min_height_delta) / cell_width) * velocity * water * sediment_capacity_constant)
		val_if_negative_delta = cp.minimum(height_delta, sediment)
		val_if_over_capacity = deposition_rate * (sediment - sediment_capacity)
		val_else = dissolving_rate * (sediment - sediment_capacity)
		deposited_sediment = cp.where(height_delta < 0,
		                              val_if_negative_delta,
		                              cp.where(sediment > sediment_capacity,
		                                       val_if_over_capacity,
		                                       val_else)).astype(cp.float16)
		deposited_sediment = cp.maximum(-height_delta, deposited_sediment)
		sediment -= deposited_sediment
		terrain += deposited_sediment
		sediment = util.displace(sediment, gradient).astype(cp.float16)
		water = util.displace(water, gradient).astype(cp.float16)
		terrain = apply_slippage(terrain, repose_slope, cell_width)
		velocity = gravity * height_delta / cell_width
		water *= (1 - evaporation_rate)

		if i % 200 == 0:
			sys.stdout.write(f"\r[{run_id}] Step: {i}/{iterations}")
			sys.stdout.flush()

	print(f"\n[{run_id}] Sim Done ({time.time() - sim_start:.2f}s).")

	# 3. Save After
	terrain = util.normalize(terrain)
	save_state_package(tf, f"{run_id}_{generator_name}", "02_after", terrain)

	del sediment, water, velocity, terrain
	print(f"[{run_id}] Cycle Done. Total: {time.time() - t0:.2f}s\n" + "-" * 50)


def main(argv):
	# Use .tar extension
	output_path = 'terrain_archive_8k.tar'
	if len(argv) > 1:
		output_path = argv[1]

	signal.signal(signal.SIGINT, signal_handler)

	print(f"=== Continuous Terrain Generator (SAFE TAR MODE) ===")
	print(f"Output: {output_path}")
	print(f"Note: This uses TAR format. You can kill the process at any time,")
	print(f"      and the file will remain valid up to the last written image.")

	# Initialize GPU
	try:
		cp.cuda.Device(0).use()
	except Exception as e:
		print(e);
		sys.exit(1)

	run_counter = 0

	# Open TAR in append mode ('a') or write mode ('w').
	# 'w' is better if starting fresh, 'a' if resuming.
	# We use uncompressed stream for maximum safety and speed.
	mode = 'a' if os.path.exists(output_path) else 'w'

	# We keep the file open continuously to allow appending
	with tarfile.open(output_path, mode) as tf:
		for j in range(4096):
			gen_idx = run_counter % len(GENERATORS)
			gen_name, gen_func = GENERATORS[gen_idx]
			run_id = f"{int(time.time())}_{run_counter:04d}"

			try:
				run_simulation_cycle(tf, run_id, gen_name, gen_func)
				run_counter += 1
			except Exception as e:
				print(f"\n[ERROR] {e}")
				import traceback
				traceback.print_exc()
				break

	print("\n=== Exiting Program ===")


if __name__ == '__main__':
	main(sys.argv)
