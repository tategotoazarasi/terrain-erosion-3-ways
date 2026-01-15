#!/usr/bin/python3
"""
Step 1: Data Preparation Script.

Functionality:
1. Scans specific TAR archives (1463094.tar, etc.).
2. Filters files to keep ONLY 'Standard_FBM' generator data.
3. Matches '01_before' (Input) with '02_after' (Target) based on Run ID.
4. Converts float16 data to float32 (standard for PyTorch training).
5. Saves paired data to disk for efficient random access during training.

Usage:
    python3 step1_prepare_dataset.py
"""

import io
import os
import re
import tarfile

import numpy as np
from tqdm import tqdm

# Configuration
TAR_FILES = [
	"1463094.tar",
	"1463095.tar",
	"1463096.tar",
	"1463097.tar"
]

OUTPUT_DIR = "dataset_standard_fbm"
INPUT_SUBDIR = "inputs"  # Before erosion
TARGET_SUBDIR = "targets"  # After erosion

# Regex to parse filenames inside TAR
# Matches: {timestamp}_{id}_Standard_FBM_{stage}.npy
# Group 1: Run ID (e.g., "1735689000_0001")
# Group 2: Stage ("01_before" or "02_after")
FILENAME_PATTERN = re.compile(r"(\d+_\d+)_Standard_FBM_(01_before|02_after)\.npy")


def ensure_dirs():
	"""Creates output directories if they don't exist."""
	os.makedirs(os.path.join(OUTPUT_DIR, INPUT_SUBDIR), exist_ok=True)
	os.makedirs(os.path.join(OUTPUT_DIR, TARGET_SUBDIR), exist_ok=True)


def process_tar_file(tar_path, buffer_dict):
	"""
	Reads a TAR file streamingly.
	Stores partial pairs in buffer_dict.
	Writes complete pairs to disk immediately to save RAM.
	"""
	if not os.path.exists(tar_path):
		print(f"[Warning] File not found: {tar_path}")
		return

	print(f"Processing {tar_path}...")

	# Open as stream ('r|') allows reading without loading the whole index (fast)
	try:
		with tarfile.open(tar_path, 'r|') as tf:
			for member in tqdm(tf):
				if not member.isfile():
					continue

				# Check if file matches our filter (Standard_FBM + .npy)
				match = FILENAME_PATTERN.search(member.name)
				if not match:
					continue

				run_id = match.group(1)
				stage = match.group(2)

				# Extract file content
				f = tf.extractfile(member)
				if f is None:
					continue

				content_bytes = f.read()

				# Load numpy array from bytes
				try:
					with io.BytesIO(content_bytes) as bio:
						# Allow pickle=True is sometimes needed, but default is safer.
						# The simulation saved simple arrays, so allow_pickle=False is fine usually.
						# We cast to float32 immediately to ensure consistency.
						arr = np.load(bio).astype(np.float32)
				except Exception as e:
					print(f"  [Error] Failed to load NPY {member.name}: {e}")
					continue

				# Store in buffer
				if run_id not in buffer_dict:
					buffer_dict[run_id] = {}

				buffer_dict[run_id][stage] = arr

				# Check if we have a complete pair
				if "01_before" in buffer_dict[run_id] and "02_after" in buffer_dict[run_id]:
					save_pair(run_id, buffer_dict[run_id])
					del buffer_dict[run_id]  # Free memory

	except Exception as e:
		print(f"[Error] reading tar {tar_path}: {e}")


def save_pair(run_id, data_pair):
	"""Saves a matched pair to disk."""
	input_arr = data_pair["01_before"]
	target_arr = data_pair["02_after"]

	# Basic Validation: Shapes must match and be valid
	if input_arr.shape != (512, 512) or target_arr.shape != (512, 512):
		print(f"  [Skip] Invalid shape for {run_id}")
		return

	if np.isnan(input_arr).any() or np.isnan(target_arr).any():
		print(f"  [Skip] NaNs detected in {run_id}")
		return

	# Save
	np.save(os.path.join(OUTPUT_DIR, INPUT_SUBDIR, f"{run_id}.npy"), input_arr)
	np.save(os.path.join(OUTPUT_DIR, TARGET_SUBDIR, f"{run_id}.npy"), target_arr)


def main():
	ensure_dirs()

	# Dictionary to hold partial matches: {run_id: {'01_before': arr, ...}}
	# We use a single buffer across all tars in case a pair is split across files (unlikely but safe)
	pair_buffer = {}

	total_files_before = len(os.listdir(os.path.join(OUTPUT_DIR, INPUT_SUBDIR)))

	for tar_file in TAR_FILES:
		process_tar_file(tar_file, pair_buffer)

	# Report
	total_files_after = len(os.listdir(os.path.join(OUTPUT_DIR, INPUT_SUBDIR)))
	new_pairs = total_files_after - total_files_before

	print("-" * 50)
	print(f"Extraction Complete.")
	print(f"Total pairs in '{OUTPUT_DIR}': {total_files_after}")
	print(f"Newly extracted: {new_pairs}")
	print(f"Incomplete pairs discarded (buffer remainder): {len(pair_buffer)}")
	print("-" * 50)


if __name__ == "__main__":
	main()
