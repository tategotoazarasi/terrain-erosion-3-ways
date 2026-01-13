#!/usr/bin/python3

# Reads the Numpy arrays in array_files/ and generates images.
# Image processing pipeline using OpenCV (CPU-bound generally) and NumPy.
# JAX acceleration is not applied here as the bottleneck is typically IO (reading thousands of files).

import os
import sys

import cv2
import numpy as np
import skimage.measure

import util


# Filters and cleans the given sample.
def clean_sample(sample):
	# Get rid of "out-of-bounds" magic values.
	sample[sample == np.finfo('float32').min] = 0.0

	# Ignore any samples with NaNs.
	if np.isnan(sample).any(): return None

	# Only accept values that span a given range (mountainous).
	if (sample.max() - sample.min()) < 40: return None

	# Filter out samples with too much flat low-lying area (water).
	near_min_fraction = (sample < (sample.min() + 8)).sum() / sample.size
	if near_min_fraction > 0.2: return None

	# Low entropy samples likely have corruption or artifacts.
	entropy = skimage.measure.shannon_entropy(sample)
	if entropy < 10.0: return None

	# util.normalize handles numpy arrays fine
	return util.normalize_np(sample)


def get_variants(a):
	for b in (a, a.T):  # Original and flipped.
		for k in range(0, 4):  # Rotated 90 degrees x 4
			yield np.rot90(b, k)


def main(argv):
	my_dir = os.path.dirname(os.path.abspath(argv[0]))
	source_array_dir = os.path.join(my_dir, 'array_files')
	training_samples_dir = os.path.join(my_dir, 'training_samples')
	sample_dim = 512
	sample_shape = (sample_dim,) * 2
	sample_area = np.prod(sample_shape)

	try:
		os.makedirs(training_samples_dir, exist_ok=True)
	except OSError:
		pass

	if not os.path.exists(source_array_dir):
		print(f"Source directory {source_array_dir} does not exist.")
		sys.exit(0)

	source_array_paths = [os.path.join(source_array_dir, path)
	                      for path in os.listdir(source_array_dir)
	                      if path.endswith('.npz') or path.endswith('.npy')]

	training_id = 0
	for (index, source_array_path) in enumerate(source_array_paths):
		if index % 10 == 0:
			print('(%d / %d) Created %d samples so far'
			      % (index + 1, len(source_array_paths), training_id))

		try:
			data = np.load(source_array_path)
			# Handle both .npz and .npy
			if isinstance(data, np.lib.npyio.NpzFile):
				source_array_raw = data['height']
				min_y = data['minY']
				max_y = data['maxY']
			else:
				source_array_raw = data
				min_y, max_y = 0, 0  # Fallback

			# Latitude correction
			latitude_deg = (min_y + max_y) / 2
			latitude_correction = np.cos(np.radians(latitude_deg))
			if np.isnan(latitude_correction) or latitude_correction < 0.1:
				latitude_correction = 1.0

			source_array_shape = (
				int(np.round(source_array_raw.shape[0] * latitude_correction)),
				source_array_raw.shape[1])

			# Resize
			source_array = cv2.resize(source_array_raw, (source_array_shape[1], source_array_shape[0]))

			# Check validity
			if len(source_array.shape) < 2:
				print('Invalid array shape at %s' % source_array_path)
				continue

			# Determine samples
			sampleable_area = np.subtract(source_array.shape, sample_shape).prod()
			if sampleable_area <= 0:
				continue

			samples_per_array = int(np.ceil(sampleable_area / sample_area))

			for _ in range(samples_per_array):
				# Select a sample
				row = np.random.randint(source_array.shape[0] - sample_shape[0])
				col = np.random.randint(source_array.shape[1] - sample_shape[1])
				sample = source_array[row:(row + sample_shape[0]),
				col:(col + sample_shape[1])]

				# Scale and clean
				sample = clean_sample(sample)

				# Write variants
				if sample is not None:
					for variant in get_variants(sample):
						output_path = os.path.join(
							training_samples_dir, str(training_id) + '.png')
						util.save_as_png(variant, output_path)
						training_id += 1

		except Exception as e:
			print(f"Error processing {source_array_path}: {e}")
			continue


if __name__ == '__main__':
	main(sys.argv)
