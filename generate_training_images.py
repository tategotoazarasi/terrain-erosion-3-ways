#!/usr/bin/python

# Reads the Numpy arrays in array_files/ and generates images for use in
# training. Please note that this script takes a long time to run.

import os
import sys

import cupy as cp
import cv2
import numpy as np
import skimage.measure

import util


# Filters and cleans the given sample. Uses rough heuristics to determine which
# samples are suitable for training via rough heuristics.
def clean_sample(sample_gpu):
	# Note: This function receives a CuPy array.

	# Get rid of "out-of-bounds" magic values.
	sample_gpu[sample_gpu == cp.finfo('float32').min] = 0.0

	# Ignore any samples with NaNs, for one reason or another.
	if cp.isnan(sample_gpu).any(): return None

	# Only accept values that span a given range. This is to capture more
	# mountainous samples.
	if (sample_gpu.max() - sample_gpu.min()) < 40: return None

	# Filter out samples for which a significant portion is within a small
	# threshold from the minimum value. This helps filter out samples that
	# contain a lot of water.
	near_min_fraction = (sample_gpu < (sample_gpu.min() + 8)).sum() / sample_gpu.size
	if near_min_fraction > 0.2: return None

	# Low entropy samples likely have some file corruption or some other artifact
	# that would make it unsuitable as a training sample.
	# Entropy calculation is complex to vectorize efficiently on GPU without
	# custom kernels for histogramming, so we pull to CPU for this check.
	sample_cpu = cp.asnumpy(sample_gpu)
	entropy = skimage.measure.shannon_entropy(sample_cpu)
	if entropy < 10.0: return None

	return util.normalize(sample_gpu)


# This function returns rotated and flipped variants of the provided array. This
# increases the number of training samples by a factor of 8.
def get_variants(a_gpu):
	for b_gpu in (a_gpu, a_gpu.T):  # Original and flipped.
		for k in range(0, 4):  # Rotated 90 degrees x 4
			yield cp.rot90(b_gpu, k)


def main(argv):
	my_dir = os.path.dirname(argv[0])
	source_array_dir = os.path.join(my_dir, 'array_files')
	training_samples_dir = os.path.join(my_dir, 'training_samples')
	sample_dim = 512
	sample_shape = (sample_dim,) * 2
	sample_area = np.prod(sample_shape)

	# Create the training sample directory, if it doesn't already exist.
	try:
		os.mkdir(training_samples_dir)
	except:
		pass

	source_array_paths = [os.path.join(source_array_dir, path)
	                      for path in os.listdir(source_array_dir)]

	training_id = 0
	for (index, source_array_path) in enumerate(source_array_paths):
		print('(%d / %d) Created %d samples so far'
		      % (index + 1, len(source_array_paths), training_id))

		# Load data. This is usually stored as .npz by extract_height_arrays.
		# We load it into host memory first because we need CV2 for resizing.
		data = np.load(source_array_path)

		# Load heightmap and correct for latitude (to an approximation)
		source_array_raw = data['height']
		latitude_deg = (data['minY'] + data['maxY']) / 2
		latitude_correction = np.cos(np.radians(latitude_deg))
		source_array_shape = (
			int(np.round(source_array_raw.shape[0] * latitude_correction)),
			source_array_raw.shape[1])

		# cv2.resize is CPU based.
		source_array_cpu = cv2.resize(source_array_raw, source_array_shape)

		# Move to GPU for slicing and heuristics
		source_array = cp.asarray(source_array_cpu)

		# Determine the number of samples to use per source array.
		sampleable_area = np.subtract(source_array_shape, sample_shape).prod()
		samples_per_array = int(np.ceil(sampleable_area / sample_area))

		if len(source_array.shape) == 0:
			print('Invalid array at %s' % source_array_path)
			continue

		for _ in range(samples_per_array):
			# Select a sample from the source array.
			row = np.random.randint(source_array.shape[0] - sample_shape[0])
			col = np.random.randint(source_array.shape[1] - sample_shape[1])
			sample = source_array[row:(row + sample_shape[0]),
			col:(col + sample_shape[1])]

			# Scale and clean the sample (mostly on GPU)
			sample = clean_sample(sample)

			# Write the sample to a file
			if sample is not None:
				for variant in get_variants(sample):
					output_path = os.path.join(
						training_samples_dir, str(training_id) + '.png')
					# util.save_as_png handles the download from GPU
					util.save_as_png(variant, output_path)

					training_id += 1


if __name__ == '__main__':
	main(sys.argv)
