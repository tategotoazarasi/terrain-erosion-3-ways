#!/usr/bin/python

# Reads the Numpy arrays in array_files/ and generates images for use in
# training.

import os
import sys

import cupy as cp
import cv2
import numpy as np
import skimage.measure

import util


# Filters and cleans the given sample.
def clean_sample(sample_gpu):
	# Using FP16
	sample_gpu = sample_gpu.astype(cp.float16)

	# Get rid of "out-of-bounds" magic values.
	# Standard finfo for float32 min often used as magic value, check compat with float16
	min_val = cp.finfo('float32').min
	sample_gpu[sample_gpu <= (min_val + 1000)] = 0.0  # Approximate check due to precision

	if cp.isnan(sample_gpu).any(): return None

	if (sample_gpu.max() - sample_gpu.min()) < 40: return None

	near_min_fraction = (sample_gpu < (sample_gpu.min() + 8)).sum() / sample_gpu.size
	if near_min_fraction > 0.2: return None

	sample_cpu = cp.asnumpy(sample_gpu).astype(np.float32)
	entropy = skimage.measure.shannon_entropy(sample_cpu)
	if entropy < 10.0: return None

	return util.normalize(sample_gpu)


def get_variants(a_gpu):
	for b_gpu in (a_gpu, a_gpu.T):
		for k in range(0, 4):
			yield cp.rot90(b_gpu, k)


def main(argv):
	my_dir = os.path.dirname(argv[0])
	source_array_dir = os.path.join(my_dir, 'array_files')
	training_samples_dir = os.path.join(my_dir, 'training_samples')
	sample_dim = 512
	sample_shape = (sample_dim,) * 2
	sample_area = np.prod(sample_shape)

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

		# Load data (likely saved as float16 from extract_height_arrays)
		data = np.load(source_array_path)

		source_array_raw = data['height']
		latitude_deg = (data['minY'] + data['maxY']) / 2
		latitude_correction = np.cos(np.radians(latitude_deg))
		source_array_shape = (
			int(np.round(source_array_raw.shape[0] * latitude_correction)),
			source_array_raw.shape[1])

		# Resize on CPU (float32 usually better for resizing quality then cast back)
		source_array_cpu = cv2.resize(source_array_raw.astype(np.float32), source_array_shape)

		# Move to GPU as FP16
		source_array = cp.asarray(source_array_cpu, dtype=cp.float16)

		sampleable_area = np.subtract(source_array_shape, sample_shape).prod()
		samples_per_array = int(np.ceil(sampleable_area / sample_area))

		if len(source_array.shape) == 0:
			print('Invalid array at %s' % source_array_path)
			continue

		for _ in range(samples_per_array):
			row = np.random.randint(source_array.shape[0] - sample_shape[0])
			col = np.random.randint(source_array.shape[1] - sample_shape[1])
			sample = source_array[row:(row + sample_shape[0]),
			col:(col + sample_shape[1])]

			sample = clean_sample(sample)

			if sample is not None:
				for variant in get_variants(sample):
					output_path = os.path.join(
						training_samples_dir, str(training_id) + '.png')
					util.save_as_png(variant, output_path)

					training_id += 1


if __name__ == '__main__':
	main(sys.argv)
