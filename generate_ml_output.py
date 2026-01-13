#!/usrb/bin/python3

# Prouces terrain samples from the trained generator network.

import os
import pickle
import sys

import cupy as cp
import numpy as np
import tensorflow as tf


def main(argv):
	if len(argv) < 3:
		print('Usage: %s path/to/progressive_growing_of_gans weights.pkl '
		      '[number_of_samples]' % argv[0])
		sys.exit(-1)
	my_dir = os.path.dirname(argv[0])
	pgog_path = argv[1]
	weight_path = argv[2]
	num_samples = int(argv[3]) if len(argv) >= 4 else 20

	# Load the GAN tensors.
	tf.InteractiveSession()
	sys.path.append(pgog_path)
	with open(weight_path, 'rb') as f:
		G, D, Gs = pickle.load(f)

	# Generate input vectors on GPU using CuPy (FP16)
	latents_gpu = cp.random.randn(num_samples, *Gs.input_shapes[0][1:], dtype=cp.float16)

	# TF usually expects Float32, so we convert back to numpy float32 for the TF run
	latents = cp.asnumpy(latents_gpu).astype(np.float32)

	labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

	# Run generator to create samples.
	samples = Gs.run(latents, labels)

	# Make output directory
	output_dir = os.path.join(my_dir, 'ml_outputs')
	try:
		os.mkdir(output_dir)
	except:
		pass

	# Write outputs.
	# Move to GPU and cast to FP16 for post-processing
	samples_gpu = cp.asarray(samples, dtype=cp.float16)

	for idx in range(samples_gpu.shape[0]):
		sample_gpu = (cp.clip(cp.squeeze((samples_gpu[idx, 0, :, :] + 1.0) / 2), 0.0, 1.0)
		              .astype(cp.float16))

		# Save as FP16 .npy
		np.save(os.path.join(output_dir, '%d.npy' % idx), cp.asnumpy(sample_gpu))


if __name__ == '__main__':
	main(sys.argv)
