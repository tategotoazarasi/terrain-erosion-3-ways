#!/usr/bin/python3

# Produces terrain samples from the trained generator network.
# NOTE: This script interacts with legacy TensorFlow 1.x pickle files.
# Porting the model itself to JAX is out of scope without the original training code/architecture definition.
# We keep TF 1.x interaction but use numpy for processing.

import os
import pickle
import sys

import numpy as np

# Suppress TF 1.x deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
	import tensorflow.compat.v1 as tf

	tf.disable_eager_execution()
except ImportError:
	import tensorflow as tf


def main(argv):
	if len(argv) < 3:
		print('Usage: %s path/to/progressive_growing_of_gans weights.pkl '
		      '[number_of_samples]' % argv[0])
		sys.exit(-1)

	my_dir = os.path.dirname(os.path.abspath(argv[0]))
	pgog_path = argv[1]
	weight_path = argv[2]
	num_samples = int(argv[3]) if len(argv) >= 4 else 20

	# Load the GAN tensors.
	sess = tf.InteractiveSession()
	sys.path.append(pgog_path)

	# Pickle loading of TF graphs requires the class definitions from the path
	try:
		with open(weight_path, 'rb') as f:
			G, D, Gs = pickle.load(f)
	except FileNotFoundError:
		print(f"Error: Weights file not found at {weight_path}")
		sys.exit(-1)
	except Exception as e:
		print(f"Error loading pickle: {e}")
		print("Ensure the progressive_growing_of_gans repo is in your python path or provided as arg.")
		sys.exit(-1)

	# Generate input vectors.
	latents = np.random.randn(num_samples, *Gs.input_shapes[0][1:])
	labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

	# Run generator to create samples.
	print(f"Generating {num_samples} samples using TensorFlow...")
	samples = Gs.run(latents, labels)

	# Make output directory
	output_dir = os.path.join(my_dir, 'ml_outputs')
	try:
		os.makedirs(output_dir, exist_ok=True)
	except OSError:
		pass

	# Write outputs.
	print("Saving output files...")
	for idx in range(samples.shape[0]):
		# Post-processing math
		sample = (np.clip(np.squeeze((samples[idx, 0, :, :] + 1.0) / 2), 0.0, 1.0)
		          .astype('float64'))
		np.save(os.path.join(output_dir, '%d.npy' % idx), sample)


if __name__ == '__main__':
	main(sys.argv)
