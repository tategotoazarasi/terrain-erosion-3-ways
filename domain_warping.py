#!/usr/bin/python3

# A simple domain warping example.
# Refactored to use JAX for GPU acceleration.

import sys

import jax
import numpy as np  # Used for saving

import util


def main(argv):
	shape = (512,) * 2

	# Initialize PRNG key for JAX
	rng = jax.random.PRNGKey(0)
	rng, key1, key2, key3 = jax.random.split(rng, 4)

	# Move computation to JAX (GPU)
	print("Generating base values...")
	values = util.fbm(key1, shape, -2, lower=2.0)

	print("Generating offsets...")
	# Note: We need separate keys for the two FBM calls to ensure different noise patterns
	offset_real = util.fbm(key2, shape, -2, lower=1.5)
	offset_imag = util.fbm(key3, shape, -2, lower=1.5)

	offsets = 150 * (offset_real + 1j * offset_imag)

	print("Sampling...")
	result = util.sample(values, offsets)

	# Transfer result back to CPU for saving
	result_cpu = jax.device_get(result)

	print("Saving...")
	np.save('domain_warping', result_cpu)


if __name__ == '__main__':
	main(sys.argv)
