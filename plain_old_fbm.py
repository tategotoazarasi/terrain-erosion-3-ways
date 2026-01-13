#!/usr/bin/python3

# A demo of just regular FBM noise using JAX.

import sys

import jax
import numpy as np

import util


def main(argv):
	shape = (512,) * 2

	# JAX PRNG Key
	key = jax.random.PRNGKey(42)

	print("Computing FBM on GPU...")
	# Computes on GPU/Accelerator
	noise = util.fbm(key, shape, -2, lower=2.0)

	# Transfer to CPU
	noise_cpu = jax.device_get(noise)

	print("Saving...")
	np.save('fbm', noise_cpu)


if __name__ == '__main__':
	main(sys.argv)
