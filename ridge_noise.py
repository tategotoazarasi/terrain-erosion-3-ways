#!/usr/bin/python3

# A demo of ridge noise using JAX.

import sys

import jax
import jax.numpy as jnp
import numpy as np

import util


# JIT-compiled helper for faster octave accumulation
@jax.jit
def noise_octave(key, shape, f):
	return util.fbm(key, shape, -1, lower=f, upper=(2 * f))


def main(argv):
	shape = (512,) * 2

	key = jax.random.PRNGKey(123)

	values = jnp.zeros(shape)

	print("Generating ridge noise octaves on GPU...")
	for p in range(1, 10):
		key, subkey = jax.random.split(key)
		a = 2.0 ** p
		# Compute octave
		octave = noise_octave(subkey, shape, a)
		values += jnp.abs(octave - 0.5) / a

	result = (1.0 - util.normalize(values)) ** 2

	# Move to CPU
	result_cpu = jax.device_get(result)
	np.save('ridge', result_cpu)


if __name__ == '__main__':
	main(sys.argv)
