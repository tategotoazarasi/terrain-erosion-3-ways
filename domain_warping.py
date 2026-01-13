#!/usr/bin/python3

# A simple domain warping example.

import sys

import cupy as cp
import numpy as np  # Used for saving

import util


def main(argv):
	shape = (512,) * 2

	# Util functions now return CuPy arrays
	values = util.fbm(shape, -2, lower=2.0)

	# Complex math is fully supported in CuPy
	offsets = 150 * (util.fbm(shape, -2, lower=1.5) +
	                 1j * util.fbm(shape, -2, lower=1.5))

	result = util.sample(values, offsets)

	# Transfer to host for saving
	np.save('domain_warping', cp.asnumpy(result))


if __name__ == '__main__':
	main(sys.argv)
