#!/usr/bin/python3

# A simple domain warping example.

import sys

import cupy as cp
import numpy as np

import util


def main(argv):
	shape = (512,) * 2

	# Util functions now return FP16 CuPy arrays
	values = util.fbm(shape, -2, lower=2.0)

	# Intermediate calc in FP16, complex parts handle roughly as complex64 then back
	noise_real = util.fbm(shape, -2, lower=1.5)
	noise_imag = util.fbm(shape, -2, lower=1.5)

	offsets = 150 * (noise_real + 1j * noise_imag)

	result = util.sample(values, offsets).astype(cp.float16)

	# Transfer to host for saving
	np.save('domain_warping', cp.asnumpy(result))


if __name__ == '__main__':
	main(sys.argv)
