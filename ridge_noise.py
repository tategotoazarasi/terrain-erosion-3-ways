#!/usr/bin/python3

import sys

import cupy as cp
import numpy as np

import util


def noise_octave(shape, f):
	return util.fbm(shape, -1, lower=f, upper=(2 * f))


def main(argv):
	shape = (512,) * 2

	# Compute in FP16
	values = cp.zeros(shape, dtype=cp.float16)
	for p in range(1, 10):
		a = 2 ** p
		values += cp.abs(noise_octave(shape, a) - 0.5) / a
	result = (1.0 - util.normalize(values)) ** 2

	np.save('ridge', cp.asnumpy(result))


if __name__ == '__main__':
	main(sys.argv)
