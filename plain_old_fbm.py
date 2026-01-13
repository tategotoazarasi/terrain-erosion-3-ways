#!/usr/bin/python3

# A demo of just regular FBM noise

import sys

import cupy as cp
import numpy as np  # For saving

import util


def main(argv):
	shape = (512,) * 2
	# fbm runs on GPU, returns CuPy array.
	# np.save handles CuPy arrays in recent versions, or we explicit cast.
	# To be safe and minimal copy:
	result = util.fbm(shape, -2, lower=2.0)
	np.save('fbm', cp.asnumpy(result))


if __name__ == '__main__':
	main(sys.argv)
