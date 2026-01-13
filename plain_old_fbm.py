#!/usr/bin/python3

import sys

import cupy as cp
import numpy as np

import util


def main(argv):
	shape = (512,) * 2
	# fbm returns FP16
	result = util.fbm(shape, -2, lower=2.0)
	np.save('fbm', cp.asnumpy(result))


if __name__ == '__main__':
	main(sys.argv)
