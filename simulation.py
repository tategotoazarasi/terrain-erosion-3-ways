#!/usr/bin/python3

# Semi-phisically-based hydraulic erosion simulation.
# FULL FP16 IMPLEMENTATION

import os
import sys
from math import sqrt

import cupy as cp
import numpy as np

import util


def apply_slippage(terrain, repose_slope, cell_width):
	delta = util.simple_gradient(terrain) / cell_width
	smoothed = util.gaussian_blur(terrain, sigma=1.5)

	result = cp.where(cp.abs(delta) > repose_slope, smoothed, terrain)
	return result.astype(cp.float16)


def main(argv):
	# Grid dimension constants
	full_width = 200
	dim = 512
	shape = [dim] * 2
	cell_width = full_width / dim
	cell_area = cell_width ** 2

	# Snapshotting parameters.
	enable_snapshotting = False
	my_dir = os.path.dirname(argv[0])
	snapshot_dir = os.path.join(my_dir, 'sim_snaps')
	snapshot_file_template = 'sim-%05d.png'
	if enable_snapshotting:
		try:
			os.mkdir(snapshot_dir)
		except:
			pass

	# Water-related constants
	rain_rate = 0.0008 * cell_area
	evaporation_rate = 0.0005

	# Slope constants
	min_height_delta = 0.05
	repose_slope = 0.03
	gravity = 30.0
	gradient_sigma = 0.5

	# Sediment constants
	sediment_capacity_constant = 50.0
	dissolving_rate = 0.25
	deposition_rate = 0.001

	# The numer of iterations is proportional to the grid dimension.
	iterations = int(sqrt(2) * dim)

	# --- FP16 Initialization ---
	terrain = util.fbm(shape, -2.0).astype(cp.float16)
	sediment = cp.zeros_like(terrain, dtype=cp.float16)
	water = cp.zeros_like(terrain, dtype=cp.float16)
	velocity = cp.zeros_like(terrain, dtype=cp.float16)

	for i in range(0, iterations):
		print('%d / %d' % (i + 1, iterations))

		# Add precipitation.
		water += (cp.random.rand(*shape, dtype=cp.float32) * rain_rate).astype(cp.float16)

		# Compute the normalized gradient.
		# Note: Complex numbers must be at least complex64 (2x float32),
		# we assume float16 terrain can be cast to that for calc.
		gradient = cp.zeros_like(terrain, dtype=cp.complex64)
		gradient = util.simple_gradient(terrain)  # returns complex (likely complex64 from op)

		# Generate random direction in complex64
		random_gradient = cp.exp(2j * cp.pi * cp.random.rand(*shape, dtype=cp.float32)).astype(cp.complex64)

		gradient = cp.where(cp.abs(gradient) < 1e-10, random_gradient, gradient)
		gradient /= cp.abs(gradient)

		# Gradient is complex64, terrain is float16.
		# util.sample handles this, but returns promoted types potentially.
		# We force cast back to float16.
		neighbor_height = util.sample(terrain, -gradient).astype(cp.float16)
		height_delta = terrain - neighbor_height

		# Calculation in FP16
		sediment_capacity = (
				(cp.maximum(height_delta, min_height_delta) / cell_width) * velocity *
				water * sediment_capacity_constant)

		val_if_negative_delta = cp.minimum(height_delta, sediment)
		val_if_over_capacity = deposition_rate * (sediment - sediment_capacity)
		val_else = dissolving_rate * (sediment - sediment_capacity)

		deposited_sediment = cp.where(
			height_delta < 0,
			val_if_negative_delta,
			cp.where(
				sediment > sediment_capacity,
				val_if_over_capacity,
				val_else
			)
		).astype(cp.float16)

		deposited_sediment = cp.maximum(-height_delta, deposited_sediment)

		# Update terrain and sediment quantities.
		sediment -= deposited_sediment
		terrain += deposited_sediment

		# Displace returns float16 if input is float16
		sediment = util.displace(sediment, gradient).astype(cp.float16)
		water = util.displace(water, gradient).astype(cp.float16)

		# Smooth out steep slopes.
		terrain = apply_slippage(terrain, repose_slope, cell_width)

		# Update velocity
		velocity = gravity * height_delta / cell_width

		# Apply evaporation
		water *= (1 - evaporation_rate)

		# Snapshot, if applicable.
		if enable_snapshotting:
			output_path = os.path.join(snapshot_dir, snapshot_file_template % i)
			util.save_as_png(terrain, output_path)

	np.save('simulation', cp.asnumpy(util.normalize(terrain)))


if __name__ == '__main__':
	main(sys.argv)
