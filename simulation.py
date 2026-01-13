#!/usr/bin/python3

# Semi-phisically-based hydraulic erosion simulation.

import os
import sys

import cupy as cp
import numpy as np

import util


# Smooths out slopes of `terrain` that are too steep.
def apply_slippage(terrain, repose_slope, cell_width):
	delta = util.simple_gradient(terrain) / cell_width
	smoothed = util.gaussian_blur(terrain, sigma=1.5)

	# FIX: Use cp.where instead of cp.select to avoid "default only accepts scalar" error
	# result = cp.select([cp.abs(delta) > repose_slope], [smoothed], terrain)
	result = cp.where(cp.abs(delta) > repose_slope, smoothed, terrain)
	return result


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
	iterations = int(1.4 * dim)

	# `terrain` represents the actual terrain height we're interested in
	# Everything is on GPU via util.fbm returning a CuPy array
	terrain = util.fbm(shape, -2.0)

	# `sediment` is the amount of suspended "dirt" in the water.
	sediment = cp.zeros_like(terrain)

	# The amount of water. Responsible for carrying sediment.
	water = cp.zeros_like(terrain)

	# The water velocity.
	velocity = cp.zeros_like(terrain)

	for i in range(0, iterations):
		print('%d / %d' % (i + 1, iterations))

		# Add precipitation.
		water += cp.random.rand(*shape) * rain_rate

		# Compute the normalized gradient of the terrain height
		gradient = cp.zeros_like(terrain, dtype='complex128')
		gradient = util.simple_gradient(terrain)

		# FIX: Use cp.where instead of cp.select
		# gradient = cp.select([cp.abs(gradient) < 1e-10],
		#                      [cp.exp(2j * cp.pi * cp.random.rand(*shape))],
		#                      gradient)
		random_gradient = cp.exp(2j * cp.pi * cp.random.rand(*shape))
		gradient = cp.where(cp.abs(gradient) < 1e-10, random_gradient, gradient)

		gradient /= cp.abs(gradient)

		# Compute the difference between teh current height the height offset by
		# `gradient`.
		neighbor_height = util.sample(terrain, -gradient)
		height_delta = terrain - neighbor_height

		# The sediment capacity represents how much sediment can be suspended in
		# water.
		sediment_capacity = (
				(cp.maximum(height_delta, min_height_delta) / cell_width) * velocity *
				water * sediment_capacity_constant)

		# FIX: Use nested cp.where instead of cp.select
		# deposited_sediment = cp.select(
		# 	[
		# 		height_delta < 0,
		# 		sediment > sediment_capacity,
		# 	], [
		# 		cp.minimum(height_delta, sediment),
		# 		deposition_rate * (sediment - sediment_capacity),
		# 	],
		# 	# If sediment <= sediment_capacity
		# 	dissolving_rate * (sediment - sediment_capacity))

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
		)

		# Don't erode more sediment than the current terrain height.
		deposited_sediment = cp.maximum(-height_delta, deposited_sediment)

		# Update terrain and sediment quantities.
		sediment -= deposited_sediment
		terrain += deposited_sediment
		sediment = util.displace(sediment, gradient)
		water = util.displace(water, gradient)

		# Smooth out steep slopes.
		terrain = apply_slippage(terrain, repose_slope, cell_width)

		# Update velocity
		velocity = gravity * height_delta / cell_width

		# Apply evaporation
		water *= 1 - evaporation_rate

		# Snapshot, if applicable.
		if enable_snapshotting:
			output_path = os.path.join(snapshot_dir, snapshot_file_template % i)
			util.save_as_png(terrain, output_path)

	# Normalize on GPU then save (save_as_png handles download or np.save handles it)
	np.save('simulation', cp.asnumpy(util.normalize(terrain)))


if __name__ == '__main__':
	main(sys.argv)
