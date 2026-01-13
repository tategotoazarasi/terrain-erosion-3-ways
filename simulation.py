#!/usr/bin/python3

# Semi-physically-based hydraulic erosion simulation.
# MASSIVELY ACCELERATED using JAX (jax.lax.scan).
# The loop runs entirely on the GPU without Python control flow overhead per iteration.

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import util


# Smooths out slopes of `terrain` that are too steep.
@jax.jit
def apply_slippage(terrain, repose_slope, cell_width):
	delta = util.simple_gradient(terrain) / cell_width
	smoothed = util.gaussian_blur(terrain, sigma=1.5)
	should_smooth = jnp.abs(delta) > repose_slope
	# Use jnp.where (JAX's select/if logic)
	result = jnp.where(should_smooth, smoothed, terrain)
	return result


# Single step of erosion simulation, JIT-compiled
@jax.jit
def simulation_step(carry, iteration_idx):
	# Unpack state
	state, constants, base_key = carry
	terrain, sediment, water, velocity = state

	# Constants
	(shape, rain_rate, evaporation_rate, min_height_delta, repose_slope,
	 gravity, cell_width, sediment_capacity_constant,
	 dissolving_rate, deposition_rate) = constants

	# Randomness for this step
	step_key = jax.random.fold_in(base_key, iteration_idx)
	rain_key, grad_key = jax.random.split(step_key)

	# 1. Add Precipitation
	water += jax.random.uniform(rain_key, shape) * rain_rate

	# 2. Compute Gradient
	gradient = util.simple_gradient(terrain)

	# Handle flat areas with random gradient
	random_angle = jax.random.uniform(grad_key, shape) * 2 * jnp.pi
	random_grad = jnp.exp(2j * random_angle)

	# If gradient is effectively zero, use random
	mask_flat = jnp.abs(gradient) < 1e-10
	gradient = jnp.where(mask_flat, random_grad, gradient)
	gradient = gradient / jnp.abs(gradient)  # Normalize

	# 3. Movement
	neighbor_height = util.sample(terrain, -gradient)
	height_delta = terrain - neighbor_height

	# 4. Sediment Capacity
	sediment_capacity = (
			(jnp.maximum(height_delta, min_height_delta) / cell_width) * velocity *
			water * sediment_capacity_constant
	)

	# 5. Deposition / Erosion
	# Logic:
	# If delta < 0 (uphill?): Drop everything -> min(delta, sediment) ??
	# Actually if height_delta < 0, it means neighbor is higher. We can't move there easily?
	# Original code logic:
	# [height_delta < 0, sediment > cap]

	cond_uphill = height_delta < 0
	cond_over_cap = sediment > sediment_capacity

	amount_to_deposit = jnp.select(
		[cond_uphill, cond_over_cap],
		[jnp.minimum(height_delta, sediment), deposition_rate * (sediment - sediment_capacity)],
		default=dissolving_rate * (sediment - sediment_capacity)
	)

	# Don't erode more than available height difference (don't dig holes deeper than flow)
	# Original: deposited_sediment = np.maximum(-height_delta, deposited_sediment)
	amount_to_deposit = jnp.maximum(-height_delta, amount_to_deposit)

	# Update quantities
	sediment -= amount_to_deposit
	terrain += amount_to_deposit

	# Transport
	sediment = util.displace(sediment, gradient)
	water = util.displace(water, gradient)

	# 6. Slippage
	terrain = apply_slippage(terrain, repose_slope, cell_width)

	# 7. Update Velocity
	velocity = gravity * height_delta / cell_width

	# 8. Evaporation
	water *= (1.0 - evaporation_rate)

	# Pack state
	new_state = (terrain, sediment, water, velocity)

	# Return (carry, output) - we don't need per-step output for the scan result usually
	return (new_state, constants, base_key), None


def main(argv):
	# Grid dimension constants
	full_width = 200.0
	dim = 512
	shape = (dim, dim)
	cell_width = full_width / dim
	cell_area = cell_width ** 2

	my_dir = os.path.dirname(os.path.abspath(argv[0]))
	snapshot_dir = os.path.join(my_dir, 'sim_snaps')
	# Snapshotting in JAX scan is hard. We disable it for pure performance or would need host_callback.
	# We will disable snapshotting for the GPU rewrite.

	# Water-related constants
	rain_rate = 0.0008 * cell_area
	evaporation_rate = 0.0005

	# Slope constants
	min_height_delta = 0.05
	repose_slope = 0.03
	gravity = 30.0

	# Sediment constants
	sediment_capacity_constant = 50.0
	dissolving_rate = 0.25
	deposition_rate = 0.001

	iterations = int(1.4 * dim)

	# Setup JAX Key
	key = jax.random.PRNGKey(0)
	key, noise_key = jax.random.split(key)

	print("Initializing terrain on GPU...")
	# `terrain` represents the actual terrain height we're interested in
	terrain = util.fbm(noise_key, shape, -2.0)
	sediment = jnp.zeros_like(terrain)
	water = jnp.zeros_like(terrain)
	velocity = jnp.zeros_like(terrain)

	# Pack constants
	constants = (shape, rain_rate, evaporation_rate, min_height_delta, repose_slope,
	             gravity, cell_width, sediment_capacity_constant,
	             dissolving_rate, deposition_rate)

	print(f"Running simulation for {iterations} iterations on GPU (JIT+Scan)...")

	init_val = ((terrain, sediment, water, velocity), constants, key)

	# lax.scan runs the loop in compiled XLA code
	final_val, _ = jax.lax.scan(simulation_step, init_val, jnp.arange(iterations))

	(final_terrain, final_sediment, final_water, final_vel), _, _ = final_val

	# Normalize and Save
	print("Simulation complete. Saving...")
	result_cpu = jax.device_get(util.normalize(final_terrain))
	np.save('simulation', result_cpu)


if __name__ == '__main__':
	main(sys.argv)
