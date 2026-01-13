#!/usr/bin/python3

# River network generation.
# NOTE: This file uses SciPy's Delaunay triangulation and Python's heapq.
# These algorithms are inherently sequential/CPU-bound or require dynamic data structures
# that do not map well to JAX/GPU static graphs.
# We utilize JAX for the initial terrain generation (FBM, bumps), but keep the
# graph traversal logic in NumPy/CPU for correctness and stability.

import heapq
import sys

import jax
import jax.numpy as jnp
import matplotlib.tri
import numpy as np
import scipy.spatial as sp
import skimage.measure

import util


# Returns the index of the smallest value of `a`
def min_index(a): return a.index(min(a))


# Returns an array with a bump centered in the middle of `shape`.
# Accelerated with JAX
@jax.jit
def bump(shape, sigma):
	y, x = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij')
	r = jnp.hypot(x - shape[0] / 2, y - shape[1] / 2)
	c = min(shape) / 2
	return jnp.tanh(jnp.maximum(c - r, 0.0) / sigma)


# Returns a list of heights for each point in `points`.
# Graph algorithm (Dijkstra-like) -> Keep on CPU
def compute_height(points, neighbors, deltas, get_delta_fn=None):
	if get_delta_fn is None:
		get_delta_fn = lambda src, dst: deltas[dst]

	dim = len(points)
	result = [None] * dim
	seed_idx = min_index([sum(p) for p in points])
	q = [(0.0, seed_idx)]

	while len(q) > 0:
		(height, idx) = heapq.heappop(q)
		if result[idx] is not None: continue
		result[idx] = height
		for n in neighbors[idx]:
			if result[n] is not None: continue
			heapq.heappush(q, (get_delta_fn(idx, n) + height, n))

	# Convert result to numpy array replacing None with 0 (should cover all though)
	res_array = np.array([r if r is not None else 0.0 for r in result])
	return util.normalize_np(res_array)


def compute_final_height(points, neighbors, deltas, volume, upstream,
                         max_delta, river_downcutting_constant):
	dim = len(points)

	# result = [None] * dim

	def get_delta(src, dst):
		v = volume[dst] if (dst in upstream[src]) else 0.0
		downcut = 1.0 / (1.0 + v ** river_downcutting_constant)
		return min(max_delta, deltas[dst] * downcut)

	return compute_height(points, neighbors, deltas, get_delta_fn=get_delta)


# Computes the river network that traverses the terrain.
# CPU Bound Logic
def compute_river_network(points, neighbors, heights, land,
                          directional_inertia, default_water_level,
                          evaporation_rate):
	num_points = len(points)

	# The normalized vector between points i and j
	def unit_delta(i, j):
		delta = points[j] - points[i]
		return delta / np.linalg.norm(delta)

	q = []
	roots = set()
	for i in range(num_points):
		if land[i]: continue
		is_root = True
		for j in neighbors[i]:
			if not land[j]: continue
			is_root = True
			heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
		if is_root: roots.add(i)

	downstream = [None] * num_points

	while len(q) > 0:
		(_, (i, j, direction)) = heapq.heappop(q)

		if downstream[j] is not None: continue
		downstream[j] = i

		for k in neighbors[j]:
			if (heights[k] < heights[j] or downstream[k] is not None
					or not land[k]):
				continue

			neighbor_direction = unit_delta(j, k)
			priority = -np.dot(direction, neighbor_direction)

			weighted_direction = util.lerp(neighbor_direction, direction,
			                               directional_inertia)
			heapq.heappush(q, (priority, (j, k, weighted_direction)))

	upstream = [set() for _ in range(num_points)]
	for i, j in enumerate(downstream):
		if j is not None: upstream[j].add(i)

	volume = [None] * num_points

	def compute_volume(i):
		if volume[i] is not None: return
		v = default_water_level
		for j in upstream[i]:
			compute_volume(j)
			v += volume[j]
		volume[i] = v * (1 - evaporation_rate)

	for i in range(0, num_points): compute_volume(i)

	return (upstream, downstream, volume)


# Renders triangulation. Uses Matplotlib (CPU)
def render_triangulation(shape, tri, values):
	points = util.make_grid_points_np(shape)
	triangulation = matplotlib.tri.Triangulation(
		tri.points[:, 0], tri.points[:, 1], tri.simplices)
	interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
	return interp(points[:, 0], points[:, 1]).reshape(shape).filled(0.0)


def remove_lakes(mask):
	labels = skimage.measure.label(mask)
	new_mask = np.zeros_like(mask, dtype=bool)
	# Note: Using Scikit-Image on CPU as it handles connectivity well
	labels_inv = skimage.measure.label(~mask, connectivity=1)
	new_mask[labels_inv != labels_inv[0, 0]] = True
	return new_mask


def main(argv):
	dim = 512
	shape = (dim,) * 2
	disc_radius = 1.0
	max_delta = 0.05
	river_downcutting_constant = 1.3
	directional_inertia = 0.4
	default_water_level = 1.0
	evaporation_rate = 0.2

	key = jax.random.PRNGKey(42)
	key1, key2 = jax.random.split(key)

	print('Generating...')

	print('  ...initial terrain shape (JAX accelerated)')
	# JAX calculations
	fbm_noise = util.fbm(key1, shape, -2, lower=2.0)
	bump_map = bump(shape, 0.2 * dim)

	# Transfer to CPU for masking/labeling
	base_terrain = jax.device_get(fbm_noise + bump_map - 1.1)
	land_mask = remove_lakes(base_terrain > 0)

	# Back to JAX for calculations involving masks if we wanted,
	# but dist_to_mask is CPU based (KDTree), so stay on CPU.
	coastal_dropoff = np.tanh(util.dist_to_mask(land_mask) / 80.0) * land_mask

	# Mountain shapes on GPU
	mountain_shapes = jax.device_get(util.fbm(key2, shape, -2, lower=2.0, upper=np.inf))

	# Gaussian blur is JAX accelerated in util, but we need to marshal data
	# Let's perform the blur on GPU
	pre_blur = jnp.array(np.maximum(mountain_shapes - 0.40, 0.0))
	blurred = jax.device_get(util.gaussian_blur(pre_blur, sigma=5.0))

	initial_height = ((blurred + 0.1) * coastal_dropoff)

	# Gradient on GPU
	ih_gpu = jnp.array(initial_height)
	grad = util.gaussian_gradient(ih_gpu)
	deltas = jax.device_get(util.normalize(jnp.abs(grad)))

	print('  ...sampling points (CPU)')
	# Poisson disc is purely sequential/CPU
	points = util.poisson_disc_sampling(shape, disc_radius)
	coords = np.floor(points).astype(int)

	print('  ...delaunay triangulation (CPU)')
	tri = sp.spatial.Delaunay(points)
	(indices, indptr) = tri.vertex_neighbor_vertices
	neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]

	# Safe indexing on CPU
	points_land = land_mask[coords[:, 0], coords[:, 1]]
	points_deltas = deltas[coords[:, 0], coords[:, 1]]

	print('  ...initial height map (Graph traversal)')
	points_height = compute_height(points, neighbors, points_deltas)

	print('  ...river network (Graph traversal)')
	(upstream, downstream, volume) = compute_river_network(
		points, neighbors, points_height, points_land,
		directional_inertia, default_water_level, evaporation_rate)

	print('  ...final terrain height')
	new_height = compute_final_height(
		points, neighbors, points_deltas, volume, upstream,
		max_delta, river_downcutting_constant)

	print('  ...rendering')
	terrain_height = render_triangulation(shape, tri, new_height)

	np.savez('river_network', height=terrain_height, land_mask=land_mask)


if __name__ == '__main__':
	main(sys.argv)
