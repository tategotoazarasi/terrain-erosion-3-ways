#!/usr/bin/python3

import heapq
import sys

import cupy as cp
import matplotlib
import matplotlib.tri
import numpy as np
import scipy as sp
import skimage.measure

import util


# Returns the index of the smallest value of `a`
def min_index(a): return a.index(min(a))


# Returns an array with a bump centered in the middle of `shape`. `sigma`
# determines how wide the bump is.
def bump(shape, sigma):
	[y, x] = cp.meshgrid(*map(cp.arange, shape))
	r = cp.hypot(x - shape[0] / 2, y - shape[1] / 2)
	c = min(shape) / 2
	return cp.tanh(cp.maximum(c - r, 0.0) / sigma)


# Returns a list of heights for each point in `points`.
# Note: This runs on CPU as it is graph logic on sparse points.
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
	return util.normalize(np.array(result))


# Same as above, but computes height taking into account river downcutting.
def compute_final_height(points, neighbors, deltas, volume, upstream,
                         max_delta, river_downcutting_constant):
	dim = len(points)
	result = [None] * dim
	seed_idx = min_index([sum(p) for p in points])
	q = [(0.0, seed_idx)]

	def get_delta(src, dst):
		v = volume[dst] if (dst in upstream[src]) else 0.0
		downcut = 1.0 / (1.0 + v ** river_downcutting_constant)
		return min(max_delta, deltas[dst] * downcut)

	return compute_height(points, neighbors, deltas, get_delta_fn=get_delta)


# Computes the river network that traverses the terrain.
# Runs on CPU (Dijkstra/Prim like logic).
def compute_river_network(points, neighbors, heights, land,
                          directional_inertia, default_water_level,
                          evaporation_rate):
	num_points = len(points)

	# The normalized vector between points i and j
	def unit_delta(i, j):
		delta = points[j] - points[i]
		return delta / np.linalg.norm(delta)

	# Initialize river priority queue with all edges between non-land points to
	# land points. Each entry is a tuple of (priority, (i, j, river direction))
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

	# Compute the map of each node to its downstream node.
	downstream = [None] * num_points

	while len(q) > 0:
		(_, (i, j, direction)) = heapq.heappop(q)

		# Assign i as being downstream of j, assuming such a point doesn't
		# already exist.
		if downstream[j] is not None: continue
		downstream[j] = i

		# Go through each neighbor of upstream point j.
		for k in neighbors[j]:
			# Ignore neighbors that are lower than the current point, or who already
			# have an assigned downstream point.
			if (heights[k] < heights[j] or downstream[k] is not None
					or not land[k]):
				continue

			# Edges that are aligned with the current direction vector are
			# prioritized.
			neighbor_direction = unit_delta(j, k)
			priority = -np.dot(direction, neighbor_direction)

			# Add new edge to queue.
			# lerp here is scalar/vector on CPU
			weighted_direction = (1.0 - directional_inertia) * neighbor_direction + directional_inertia * direction
			heapq.heappush(q, (priority, (j, k, weighted_direction)))

	# Compute the mapping of each node to its upstream nodes.
	upstream = [set() for _ in range(num_points)]
	for i, j in enumerate(downstream):
		if j is not None: upstream[j].add(i)

	# Compute the water volume for each node.
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


# Renders `values` for each triangle in `tri` on an array the size of `shape`.
# matplotlib.tri is CPU only.
def render_triangulation(shape, tri, values):
	points = cp.asnumpy(util.make_grid_points(shape))
	triangulation = matplotlib.tri.Triangulation(
		tri.points[:, 0], tri.points[:, 1], tri.simplices)
	interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)

	# Result is numpy, move to cupy if further processing needed
	result_cpu = interp(points[:, 0], points[:, 1]).reshape(shape).filled(0.0)
	return cp.asarray(result_cpu)


# Removes any bodies of water completely enclosed by land.
# Skimage label is CPU only.
def remove_lakes(mask):
	mask_cpu = cp.asnumpy(mask)
	labels = skimage.measure.label(mask_cpu)
	new_mask_cpu = np.zeros_like(mask_cpu, dtype=bool)
	labels = skimage.measure.label(~mask_cpu, connectivity=1)
	new_mask_cpu[labels != labels[0, 0]] = True
	return cp.asarray(new_mask_cpu)


def main(argv):
	dim = 512
	shape = (dim,) * 2
	disc_radius = 1.0
	max_delta = 0.05
	river_downcutting_constant = 1.3
	directional_inertia = 0.4
	default_water_level = 1.0
	evaporation_rate = 0.2

	print('Generating...')

	print('  ...initial terrain shape')
	# Generating terrain on GPU
	land_mask = remove_lakes(
		(util.fbm(shape, -2, lower=2.0) + bump(shape, 0.2 * dim) - 1.1) > 0)

	# util.dist_to_mask handles GPU->CPU->GPU transition internally
	coastal_dropoff = cp.tanh(util.dist_to_mask(land_mask) / 80.0) * land_mask
	mountain_shapes = util.fbm(shape, -2, lower=2.0, upper=np.inf)

	initial_height = (
			(util.gaussian_blur(cp.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0)
			 + 0.1) * coastal_dropoff)

	deltas = util.normalize(cp.abs(util.gaussian_gradient(initial_height)))

	print('  ...sampling points')
	# Points are coordinates (N, 2), returned as Numpy from util for Triangulation compatibility
	points = util.poisson_disc_sampling(shape, disc_radius)  # Returns Numpy
	coords = np.floor(points).astype(int)

	print('  ...delaunay triangulation')
	tri = sp.spatial.Delaunay(points)
	(indices, indptr) = tri.vertex_neighbor_vertices
	neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]

	# Sample GPU arrays at coordinates (Pulling specific values to CPU)
	# Transfer full arrays to CPU once for indexing is more efficient than elementwise
	land_mask_cpu = cp.asnumpy(land_mask)
	deltas_cpu = cp.asnumpy(deltas)

	points_land = land_mask_cpu[coords[:, 0], coords[:, 1]]
	points_deltas = deltas_cpu[coords[:, 0], coords[:, 1]]

	print('  ...initial height map')
	# Running graph logic on CPU
	points_height = compute_height(points, neighbors, points_deltas)

	print('  ...river network')
	(upstream, downstream, volume) = compute_river_network(
		points, neighbors, points_height, points_land,
		directional_inertia, default_water_level, evaporation_rate)

	print('  ...final terrain height')
	new_height = compute_final_height(
		points, neighbors, points_deltas, volume, upstream,
		max_delta, river_downcutting_constant)

	# Render back to grid (CPU -> GPU)
	terrain_height = render_triangulation(shape, tri, new_height)

	np.savez('river_network', height=cp.asnumpy(terrain_height), land_mask=cp.asnumpy(land_mask))


if __name__ == '__main__':
	main(sys.argv)
