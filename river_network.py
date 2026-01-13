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


def min_index(a): return a.index(min(a))


def bump(shape, sigma):
	# FP16 Bump
	[y, x] = cp.meshgrid(*map(cp.arange, shape))
	r = cp.hypot(x - shape[0] / 2, y - shape[1] / 2).astype(cp.float16)
	c = min(shape) / 2
	return cp.tanh(cp.maximum(c - r, 0.0) / sigma).astype(cp.float16)


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

	# CPU calculation remains numpy/float, normalize handles cast if needed later
	return util.normalize(np.array(result))


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


def compute_river_network(points, neighbors, heights, land,
                          directional_inertia, default_water_level,
                          evaporation_rate):
	num_points = len(points)

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

			weighted_direction = (1.0 - directional_inertia) * neighbor_direction + directional_inertia * direction
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


def render_triangulation(shape, tri, values):
	points = cp.asnumpy(util.make_grid_points(shape))
	triangulation = matplotlib.tri.Triangulation(
		tri.points[:, 0], tri.points[:, 1], tri.simplices)
	interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)

	result_cpu = interp(points[:, 0], points[:, 1]).reshape(shape).filled(0.0)
	# Result back to FP16 GPU
	return cp.asarray(result_cpu, dtype=cp.float16)


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
	# FP16 Ops
	land_mask = remove_lakes(
		(util.fbm(shape, -2, lower=2.0) + bump(shape, 0.2 * dim) - 1.1) > 0)

	coastal_dropoff = cp.tanh(util.dist_to_mask(land_mask) / 80.0).astype(cp.float16) * land_mask
	mountain_shapes = util.fbm(shape, -2, lower=2.0, upper=np.inf)

	initial_height = (
			(util.gaussian_blur(cp.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0)
			 + 0.1) * coastal_dropoff)

	# Gradient result is complex, we take abs then normalize -> FP16
	deltas = util.normalize(cp.abs(util.gaussian_gradient(initial_height))).astype(cp.float16)

	print('  ...sampling points')
	points = util.poisson_disc_sampling(shape, disc_radius)
	coords = np.floor(points).astype(int)

	print('  ...delaunay triangulation')
	tri = sp.spatial.Delaunay(points)
	(indices, indptr) = tri.vertex_neighbor_vertices
	neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]

	land_mask_cpu = cp.asnumpy(land_mask)
	deltas_cpu = cp.asnumpy(deltas)

	points_land = land_mask_cpu[coords[:, 0], coords[:, 1]]
	points_deltas = deltas_cpu[coords[:, 0], coords[:, 1]]

	print('  ...initial height map')
	points_height = compute_height(points, neighbors, points_deltas)

	print('  ...river network')
	(upstream, downstream, volume) = compute_river_network(
		points, neighbors, points_height, points_land,
		directional_inertia, default_water_level, evaporation_rate)

	print('  ...final terrain height')
	new_height = compute_final_height(
		points, neighbors, points_deltas, volume, upstream,
		max_delta, river_downcutting_constant)

	terrain_height = render_triangulation(shape, tri, new_height)

	np.savez('river_network', height=cp.asnumpy(terrain_height), land_mask=cp.asnumpy(land_mask))


if __name__ == '__main__':
	main(sys.argv)
