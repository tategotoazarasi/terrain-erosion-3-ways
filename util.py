# Various common functions.

import collections
import csv

# GPU Acceleration
import cupy as cp
import numpy as np
import scipy as sp
from PIL import Image
from matplotlib.colors import LightSource, LinearSegmentedColormap


# Open CSV file as a dict.
def read_csv(csv_path):
	with open(csv_path, 'r') as csv_file:
		return list(csv.DictReader(csv_file))


# Renormalizes the values of `x` to `bounds` (GPU compatible)
def normalize(x, bounds=(0, 1)):
	# Fix 1: Ensure input x is C-contiguous.
	if not x.flags.c_contiguous:
		x = cp.ascontiguousarray(x)

	# Fix 2: cp.interp requires xp and fp to be arrays, not tuples.
	xp = cp.stack([x.min(), x.max()])
	fp = cp.asarray(bounds)

	return cp.interp(x, xp, fp)


# Fourier-based power law noise with frequency bounds. (GPU Accelerated)
def fbm(shape, p, lower=-cp.inf, upper=cp.inf):
	# np.fft -> cp.fft
	freqs = tuple(cp.fft.fftfreq(n, d=1.0 / n) for n in shape)
	freq_radial = cp.hypot(*cp.meshgrid(*freqs))
	envelope = (cp.power(freq_radial, p) *
	            (freq_radial > lower) * (freq_radial < upper))
	# Fix for potential divide by zero in power, and handle 0 freq
	envelope[0][0] = 0.0

	phase_noise = cp.exp(2j * cp.pi * cp.random.rand(*shape))
	return normalize(cp.real(cp.fft.ifft2(cp.fft.fft2(phase_noise) * envelope)))


# Returns each value of `a` with coordinates offset by `offset`.
# Runs on GPU.
def sample(a, offset):
	# FIX: Keep shape as a tuple for range/meshgrid generation on CPU side logic
	shape_tuple = a.shape
	# Create a GPU array version for the modulo arithmetic below
	shape_gpu = cp.array(shape_tuple)

	delta = cp.array((offset.real, offset.imag))

	# Create grid on GPU
	# We use cp.arange on the tuple dimensions to generate coordinates directly on device
	grid_vectors = [cp.arange(n) for n in shape_tuple]
	# Note: Original code used np.meshgrid(*map(range, shape)), which defaults to indexing='xy'
	coords = cp.array(cp.meshgrid(*grid_vectors)) - delta

	lower_coords = cp.floor(coords).astype(int)
	upper_coords = lower_coords + 1
	coord_offsets = coords - lower_coords

	# Use shape_gpu for broadcasting modulo
	lower_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]
	upper_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]

	result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
	                   a[lower_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              lerp(a[upper_coords[1], lower_coords[0]],
	                   a[upper_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              coord_offsets[1])
	return result


# Takes each value of `a` and offsets them by `delta`.
# Runs on GPU.
def displace(a, delta):
	fns = {
		-1: lambda x: -x,
		0: lambda x: 1 - cp.abs(x),
		1 : lambda x: x,
	}
	result = cp.zeros_like(a)
	for dx in range(-1, 2):
		wx = cp.maximum(fns[dx](delta.real), 0.0)
		for dy in range(-1, 2):
			wy = cp.maximum(fns[dy](delta.imag), 0.0)
			# cp.roll instead of np.roll
			result += cp.roll(cp.roll(wx * wy * a, dy, axis=0), dx, axis=1)

	return result


# Returns the gradient of the gaussian blur of `a` encoded as a complex number.
# Runs on GPU via FFT.
def gaussian_gradient(a, sigma=1.0):
	[fy, fx] = cp.meshgrid(*(cp.fft.fftfreq(n, 1.0 / n) for n in a.shape))
	sigma2 = sigma ** 2
	g = lambda x: ((2 * cp.pi * sigma2) ** -0.5) * cp.exp(-0.5 * (x / sigma) ** 2)
	dg = lambda x: g(x) * (x / sigma2)

	fa = cp.fft.fft2(a)
	dy = cp.fft.ifft2(cp.fft.fft2(dg(fy) * g(fx)) * fa).real
	dx = cp.fft.ifft2(cp.fft.fft2(g(fy) * dg(fx)) * fa).real
	return 1j * dx + dy


# Simple gradient by taking the diff of each cell's horizontal and vertical neighbors.
# Runs on GPU.
def simple_gradient(a):
	dx = 0.5 * (cp.roll(a, 1, axis=0) - cp.roll(a, -1, axis=0))
	dy = 0.5 * (cp.roll(a, 1, axis=1) - cp.roll(a, -1, axis=1))
	return 1j * dx + dy


# Loads the terrain height array.
# Handles Numpy (.npy/.npz) files and moves them to CuPy arrays.
def load_from_file(path):
	# np.load handles reading the file, we then move to GPU.
	result = np.load(path)
	if isinstance(result, np.lib.npyio.NpzFile):
		# Load specific keys to GPU
		return (cp.asarray(result['height']), cp.asarray(result['land_mask']))
	else:
		return (cp.asarray(result), None)


# Saves the array as a PNG image. Assumes all input values are [0, 1]
# Transfers from GPU to CPU for saving.
def save_as_png(a, path):
	a_cpu = cp.asnumpy(a)
	image = Image.fromarray(np.round(a_cpu * 255).astype('uint8'))
	image.save(path)


# Creates a hillshaded RGB array of heightmap `a`.
_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
	(0.00, (0.15, 0.3, 0.15)),
	(0.25, (0.3, 0.45, 0.3)),
	(0.50, (0.5, 0.5, 0.35)),
	(0.80, (0.4, 0.36, 0.33)),
	(1.00, (1.0, 1.0, 1.0)),
])


# Hillshading using Matplotlib (CPU based).
# Transfers inputs to CPU, computes, returns GPU array if needed,
# but usually this is used right before saving.
def hillshaded(a, land_mask=None, angle=270):
	if land_mask is None: land_mask = cp.ones_like(a)

	a_cpu = cp.asnumpy(a)
	land_mask_cpu = cp.asnumpy(land_mask)

	ls = LightSource(azdeg=angle, altdeg=30)
	land = ls.shade(a_cpu, cmap=_TERRAIN_CMAP, vert_exag=10.0,
	                blend_mode='overlay')[:, :, :3]

	water = np.tile((0.25, 0.35, 0.55), a_cpu.shape + (1,))
	result_cpu = lerp(water, land, land_mask_cpu[:, :, np.newaxis])

	return cp.asarray(result_cpu)


# Linear interpolation of `x` to `y` with respect to `a`
# Works on both Numpy and CuPy arrays
def lerp(x, y, a): return (1.0 - a) * x + a * y


# Returns a list of grid coordinates for every (x, y) position bounded by `shape`.
# Returns CuPy array.
def make_grid_points(shape):
	[Y, X] = cp.meshgrid(cp.arange(shape[0]), cp.arange(shape[1]))
	grid_points = cp.column_stack([X.flatten(), Y.flatten()])
	return grid_points


# Returns a list of points sampled within the bounds of `shape`.
# NOTE: This uses sequential CPU logic. We return a Numpy array
# because downstream usage (Delaunay) often requires CPU data.
def poisson_disc_sampling(shape, radius, retries=16):
	grid = {}
	points = []

	# The bounds of `shape` are divided into a grid of cells.
	cell_size = radius / np.sqrt(2)
	# Use numpy for the shape math here
	cells = np.ceil(np.divide(shape, cell_size)).astype(int)
	offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
	           (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
	to_cell = lambda p: (p / cell_size).astype('int')

	def has_neighbors_in_radius(p):
		cell = to_cell(p)
		for offset in offsets:
			cell_neighbor = (cell[0] + offset[0], cell[1] + offset[1])
			if cell_neighbor in grid:
				p2 = grid[cell_neighbor]
				diff = np.subtract(p2, p)
				if np.dot(diff, diff) <= radius * radius:
					return True
		return False

	def add_point(p):
		grid[tuple(to_cell(p))] = p
		q.append(p)
		points.append(p)

	q = collections.deque()
	# Random generation on CPU for this specific sequential algo
	first = np.array(shape) * np.random.rand(2)
	add_point(first)
	while len(q) > 0:
		point = q.pop()

		for _ in range(retries):
			diff = 2 * radius * (2 * np.random.rand(2) - 1)
			r2 = np.dot(diff, diff)
			new_point = diff + point
			if (new_point[0] >= 0 and new_point[0] < shape[0] and
					new_point[1] >= 0 and new_point[1] < shape[1] and
					not has_neighbors_in_radius(new_point) and
					r2 > radius * radius and r2 < 4 * radius * radius):
				add_point(new_point)
	num_points = len(points)

	return np.concatenate(points).reshape((num_points, 2))


# Returns an array in which all True values of `mask` contain the distance to
# the nearest False value.
# Uses scipy.spatial.cKDTree which is CPU only.
# Automatically handles transfer to CPU and back to GPU.
def dist_to_mask(mask_gpu):
	# Move to CPU
	mask = cp.asnumpy(mask_gpu)

	border_mask = (np.maximum.reduce([
		np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
		np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
	border_points = np.column_stack(np.where(border_mask > 0))

	kdtree = sp.spatial.cKDTree(border_points)

	# Generate grid points on CPU for the query
	[Y, X] = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
	grid_points = np.column_stack([X.flatten(), Y.flatten()])

	result = kdtree.query(grid_points)[0].reshape(mask.shape)

	# Return to GPU
	return cp.asarray(result)


# Generates worley noise with points separated by `spacing`.
def worley(shape, spacing):
	# Poisson returns Numpy
	points = poisson_disc_sampling(shape, spacing)
	coords = np.floor(points).astype(int)

	# Create mask on GPU
	mask = cp.zeros(shape, dtype=bool)
	# Must use host indices to set values on device array, or move coords to device
	coords_gpu = cp.asarray(coords)
	mask[coords_gpu[:, 0], coords_gpu[:, 1]] = True

	return normalize(dist_to_mask(mask))


# Peforms a gaussian blur of `a`.
# Uses FFT on GPU.
def gaussian_blur(a, sigma=1.0):
	freqs = tuple(cp.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
	freq_radial = cp.hypot(*cp.meshgrid(*freqs))
	sigma2 = sigma ** 2
	g = lambda x: ((2 * cp.pi * sigma2) ** -0.5) * cp.exp(-0.5 * (x / sigma) ** 2)
	kernel = g(freq_radial)
	kernel /= kernel.sum()
	return cp.fft.ifft2(cp.fft.fft2(a) * cp.fft.fft2(kernel)).real
