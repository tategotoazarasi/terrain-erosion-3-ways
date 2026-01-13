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


# Renormalizes the values of `x` to `bounds` (GPU compatible, FP16 friendly)
def normalize(x, bounds=(0, 1)):
	# Fix 1: Ensure input x is C-contiguous.
	if not x.flags.c_contiguous:
		x = cp.ascontiguousarray(x)

	# Fix 2: cp.interp requires xp and fp to be arrays.
	# We perform interpolation in the native dtype of x (likely FP16).
	xp = cp.stack([x.min(), x.max()])
	fp = cp.asarray(bounds, dtype=x.dtype)

	return cp.interp(x, xp, fp).astype(x.dtype)


# Fourier-based power law noise with frequency bounds. (GPU Accelerated)
def fbm(shape, p, lower=-cp.inf, upper=cp.inf):
	# FFT operations in CuPy generally use float32/complex64 minimum.
	# We perform the generation in standard precision, then cast result to FP16.

	freqs = tuple(cp.fft.fftfreq(n, d=1.0 / n) for n in shape)
	freq_radial = cp.hypot(*cp.meshgrid(*freqs))
	envelope = (cp.power(freq_radial, p) *
	            (freq_radial > lower) * (freq_radial < upper))
	envelope[0][0] = 0.0

	phase_noise = cp.exp(2j * cp.pi * cp.random.rand(*shape, dtype=cp.float32))

	# Complex math happens here
	complex_res = cp.fft.ifft2(cp.fft.fft2(phase_noise) * envelope)

	# Extract real part and convert to FP16
	return normalize(cp.real(complex_res)).astype(cp.float16)


# Returns each value of `a` with coordinates offset by `offset`.
# Runs on GPU.
def sample(a, offset):
	# Shape tuple for CPU logic
	shape_tuple = a.shape
	# Shape GPU for broadcasting
	shape_gpu = cp.array(shape_tuple)

	delta = cp.array((offset.real, offset.imag))

	# Grid generation
	grid_vectors = [cp.arange(n) for n in shape_tuple]
	coords = cp.array(cp.meshgrid(*grid_vectors)) - delta

	lower_coords = cp.floor(coords).astype(int)
	upper_coords = lower_coords + 1
	coord_offsets = (coords - lower_coords).astype(a.dtype)  # Use FP16 if a is FP16

	# Modulo
	lower_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]
	upper_coords %= shape_gpu[:, cp.newaxis, cp.newaxis]

	# Lerp will return same dtype as inputs
	result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
	                   a[lower_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              lerp(a[upper_coords[1], lower_coords[0]],
	                   a[upper_coords[1], upper_coords[0]],
	                   coord_offsets[0]),
	              coord_offsets[1])
	return result


# Takes each value of `a` and offsets them by `delta`.
def displace(a, delta):
	fns = {
		-1: lambda x: -x,
		0: lambda x: 1 - cp.abs(x),
		1 : lambda x: x,
	}
	# Ensure result matches input dtype (FP16)
	result = cp.zeros_like(a)
	for dx in range(-1, 2):
		wx = cp.maximum(fns[dx](delta.real), 0.0).astype(a.dtype)
		for dy in range(-1, 2):
			wy = cp.maximum(fns[dy](delta.imag), 0.0).astype(a.dtype)
			result += cp.roll(cp.roll(wx * wy * a, dy, axis=0), dx, axis=1)

	return result


# Returns the gradient of the gaussian blur of `a` encoded as a complex number.
def gaussian_gradient(a, sigma=1.0):
	# Force float32 for FFT calculation precision, then cast back if needed outside
	[fy, fx] = cp.meshgrid(*(cp.fft.fftfreq(n, 1.0 / n) for n in a.shape))
	sigma2 = sigma ** 2
	g = lambda x: ((2 * cp.pi * sigma2) ** -0.5) * cp.exp(-0.5 * (x / sigma) ** 2)
	dg = lambda x: g(x) * (x / sigma2)

	fa = cp.fft.fft2(a.astype(cp.complex64))
	dy = cp.fft.ifft2(cp.fft.fft2(dg(fy) * g(fx)) * fa).real
	dx = cp.fft.ifft2(cp.fft.fft2(g(fy) * dg(fx)) * fa).real

	# Return complex64 (FP32 parts)
	return 1j * dx + dy


# Simple gradient.
def simple_gradient(a):
	dx = 0.5 * (cp.roll(a, 1, axis=0) - cp.roll(a, -1, axis=0))
	dy = 0.5 * (cp.roll(a, 1, axis=1) - cp.roll(a, -1, axis=1))
	# Return complex64
	return (1j * dx + dy).astype(cp.complex64)


# Loads the terrain height array.
def load_from_file(path):
	result = np.load(path)
	if isinstance(result, np.lib.npyio.NpzFile):
		# Load specific keys to GPU as FP16
		return (cp.asarray(result['height'], dtype=cp.float16),
		        cp.asarray(result['land_mask']))
	else:
		return (cp.asarray(result, dtype=cp.float16), None)


# Saves the array as a PNG image.
def save_as_png(a, path):
	a_cpu = cp.asnumpy(a).astype(np.float32)  # Convert to standard float for Image
	image = Image.fromarray(np.round(a_cpu * 255).astype('uint8'))
	image.save(path)


# Creates a hillshaded RGB array.
_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
	(0.00, (0.15, 0.3, 0.15)),
	(0.25, (0.3, 0.45, 0.3)),
	(0.50, (0.5, 0.5, 0.35)),
	(0.80, (0.4, 0.36, 0.33)),
	(1.00, (1.0, 1.0, 1.0)),
])


def hillshaded(a, land_mask=None, angle=270):
	if land_mask is None: land_mask = cp.ones_like(a)

	a_cpu = cp.asnumpy(a).astype(np.float32)
	land_mask_cpu = cp.asnumpy(land_mask)

	ls = LightSource(azdeg=angle, altdeg=30)
	land = ls.shade(a_cpu, cmap=_TERRAIN_CMAP, vert_exag=10.0,
	                blend_mode='overlay')[:, :, :3]

	water = np.tile((0.25, 0.35, 0.55), a_cpu.shape + (1,))
	result_cpu = lerp(water, land, land_mask_cpu[:, :, np.newaxis])

	return cp.asarray(result_cpu, dtype=cp.float16)


def lerp(x, y, a): return (1.0 - a) * x + a * y


def make_grid_points(shape):
	[Y, X] = cp.meshgrid(cp.arange(shape[0]), cp.arange(shape[1]))
	grid_points = cp.column_stack([X.flatten(), Y.flatten()])
	return grid_points


# Returns a list of points sampled within the bounds of `shape`.
def poisson_disc_sampling(shape, radius, retries=16):
	grid = {}
	points = []

	cell_size = radius / np.sqrt(2)
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


def dist_to_mask(mask_gpu):
	mask = cp.asnumpy(mask_gpu)

	border_mask = (np.maximum.reduce([
		np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
		np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
	border_points = np.column_stack(np.where(border_mask > 0))

	kdtree = sp.spatial.cKDTree(border_points)

	[Y, X] = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
	grid_points = np.column_stack([X.flatten(), Y.flatten()])

	result = kdtree.query(grid_points)[0].reshape(mask.shape)

	# Return FP16
	return cp.asarray(result, dtype=cp.float16)


def worley(shape, spacing):
	points = poisson_disc_sampling(shape, spacing)
	coords = np.floor(points).astype(int)

	mask = cp.zeros(shape, dtype=bool)
	coords_gpu = cp.asarray(coords)
	mask[coords_gpu[:, 0], coords_gpu[:, 1]] = True

	return normalize(dist_to_mask(mask)).astype(cp.float16)


def gaussian_blur(a, sigma=1.0):
	# Calc in complex64/float32
	freqs = tuple(cp.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
	freq_radial = cp.hypot(*cp.meshgrid(*freqs))
	sigma2 = sigma ** 2
	g = lambda x: ((2 * cp.pi * sigma2) ** -0.5) * cp.exp(-0.5 * (x / sigma) ** 2)
	kernel = g(freq_radial)
	kernel /= kernel.sum()

	# Cast back result to FP16
	res = cp.fft.ifft2(cp.fft.fft2(a.astype(cp.complex64)) * cp.fft.fft2(kernel)).real
	return res.astype(cp.float16)
