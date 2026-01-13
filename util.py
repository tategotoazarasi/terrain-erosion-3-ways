# Various common functions.
# Refactored to provide JAX-accelerated math functions alongside
# standard Python/Numpy IO and CPU-bound algorithms.

import collections
import csv

import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial
from PIL import Image
from matplotlib.colors import LightSource, LinearSegmentedColormap


# Ensure JAX uses 64-bit if needed, though 32-bit is standard for graphics/ML
# jax.config.update("jax_enable_x64", True)

# --- CPU / IO Functions ---

def read_csv(csv_path):
	with open(csv_path, 'r') as csv_file:
		return list(csv.DictReader(csv_file))


def normalize_np(x, bounds=(0, 1)):
	"""Numpy version of normalize for CPU arrays."""
	return np.interp(x, (x.min(), x.max()), bounds)


def load_from_file(path):
	result = np.load(path)
	if isinstance(result, np.lib.npyio.NpzFile):
		return (result['height'], result['land_mask'] if 'land_mask' in result else None)
	else:
		return (result, None)


def save_as_png(a, path):
	# Ensure input is numpy array
	if hasattr(a, 'device_buffer'):  # Is JAX array
		a = np.array(a)
	image = Image.fromarray(np.round(a * 255).astype('uint8'))
	image.save(path)


def make_grid_points_np(shape):
	"""Numpy version for Matplotlib compatibility."""
	Y, X = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
	grid_points = np.column_stack([X.flatten(), Y.flatten()])
	return grid_points


# --- JAX Accelerated Functions ---

@jax.jit
def normalize(x, bounds=(0.0, 1.0)):
	min_val = jnp.min(x)
	max_val = jnp.max(x)
	# Avoid division by zero
	scale = jnp.where(max_val > min_val, 1.0 / (max_val - min_val), 0.0)
	norm = (x - min_val) * scale
	return norm * (bounds[1] - bounds[0]) + bounds[0]


@jax.jit
def fbm(key, shape, p, lower=-jnp.inf, upper=jnp.inf):
	# JAX FFT
	freqs = tuple(jnp.fft.fftfreq(n, d=1.0 / n) for n in shape)
	# meshgrid in JAX defaults to 'xy', but we need 'ij' to match shape dimensions cleanly for FFT logic usually
	# or handle transpose. standard fftfreq order is fine.
	# We use meshgrid with 'ij' indexing to match array indexing
	freq_grids = jnp.meshgrid(*freqs, indexing='ij')
	freq_radial = jnp.hypot(*freq_grids)

	envelope = (jnp.power(freq_radial, p, where=freq_radial != 0) *
	            (freq_radial > lower) * (freq_radial < upper))
	# Immutable array update
	envelope = envelope.at[0, 0].set(0.0)

	# Random phase
	rand_vals = jax.random.uniform(key, shape)
	phase_noise = jnp.exp(2j * jnp.pi * rand_vals)

	# FFT
	return normalize(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(phase_noise) * envelope)))


@jax.jit
def lerp(x, y, a):
	return (1.0 - a) * x + a * y


@jax.jit
def sample(a, offset):
	"""
	Returns values of `a` offset by `offset`.
	Uses bilinear interpolation.
	"""
	shape = jnp.array(a.shape)
	# Extract real/imag parts
	delta_real = jnp.real(offset)
	delta_imag = jnp.imag(offset)

	# Create coordinate grid
	y_idx, x_idx = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij')

	# Apply offset (Note: original code subtracted delta)
	coords_y = y_idx - delta_imag  # Imag corresponds to axis 0/1 depending on convention, sticking to original
	coords_x = x_idx - delta_real  # Real corresponds to other axis

	# Floor and offsets
	y0 = jnp.floor(coords_y).astype(int)
	x0 = jnp.floor(coords_x).astype(int)

	y1 = y0 + 1
	x1 = x0 + 1

	dy = coords_y - y0
	dx = coords_x - x0

	# Wrap coordinates (Toroidal world)
	y0 = y0 % shape[0]
	y1 = y1 % shape[0]
	x0 = x0 % shape[1]
	x1 = x1 % shape[1]

	# Gather values
	v00 = a[y0, x0]
	v01 = a[y0, x1]
	v10 = a[y1, x0]
	v11 = a[y1, x1]

	# Interpolate
	# X interpolation
	vx0 = lerp(v00, v01, dx)
	vx1 = lerp(v10, v11, dx)

	# Y interpolation
	return lerp(vx0, vx1, dy)


@jax.jit
def displace(a, delta):
	"""
	Takes each value of `a` and offsets them by `delta`.
	JAX implementation of the 'splatting' or distribution logic.
	"""
	fns = {
		-1: lambda x: -x,
		0: lambda x: 1.0 - jnp.abs(x),
		1 : lambda x: x,
	}

	result = jnp.zeros_like(a)
	delta_real = jnp.real(delta)
	delta_imag = jnp.imag(delta)

	# Original loop unrolled effectively by summation
	for dx in range(-1, 2):
		wx = jnp.maximum(fns[dx](delta_real), 0.0)
		for dy in range(-1, 2):
			wy = jnp.maximum(fns[dy](delta_imag), 0.0)

			term = wx * wy * a
			# Roll moves data.
			# Original: np.roll(np.roll(..., dy, axis=0), dx, axis=1)
			shifted = jnp.roll(term, shift=(dy, dx), axis=(0, 1))
			result += shifted

	return result


@jax.jit
def gaussian_gradient(a, sigma=1.0):
	shape = a.shape
	freqs = tuple(jnp.fft.fftfreq(n, d=1.0 / n) for n in shape)
	# indexing='ij' for correct axis mapping
	fy, fx = jnp.meshgrid(*freqs, indexing='ij')

	sigma2 = sigma ** 2
	# g = Gaussian in freq domain
	# Standard result: FT of Gaussian is Gaussian
	# Formula used in original: 1/sqrt(...) * exp(...)
	g = lambda x: ((2 * jnp.pi * sigma2) ** -0.5) * jnp.exp(-0.5 * (x / sigma) ** 2)
	# dg is derivative factor?
	# Original code: dg = lambda x: g(x) * (x / sigma2)
	# This looks like taking derivative in freq domain? i*k * F(k)?
	# Original logic preserved exactly for consistency
	dg = lambda x: g(x) * (x / sigma2)

	fa = jnp.fft.fft2(a)

	# Calculate derivatives via convolution in freq domain
	# Note: meshgrid order. fy is axis 0, fx is axis 1
	term_dy = jnp.fft.ifft2(jnp.fft.fft2(dg(fy) * g(fx)) * fa).real
	term_dx = jnp.fft.ifft2(jnp.fft.fft2(g(fy) * dg(fx)) * fa).real

	return 1j * term_dx + term_dy


@jax.jit
def simple_gradient(a):
	dx = 0.5 * (jnp.roll(a, -1, axis=0) - jnp.roll(a, 1, axis=0))  # Swapped signs to match np.roll behavior?
	# Wait, np.roll(a, 1) shifts right/down.
	# Central diff: (f(x+1) - f(x-1)) / 2
	# roll(a, -1) brings x+1 to x. roll(a, 1) brings x-1 to x.
	# So: (roll(-1) - roll(1)) / 2

	dx = 0.5 * (jnp.roll(a, -1, axis=0) - jnp.roll(a, 1, axis=0))
	dy = 0.5 * (jnp.roll(a, -1, axis=1) - jnp.roll(a, 1, axis=1))
	return 1j * dx + dy


@jax.jit
def gaussian_blur(a, sigma=1.0):
	freqs = tuple(jnp.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
	freq_grids = jnp.meshgrid(*freqs, indexing='ij')
	freq_radial = jnp.hypot(*freq_grids)

	sigma2 = sigma ** 2
	g = lambda x: ((2 * jnp.pi * sigma2) ** -0.5) * jnp.exp(-0.5 * (x / sigma) ** 2)
	kernel = g(freq_radial)
	kernel = kernel / jnp.sum(kernel)  # Normalize? Original uses sum(). In freq domain usually set DC=1
	# Original code: kernel /= kernel.sum(). This happens in spatial or freq?
	# Original logic: kernel = g(freq_radial). This is constructing the filter in freq domain directly?
	# If constructing in freq domain, we shouldn't sum-normalize like a spatial kernel.
	# However, let's follow original code exactly.
	kernel = kernel / jnp.sum(kernel)

	return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(a) * jnp.fft.fft2(kernel)))


# --- Visualization (CPU) ---

_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
	(0.00, (0.15, 0.3, 0.15)),
	(0.25, (0.3, 0.45, 0.3)),
	(0.50, (0.5, 0.5, 0.35)),
	(0.80, (0.4, 0.36, 0.33)),
	(1.00, (1.0, 1.0, 1.0)),
])


def hillshaded(a, land_mask=None, angle=270):
	if land_mask is None: land_mask = np.ones_like(a)
	ls = LightSource(azdeg=angle, altdeg=30)
	land = ls.shade(a, cmap=_TERRAIN_CMAP, vert_exag=10.0,
	                blend_mode='overlay')[:, :, :3]
	water = np.tile((0.25, 0.35, 0.55), a.shape + (1,))
	# Use CPU lerp for visualization
	return (1.0 - land_mask[:, :, np.newaxis]) * water + land_mask[:, :, np.newaxis] * land


# --- Algorithmic Utils (CPU) ---

def poisson_disc_sampling(shape, radius, retries=16):
	"""
	Poisson Disc Sampling.
	Kept on CPU via NumPy because it is inherently sequential/branchy
	(rejection sampling with spatial hashing).
	"""
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
			cx, cy = cell[0] + offset[0], cell[1] + offset[1]
			if (cx, cy) in grid:
				p2 = grid[(cx, cy)]
				diff = np.subtract(p2, p)
				if np.dot(diff, diff) <= radius * radius:
					return True
		return False

	def add_point(p):
		grid[tuple(to_cell(p))] = p
		q.append(p)
		points.append(p)

	q = collections.deque()
	first = shape * np.random.rand(2)
	add_point(first)

	while len(q) > 0:
		# Randomized pop? Original just pop(). Depth-first growth.
		point = q.pop()
		# If we want more random growth, pop(randint). But pop() is fine.

		for _ in range(retries):
			diff = 2 * radius * (2 * np.random.rand(2) - 1)
			r2 = np.dot(diff, diff)
			new_point = diff + point
			if (new_point[0] >= 0 and new_point[0] < shape[0] and
					new_point[1] >= 0 and new_point[1] < shape[1] and
					r2 > radius * radius and r2 < 4 * radius * radius and
					not has_neighbors_in_radius(new_point)):
				add_point(new_point)

	num_points = len(points)
	return np.concatenate(points).reshape((num_points, 2))


def dist_to_mask(mask):
	"""
	Calculates distance to nearest false value in mask.
	Uses SciPy cKDTree (CPU).
	"""
	# Create border mask
	# np.roll on CPU
	border_mask = (np.maximum.reduce([
		np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
		np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))

	border_points = np.column_stack(np.where(border_mask > 0))

	if len(border_points) == 0:
		return np.zeros(mask.shape)

	kdtree = scipy.spatial.cKDTree(border_points)
	grid_points = make_grid_points_np(mask.shape)

	return kdtree.query(grid_points)[0].reshape(mask.shape)


def worley(shape, spacing):
	points = poisson_disc_sampling(shape, spacing)
	coords = np.floor(points).astype(int)
	mask = np.zeros(shape, dtype=bool)
	# Ensure coords in bounds
	coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
	coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)

	mask[coords[:, 0], coords[:, 1]] = True
	return normalize_np(dist_to_mask(mask))
