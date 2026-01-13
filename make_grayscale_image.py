#!/usr/bin/python3

# Generates a PNG containing the terrain height in grayscale.

import sys

import util


# Note: util.save_as_png expects standard numpy arrays or will convert them

def main(argv):
	if len(argv) != 3:
		print('Usage: %s <input_array.np[yz]> <output_image.png>' % (argv[0],))
		sys.exit(-1)

	input_path = argv[1]
	output_path = argv[2]

	# Load returns numpy arrays (CPU)
	height, _ = util.load_from_file(input_path)

	# Normalize if needed for PNG
	if height.max() > 1.0 or height.min() < 0.0:
		height = util.normalize_np(height)

	util.save_as_png(height, output_path)


if __name__ == '__main__':
	main(sys.argv)
