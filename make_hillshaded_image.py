#!/usr/bin/python3

# Generates a PNG containing a hillshaded version of the terrain height.

import sys

import util


def main(argv):
	if len(argv) != 3:
		print('Usage: %s <input_array.np[yz]> <output_image.png>' % (argv[0],))
		sys.exit(-1)

	input_path = argv[1]
	output_path = argv[2]

	height, land_mask = util.load_from_file(input_path)

	# Use the util function (which uses matplotlib on CPU)
	# Hillshading is fast enough on CPU for single images, and Matplotlib logic is complex to port to JAX
	rgb_array = util.hillshaded(height, land_mask=land_mask)

	util.save_as_png(rgb_array, output_path)


if __name__ == '__main__':
	main(sys.argv)
