#!/usr/bin/python3

# Extracts the underlying heightmap in each file.
# Depends on GDAL which is a binary C++ extension, generally runs on CPU.
# JAX not applicable here as this is purely IO and format parsing.

import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

import numpy as np

# Note: osgeo/gdal is required. It's not a standard pip package usually.
try:
	from osgeo import gdal
except ImportError:
	print("Error: GDAL library not found. Please install python3-gdal.")
	sys.exit(1)

import util


# Extracts the IMG file from the ZIP archive and returns a Numpy array
def get_img_array_from_zip(zip_file, img_name):
	with tempfile.NamedTemporaryFile() as temp_file:
		# Copy to temp file.
		with zip_file.open(img_name) as img_file:
			shutil.copyfileobj(img_file, temp_file)
			temp_file.flush()  # Ensure data is written

		# Extract as numpy array.
		# GDAL Open needs the file to exist on disk usually
		geo = gdal.Open(temp_file.name)
		return geo.ReadAsArray() if geo is not None else None


def main(argv):
	my_dir = os.path.dirname(os.path.abspath(argv[0]))
	input_dir = os.path.join(my_dir, 'zip_files')
	output_dir = os.path.join(my_dir, 'array_files')

	if len(argv) != 2:
		print('Usage: %s <ned_file.csv>' % (argv[0]))
		sys.exit(-1)

	csv_path = argv[1]

	try:
		os.makedirs(output_dir, exist_ok=True)
	except OSError:
		pass

	entries = util.read_csv(csv_path)
	for index, entry in enumerate(entries):
		src_id = entry['sourceId']
		print('(%d / %d) Processing %s' % (index + 1, len(entries), src_id))
		zip_path = os.path.join(input_dir, src_id + '.zip')

		if not os.path.exists(zip_path):
			print(f"Zip file not found: {zip_path}")
			continue

		try:
			# Go though each zip file.
			with zipfile.ZipFile(zip_path, mode='r') as zf:
				ext_names = [name for name in zf.namelist()
				             if os.path.splitext(name)[1].lower() == '.img']
				# Check if EXT files.
				if len(ext_names) == 0:
					print('No IMG files found for %s' % (src_id))
					continue

				# Warn if there is more than one IMG file
				if len(ext_names) > 1:
					print('More than one IMG file found for %s: %s' % (src_id, ext_names))

				# Get the bounding box.
				bounding_box_raw = entry['boundingBox']
				# Fix broken pseudo-JSON from USGS
				bounding_box_json = re.sub(r'([a-zA-Z]+):', r'"\1":', bounding_box_raw)
				bounding_box = json.loads(bounding_box_json)

				# Create numpy array from IMG file and write it to output
				array = get_img_array_from_zip(zf, ext_names[0])
				if array is not None:
					output_path = os.path.join(output_dir, src_id + '.npz')
					# Keep as standard numpy array for disk storage
					np.savez(output_path, height=array, **bounding_box)
				else:
					print('Failed to load array for %s' % src_id)

		except (zipfile.BadZipfile, IOError) as e:
			print(f"Error processing {zip_path}: {e}")
			continue


if __name__ == '__main__':
	main(sys.argv)
