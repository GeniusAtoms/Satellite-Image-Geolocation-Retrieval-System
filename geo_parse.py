"""
Parses image filenames of the form <latitude>,<longitude>.png
into (lat, lon) float tuples. Handles negative coordinates.
"""

import os


def parse_lat_lon(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split(",")
    if len(parts) != 2:
        raise ValueError(f"'{filename}' does not match <lat>,<lon>.png format")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"'{filename}' has non-numeric coordinates")
    return lat, lon
