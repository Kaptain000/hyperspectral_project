# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:03:39 2025

@author: THINKPAD
"""

import os
import rasterio

# Define the path containing TIFF files
tiff_path = r'E:\ufl\Aim\Aim_1\Masked_TIFF'

# Initialize a list to store the shapes of each TIFF file
tiff_shapes = []

# Iterate through all files in the directory
for file_name in os.listdir(tiff_path):
    if file_name.lower().endswith('.tiff'):  # Check if the file is a TIFF
        file_path = os.path.join(tiff_path, file_name)
        try:
            # Open the TIFF file and get its shape
            with rasterio.open(file_path) as src:
                tiff_shapes.append((file_name, src.shape))  # Store file name and shape
        except Exception as e:
            print(f"Error reading {file_name}: {e}")



