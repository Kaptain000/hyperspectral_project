# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:53:11 2025

@author: THINKPAD
"""

import tifffile as tiff
import matplotlib.pyplot as plt

# Function to read and display a TIFF file
def read_and_display_tiff(file_path):
    # Read the TIFF file
    image = tiff.imread(file_path)
    
    # Check the shape of the image
    print(f"Image Shape: {image.shape}")
    
    # Display the image
    plt.figure(figsize=(10, 10))
    
    # If the image is multi-channel or multi-band, display the first band
    if image.ndim == 3:
        print("Multi-band TIFF detected. Displaying the first band.")
        plt.imshow(image[:, :, 0], cmap='gray')
    else:
        plt.imshow(image, cmap='gray')
    
    plt.title("TIFF Image")
    plt.colorbar()
    plt.axis('off')
    plt.show()

# Example usage
file_path = 'E:/ufl/Aim/Aim_1/Segment_Output/0_104_2_interior_vertical.tiff'  # Replace with the path to your TIFF file
read_and_display_tiff(file_path)

file_path = 'E:/ufl/Aim/Aim_1/Masked_TIFF/0_104_4_interior_horizontal.tiff'  # Replace with the path to your TIFF file
read_and_display_tiff(file_path)