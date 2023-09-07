# -*- coding: utf-8 -*-
"""pictotxt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gY2r2qFq_ddBlRbtVFWku36PcBQArYqc
"""

from PIL import Image
import numpy as np
import os

# Define the directory where the images are located and the new dimensions
directory_path = '/content/'
new_size = (192, 256)

# Get all filenames in the specified directory
all_files = os.listdir(directory_path)

# Filter out all .jpg files
image_files = [os.path.join(directory_path, file) for file in all_files if file.endswith('.jpg')]

# Ensure only the first 49 images are processed (if there are more than 49 images in the directory)
image_files = image_files[:48]

# Initialize the final matrix with dimensions of 48x48 rows and 49 columns
final_matrix = np.zeros((192*256, 48))

for idx, image_file in enumerate(image_files):
    # Open the image file
    img = Image.open(image_file)

    # Resize the image
    img = img.resize(new_size)

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Convert the grayscale image to a numpy array
    img_matrix = np.array(img_gray)

    # Vectorize the image matrix
    img_vector = img_matrix.flatten()

    # Store the vector in the corresponding column of the final matrix
    final_matrix[:, idx] = img_vector

# Save the final matrix to a txt file
np.savetxt('/content/final_data.txt', final_matrix, fmt='%f')