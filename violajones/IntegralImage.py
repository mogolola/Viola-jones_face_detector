#!/usr/bin/env python
import os
import numpy as np
from PIL import Image

"""
In an integral image each pixel is the sum of all pixels in the original image 
that are 'left and above' the pixel.

Original    Integral
+--------   +------------
| 1 2 3 .   | 0  0  0  0 .
| 4 5 6 .   | 0  1  3  6 .
| . . . .   | 0  5 12 21 .
            | . . . . . .

"""

def load_images(path):
    """
    Load images from a local folder
    :return: List of images
    :rtype: list[numpy.array]
    """
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            img_arr /= img_arr.max()
            images.append(img_arr)
    return images

def to_integral_image(img_arr):
    """
    Calculates the integral image based on this instance's original image data.
    :param img_arr: Image source data
    :type img_arr: numpy.array
    :return Integral image for given image
    :rtype: numpy.array
    """
    row_sum = np.zeros(img_arr.shape)
    # we need an additional column and an additional row
    integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            integral_image_arr[y+1, x+1] = integral_image_arr[y+1, x-1+1] + row_sum[y, x]
    return integral_image_arr


def sum_region(integral_img_arr, top_left, bottom_right):
    """
    Calculates the sum in the rectangle specified by the given tuples.
    :param integral_img_arr:
    :type integral_img_arr: numpy.array
    :param top_left: (x, y) of the rectangle's top left corner
    :type top_left: (int, int)
    :param bottom_right: (x, y) of the rectangle's bottom right corner
    :type bottom_right: (int, int)
    :return The sum of all pixels in the given rectangle
    :rtype int
    """
    if top_left == bottom_right:
        return integral_img_arr[top_left]
    # calculate the top bottom corner and the right left corner
    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])
    return integral_img_arr[bottom_right] - integral_img_arr[top_right] - integral_img_arr[bottom_left] + integral_img_arr[top_left]


