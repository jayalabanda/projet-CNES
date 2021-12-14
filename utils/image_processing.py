import datetime as dt
import glob
import itertools
import os
import zipfile
from collections import Counter

import ee
import matplotlib.colors as mplcols
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import utm
from PIL import Image
from sentinelsat import geojson_to_wkt, read_geojson

def open_rasterio(image_path):
    """Opens the image with rasterio.

    Args:
        image_path (string): path to the tiff image

    Returns:
        img: image opened with rasterio
    """
    with rasterio.open(image_path, driver='JP2OpenJPEG') as infile:
        img = infile.read(1).astype('float64')
    return img

def split_image(image, fragment_count):
    """Split images into fragments.
       Allows to select the potion(s) of the image to be used.

    Args:
        image (image): image to be split
        fragment_count (int): number of fragments to be created

    Returns:
        split_image (array): array of the split image
    """
    n = range(fragment_count)
    frag_size = int(image.shape[0] / fragment_count)
    split_image = {}

    for y, x in itertools.product(n, n):
        split_image[(x, y)] = image[y * frag_size: (y + 1) * frag_size,
                                    x * frag_size: (x + 1) * frag_size]
    return split_image


def plot_split_image(split_image, fragment_count):
    """Plots all of the fragmented images.
       Fragemented images come from the split_image() function.

    Args:
        split_image (array): array of the split image. See split_image()
        fragment_count (int): number of fragments
    """
    n = range(fragment_count)
    fig, axs = plt.subplots(fragment_count, fragment_count, figsize=(10, 10))
    for y, x in itertools.product(n, n):
        axs[y, x].imshow(split_image[(x, y)])
        axs[y, x].axis('off')
    plt.tight_layout()
    plt.show()

def calculate_ndvi(red_band, nir_band):
    """Calculates the NDVI of a given image.

    Args:
        red_band (array): red band of the image
        nir_band (array): nir band of the image

    Returns:
        ndvi: NDVI of the image
    """
    # ignore warnings for this block of code
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(nir_band + red_band == 0.,
                        0.,
                        (nir_band - red_band) / (nir_band + red_band))


def create_ndvi_tiff_image(path, when, fire_name, output_folder='output/'):
    """Create the tiff image from the uuid.
        This function creates the tiff image.
        Be mindful of the size of the file.
    Args:
        path (string): path to save the tiff file
        when (string): name of the tiff file. Usually 'before' or 'after'
    Returns:
        image: opened image with the given uuid
    """
    image_path_b1 = get_band(path, 'B04', resolution=10)
    image_path_b2 = get_band(path, 'B08', resolution=10)

    print("First image selected for NDVI:", image_path_b1.split("/")[-1])
    print("Second image selected for NDVI:", image_path_b2.split("/")[-1])

    first_band = rasterio.open(image_path_b1, driver='JP2OpenJPEG')
    # second_band = rasterio.open(image_path_b2, driver='JP2OpenJPEG')

    # red = first_band.read(1).astype('float64')
    red = open_rasterio(image_path_b1)
    # nir = second_band.read(1).astype('float64')
    nir = open_rasterio(image_path_b2)

    ndvi = calculate_ndvi(red, nir)

    # create the tiff file
    #pylint: disable=no-member
    ndvi_img = rasterio.open(
        fp=f'{output_folder}{when}_{fire_name}.tiff',
        mode='w', driver='GTiff',
        width=first_band.width,
        height=first_band.height,
        count=1,
        crs=first_band.crs,
        transform=first_band.transform,
        dtype='float64'
    )

    first_band.close()
    # second_band.close()

    # we only need one band which corresponds to the NDVI
    ndvi_img.write(ndvi, 1)
    ndvi_img.close()

    return rasterio.open(f'{output_folder}{when}_{fire_name}.tiff').read(1)


def threshold_filter(image, threshold):
    """Puts all values below threshold to 0.

    Args:
        image: already imported image
        threshold (float): threshold value

    Returns:
        image: image where all values below threshold are set to 0
    """
    temp = image.copy()
    temp[temp < threshold] = 0
    return temp


def calculate_area(sub_image, original_image, resolution=10):
    """Calculates the surface, in squared kilometers, of the burnt area.

    Args:
        sub_image: already imported image after thresholding
        original_image: tiff image obtained from the API
        resolution (int): resolution of the image. Defaults to 10
            (10 means 1 pixel = 10m, etc.)

    Returns:
        area: area of the image in squared kilometers
    """
    count = np.count_nonzero(sub_image)
    original_area = original_image.size * resolution**2 / 1_000_000  # km^2
    sub_image_area = sub_image.size / original_image.size * original_area
    return count / sub_image.size * sub_image_area


def merge_two_images(images):
    """Merges two images.

    Args:
        images (list): list of two images

    Returns:
        new_image: concatenated image
    """
    new_image = np.concatenate((images[0], images[1]), axis=1)
    return new_image


def merge_four_images(image_array):
    """
    Takes 4 images of SAME SIZE, merges them together to get a lager field of view
    along with a bigger final picture.

    Args:
        image_array (list): list of the 4 images that need to be merged.
            First image: upper left. Second image: upper right.
            Third image: lower left. Fourth image: lower right.

    Returns:
        final_mage: one final image that has all 4 images merged together
    """
    image1 = image_array[0]
    image2 = image_array[1]
    image3 = image_array[2]
    image4 = image_array[3]
    # get the shapes of the initial images to make an image that is twice as big
    n, m = image1.shape
    final_image = np.zeros((2 * n, 2 * m), np.float64)
    final_image[:n, :m] = image1
    final_image[n:, :m] = image2
    final_image[:n, m:] = image3
    final_image[n:, m:] = image4
    return final_image


def merge_images(n_images, images):
    """Merge images together.

    Args:
        n_images (images): number of images to merge. Can equal 2 or 4.
        images (list): list of images to merge.

    Raises:
        ValueError: Raises an error if the number of images does not match the shape of the list of images
        ValueError: Raises an error if the number of images is not 2 or 4.
    """
    if n_images != len(images):
        raise ValueError(
            "Number of images must be equal to the length of the image array.")

    if n_images == 2:
        return merge_two_images(images)
    elif n_images == 4:
        return merge_four_images(images)
    else:
        raise ValueError("Number of images must be 2 or 4.")
    

def imshow(img, title, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, **kwargs)
    plt.title(title)
    plt.show()
