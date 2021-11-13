import os
import cv2
import json
import itertools

import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import utm


def select_image_cloud(images_df, cloud_threshold):
    """Select images with cloud cover less than cloud_threshold.
       Return the one with the lowest cloud coverage. 

    Arguments:
        images_df (dataframe) -- dataframe containing images
        cloud_threshold (float) -- threshold for cloud coverage

    Returns:
        uuid (string) -- uuid of the selected image.
    """
    images_df = images_df[images_df.cloudcoverpercentage < cloud_threshold]
    best_image = images_df[images_df.cloudcoverpercentage ==
                           images_df.cloudcoverpercentage.min()]

    return best_image.iloc[0]["uuid"]


def download_from_api(df, api, index):
    """Downloads images from the API and saves them to the disk as a zip file.

    Args:
        df (dataframe): dataframe containing the images to download
        api (SentinelAPI): API object
        index (int): index of the image to download
    """
    uuid = df["uuid"].values[index]
    api.download(uuid)


def get_band(image_folder, band, resolution=10):
    """Returns an image opened with rasterio with the given band and resolution    

    Args:
        image_folder (str): path to the folder containing the image
        band (str): name of the band to be extracted
        resolution (int): Resolution of the image. Defaults to 10.

    Returns:
        img: image with the given band and resolution
    """
    subfolder = [f for f in os.listdir(
        image_folder + "/GRANULE") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}/GRANULE/{subfolder}/IMG_DATA/R{resolution}m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    selected_file = [im for im in image_files if im.split("_")[2] == band][0]

    with rasterio.open(f"{image_folder_path}/{selected_file}", driver='JP2OpenJPEG') as infile:
        img = infile.read(1)

    return img


def threshold_filter(image, threshold):
    """Puts all values below threshold to 0.

    Args:
        image: already imported image
        threshold (float): threshold value

    Returns:
        image: image where all values below threshold are set to 0
    """
    image[image < threshold] = 0
    return image


def calculate_area(sub_image, original_image, resolution=10):
    """Calculates the surface of the burnt area.

    Args:
        sub_image: already imported image
        original_image: already imported image
        resolution (int): resolution of the image. Defaults to 10.
            (10m = 10, 20m = 20, 60m = 60)

    Returns:
        area: area of the image in squared kilometers
    """
    count = np.count_nonzero(sub_image)
    original_area = original_image.size * resolution**2 / 1_000_000  # km^2
    sub_image_area = sub_image.size / original_image.size * original_area
    return count / sub_image.size * sub_image_area


def merge_four_images(image_array):
    """
    Takes 4 images of SAME SIZE, merges them together to get a lager field of view
    along with a bigger final picture.

    Args:
        image_array (list): list of the 4 images that need to be merged.
        First image: upper left, second image: upper right.
        Third image: lower left, fourth image: lower right.

    Returns:
        final_mage: one final image that has all 4 images merged together
    """
    image1 = image_array[0]
    image2 = image_array[1]
    image3 = image_array[2]
    image4 = image_array[3]
    # get the shapes of the initial images to make an image that is twice as big
    n = image1.shape[0]
    m = image1.shape[1]
    final_image = np.zeros((2 * n, 2 * m), np.uint8)
    final_image[:n, :m] = image1
    final_image[n:, :m] = image2
    final_image[:n, m:] = image3
    final_image[n:, m:] = image4
    return final_image
