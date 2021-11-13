import os
import cv2
from datetime import date, datetime, timedelta
import json
import itertools
import zipfile

import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import utm


def get_band(image_folder, band, resolution=10):
    """Returns an image opened with rasterio with the given band and resolution    

    Args:
        image_folder (path): path to the folder containing the image
        resolution (int): Resolution of the image. Defaults to 10.

    Returns:
        img: image with the given band and resolution
    """
    subfolder = [f for f in os.listdir(
        image_folder + "/GRANULE") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}/GRANULE/{subfolder}/IMG_DATA/R{resolution}m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    # retrieve jp2 image file
    selected_file = [im for im in image_files if im.split("_")[2] == band][0]
    # print(selected_file)
    path = os.path.join(image_folder_path,selected_file)
    return path

def download_from_api(df, api, index):
    """Downloads images from the API and saves them to the disk as a zip file.

    Args:
        df (dataframe): dataframe containing the images to download
        api (SentinelAPI): API object
        index (int): index of the image to download
    """
    uuid = df["uuid"].values[index]
    api.download(uuid)

def open_rasterio(image_path):
    """Opens the image with rasterio.
    Args:
        image_path (string): path to the image
    Returns:
        img: image opened with rasterio
    """
    with rasterio.open(image_path, driver='JP2OpenJPEG') as infile:
        img = infile.read(1)
    return img


def threshold_filter(image, threshold):
    """Puts all values below threshold to 0.

    Args:
        image : Already imported image
        threshold (float): Threshold value

    Returns:
        image: image where all values below threshold are set to 0
    """
    image[image < threshold] = 0
    return image


def calculate_area(image):
    """Calculates the area of the image.

    Args:
        image : Already imported image

    Returns:
        area: area of the image
    """
    count = np.count_nonzero(image)
    # ndarray.size = n * m
    ratio = count / image.size
    # multiply ratio by the actual area of the image using coordinates
    # remove the line below once the area using coordinates is calculated has been implemented.
    area = ratio
    return area


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
    """,
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

def select_image_cloud(images_df,cloud_threshold):
    """Select images with cloud cover less than cloud_threshold.
       Return the one with the lowest cloud coverage. 
    
    Arguments:
        images_df {dataframe} -- dataframe containing images
        cloud_threshold {float} -- threshold for cloud coverage
    
    Returns:
        uuid {string} -- uuid of the selected image.
    """
    images_df = images_df[images_df.cloudcoverpercentage < cloud_threshold]
    best_image = images_df[images_df.cloudcoverpercentage == images_df.cloudcoverpercentage.min()]
    uuid = best_image.iloc[0]["uuid"]
    return uuid

def get_best_image_bewteen_dates(date1, date2):
    """Return the image with the lowest cloud cover percentage between two dates.

    Args:
        date1 (datetime): date of the first observation
        date2 (datetime): date of the second observation
    """
    shape = geojson_to_wkt(read_geojson(geojson_path))
    images = api.query(
        shape,
        date=(date1, date2),
        platformname="Sentinel-2",
        processinglevel="Level-2A",
        cloudcoverpercentage=(0, 30)
    )
    return select_image_cloud(images, 0.4)



def get_before_after_images(wildfire_date, observation_interval, min_size):
    """Returns the images before and after the wildfire date.
       It is filtered with a fixed cloud cover percentage at 40%.
    Args:
        wildfire_date (date): date of the wildfire
        observation_interval (int): interval between observations
    
    Returns:
        before_image: image before the wildfire
        after_image: image after the wildfire
    """
    before_date = wildfire_date - timedelta(days=1)
    before_date_one_week_ago = wildfire_date - timedelta(days = observation_interval)
    before_image_uuid = get_best_image_bewteen_dates(before_date_one_week_ago, before_date)
    last_observation_date = wildfire_date + timedelta(days = observation_interval)
    after_image_uuid = get_best_image_bewteen_dates(wildfire_date, last_observation_date)    
    return before_image, after_image

def create_tiff_image(uuid, path, name):
    """Create the tiff image from the uuid.
    Args:
        uuid (string): uuid of the image
        temporary_folder (string): path to the folder where the data will be temporarily stored.
        name (string): name of tiff file
    Returns:
        image: image with the given uuid
    """
    api.download(uuid, path)
    #unzip the file with zipfile
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path)
    image_path_b04 = get_band(path,"B04",resolution=10)
    image_path_b08 = get_band(path,"B08",resolution=10)
    band4 = open_rasterio(image_path_b04)
    band8 = open_rasterio(image_path_b08)
    red = band4.astype('float64')
    nir = band8.astype('float64')
    ndvi = np.where(
        (nir + red) == 0.,
        0.,
        (nir - red) / (nir + red)
    )
    ndvi_img = rasterio.open(
        f'./output/{name}.tiff, 'w', driver='GTiff',
        width=band4.width,
        height=band4.height, 
        count=1,
        crs=band4.crs,
        transform=band4.transform,
        dtype='float64'
    )
    ndvi_img.write(ndvi, 1)
    ndvi_img.close()
    
    return rasterio.open(f'./output/{name}.tiff')