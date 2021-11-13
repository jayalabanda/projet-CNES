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
        band (string): band to be extracted. Can be 'B01', 'B02', for example
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
    return os.path.join(image_folder_path, selected_file)


# def download_from_api(df, api, index):
#     """Downloads images from the API and saves them to the disk as a zip file.

#     Args:
#         df (dataframe): dataframe containing the images to download
#         api (SentinelAPI): API object
#         index (int): index of the image to download
#     """
#     uuid = df["uuid"].values[index]
#     api.download(uuid)


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


def create_tiff_image(api, uuid, title, path, name):
    """Create the tiff image from the uuid.

    Args:
        api (SentinelAPI): API object
        uuid (string): uuid of the image
        title (string): named column in the images dataframe
        path (string): path to save the tiff file
        name (string): name of the tiff file
    Returns:
        image: image with the given uuid
    """
    dirs = os.listdir(path)
    dirs_safe = [safe for safe in dirs if safe[-4:] == "SAFE"]

    if title not in dirs_safe:
        api.download(uuid, path)
        # unzip the file with zipfile
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(path)

    image_path_b04 = get_band(path, "B04", resolution=10)
    image_path_b08 = get_band(path, "B08", resolution=10)

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
        f'./output/{name}.tiff' 'w', driver='GTiff',
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
        sub_image: already imported image after thresholding
        original_image: tiff image obtained from the API
        resolution (int): resolution of the image. Defaults to 10
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


def select_image_cloud(images_df, cloud_threshold=0.4):
    """Select images with cloud cover less than cloud_threshold.
       Return the one with the lowest cloud coverage. 

    Args:
        images_df (dataframe): dataframe containing images
        cloud_threshold (float): threshold for cloud coverage. Defaults to 0.4

    Returns:
        uuid (string): uuid of the selected image.
    """
    images_df = images_df[images_df.cloudcoverpercentage < cloud_threshold]
    best_image = images_df[images_df.cloudcoverpercentage ==
                           images_df.cloudcoverpercentage.min()]
    uuid = best_image.iloc[0]["uuid"]
    title = best_image.iloc[0]["title"]
    return uuid, title


def get_best_image_bewteen_dates(api, date1, date2, geojson_path, cloud_threshold=0.4):
    """Return the image with the lowest cloud cover percentage between two dates.

    Args:
        api (SentinelAPI): API object
        date1 (datetime): date of the first observation
        date2 (datetime): date of the second observation
        geojson_path (string): path to the geojson file
        cloud_threshold (float): threshold for cloud coverage. Defaults to 0.4
    Returns:
        uuid and title of the image
    """
    shape = geojson_to_wkt(read_geojson(geojson_path))
    images = api.query(
        shape,
        date=(date1, date2),
        platformname="Sentinel-2",
        processinglevel="Level-2A",
        cloudcoverpercentage=(0, cloud_threshold)
    )
    return select_image_cloud(images, cloud_threshold)


def get_before_after_images(api, wildfire_date, observation_interval,
                            geojson_path, cloud_threshold=0.4):
    """Returns the images before and after the wildfire date.
       It is filtered with a fixed cloud cover percentage at 40%.

    Args:
        api (SentinelAPI): API object
        wildfire_date (date): date of the wildfire
        observation_interval (int): interval between observations
        geojson_path (string): path to the geojson file
        cloud_threshold (float): threshold for cloud coverage. Defaults to 0.4
    Returns:
        before_image: image before the wildfire
        after_image: image after the wildfire
    """
    before_date = wildfire_date - timedelta(days=1)
    before_date_one_week_ago = wildfire_date - \
        timedelta(days=observation_interval)

    before_image_uuid, title1 = get_best_image_bewteen_dates(
        api, before_date_one_week_ago, before_date,
        geojson_path, cloud_threshold
    )

    last_observation_date = wildfire_date + \
        timedelta(days=observation_interval)

    after_image_uuid, title2 = get_best_image_bewteen_dates(
        api, wildfire_date, last_observation_date,
        geojson_path, cloud_threshold
    )

    before_image = create_tiff_image(
        api=api,
        uuid=before_image_uuid,
        title=title1,
        path="./data",
        name="ndvi_before"
    )
    after_image = create_tiff_image(
        api=api,
        uuid=after_image_uuid,
        title=title2,
        path="./data",
        name="ndvi_after"
    )
    return before_image, after_image
