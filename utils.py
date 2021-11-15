import utm
from sentinelsat import read_geojson, geojson_to_wkt
import numpy as np
import os
import cv2
from datetime import timedelta
import itertools
import zipfile

import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def get_dataframe_between_dates(api, date1, date2, geojson_path, cloud_threshold=40):
    """Retrieve a dataframe containing all the information between two dates.

    Args:
        api (SentinelAPI): API object
        date1 (datetime): date of the first observation
        date2 (datetime): date of the second observation
        geojson_path (string): path to the geojson file
        cloud_threshold (float): threshold for cloud coverage. Defaults to 40

    Returns:
        a dataframe with information about the images between the two dates
    """
    shape = geojson_to_wkt(read_geojson(geojson_path))
    images = api.query(
        area=shape,
        date=(date1, date2),
        platformname="Sentinel-2",
        processinglevel="Level-2A",
        cloudcoverpercentage=(0, cloud_threshold)
    )
    images_df = api.to_dataframe(images)
    print(f"Number of images between {date1} and {date2}: {len(images_df)}")
    print("Retrieved dataframe.")
    return images_df


def minimize_dataframe(df):
    """Creates a score for the dataframe using a weighted average of
        cloud cover, vegetation cover, and water presence.

    Args:
        df (dataframe): dataframe containing all the information

    Returns:
        uuid: uuid of the best image
        title: title of the best image
    """
    coeffs = [2, 0.1, 4]
    key_columns = ["cloudcoverpercentage",
                   "vegetationpercentage",
                   "waterpercentage"]

    df_min = df.copy()
    score = (df[key_columns[0]] * coeffs[0]) +\
            (df[key_columns[1]] * coeffs[1]) +\
            (df[key_columns[2]] * coeffs[2]) / sum(coeffs)

    df_min["score"] = score
    df_min.sort_values(by="score", inplace=True, ascending=True)

    uuid = df_min.iloc[0]["uuid"]
    title = df_min.iloc[0]["title"]
    return uuid, title


def download_from_api(api, uuid, title, path='./data/'):
    """Download the image from the API.

    Args:
        api (SentinelAPI): API object
        uuid (string): uuid of the image
        title (string): title of the image (named column in the dataframe)
        path (string): path to save the image. Defaults to './data/'

    Returns:
        none
    """
    dirs = os.listdir(path)
    dirs_safe = [safe for safe in dirs if safe[-4:] == "SAFE"]

    if (title + '.SAFE') not in dirs_safe:
        print("Downloading image from the API.")
        api.download(uuid, path)

    dirs = os.listdir(path)
    zips = any(".zip" in dir for dir in dirs)
    if zips:
        # unzip the file with zipfile
        path_to_zip = path + title + ".zip"
        print("Unzipping file.")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(path)
        print("Deleting zip file.")
        os.remove(path_to_zip)


def get_band(image_folder, band, resolution=10):
    """Returns an image opened with rasterio with the given band and resolution.

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
    return image_folder_path + "/" + selected_file


def open_rasterio(image_path):
    """Opens the image with rasterio.

    Args:
        image_path (string): path to the tiff image

    Returns:
        img: image opened with rasterio
    """
    with rasterio.open(image_path, driver='JP2OpenJPEG') as infile:
        img = infile.read(1)
    return img


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


def create_tiff_image(path, when, band1, band2, output_folder, resolution=10):
    """Create the tiff image from the uuid.
        This function downloads the image from the API and saves it to the disk.
        Be mindful of the size of the file.

    Args:
        api (SentinelAPI): API object
        uuid (string): uuid of the image
        title (string): named column in the images dataframe
        path_to_zip (string): path to the zip file
        path (string): path to save the tiff file
        when (string): name of the tiff file. Usually 'before' or 'after'

    Returns:
        image: image with the given uuid
    """
    image_path_b1 = get_band(path, band1, resolution=resolution)
    image_path_b2 = get_band(path, band2, resolution=resolution)

    print("First image selected:", image_path_b1.split("/")[-1])
    print("Second image selected:", image_path_b2.split("/")[-1])

    first_band = rasterio.open(image_path_b1, driver='JP2OpenJPEG')
    second_band = rasterio.open(image_path_b2, driver='JP2OpenJPEG')

    red = first_band.read(1).astype('float64')
    nir = second_band.read(1).astype('float64')

    ndvi = calculate_ndvi(red, nir)

    ndvi_img = rasterio.open(
        fp=f'./output/{when}.tiff', mode='w', driver='GTiff',
        width=first_band.width,
        height=first_band.height,
        count=1,
        crs=first_band.crs,
        transform=first_band.transform,
        dtype='float64'
    )

    first_band.close()
    second_band.close()

    # we only need one band which corresponds to the NDVI
    ndvi_img.write(ndvi, 1)
    ndvi_img.close()

    return rasterio.open(f'{output_folder}/{when}.tiff')


# def get_image(api, geojson_path, wildfire_date, observation_interval,
#               band1, band2, output_folder, path='./data/', when='before',
#               cloud_threshold=40, resolution=10):
def get_image(api, wildfire_date, observation_interval,
              when='before', *args, **kwargs):
    if when == 'before':
        before_date = wildfire_date - timedelta(days=1)
        before_date_one_week_ago = wildfire_date - \
            timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api, before_date_one_week_ago, before_date,
            *args, **kwargs
        )
        uuid, title = minimize_dataframe(df)
        print(f'Image before the wildfire: {title}')
        download_from_api(api, uuid, title)

        return create_tiff_image(
            when='before', *args, **kwargs,
        )

    elif when == 'after':
        last_observation_date = wildfire_date + \
            timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api, wildfire_date, last_observation_date,
            *args, **kwargs
        )
        uuid, title = minimize_dataframe(df)
        print(f'Image after the wildfire: {title}')
        download_from_api(api, uuid, title)

        return create_tiff_image(
            when='after',
            *args, **kwargs
        )

    else:
        raise ValueError(
            f"{when} is not a valid value. It should be 'before' or 'after'"
        )


# def get_before_after_images(api, geojson_path, wildfire_date, observation_interval,
#                             band1, band2, output_folder,
#                             cloud_threshold=40, resolution=10):
def get_before_after_images(*args, **kwargs):
    """Returns the images before and after the wildfire date.
       It is filtered with a fixed cloud cover percentage at 40%.

    Args:
        api (SentinelAPI): API object
        wildfire_date (date): date of the wildfire
        observation_interval (int): interval between observations
        geojson_path (string): path to the geojson file
        cloud_threshold (float): threshold for cloud coverage. Defaults to 40

    Returns:
        before_image: image before the wildfire
        after_image: image after the wildfire
    """
    before_image = get_image(
        *args, **kwargs, when='before'
    )
    print("Created image from before the fire.")

    after_image = get_image(
        *args, **kwargs, when='after'
    )
    print("Created image from after the fire.")

    return before_image, after_image


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
    """Calculates the surface, in squared kilometers, of the burnt area.

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
