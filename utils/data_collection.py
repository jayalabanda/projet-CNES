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

def get_band(image_folder, band, resolution):
    """Returns an image path for the given band and resolution.

    Args:
        image_folder (path): path to the folder containing the image
        band (string): band to be extracted. Can be 'B01', 'B02', for example
        resolution (int): Resolution of the image. Defaults to 10.

    Returns:
        img: image with the given band and resolution
    """
    subfolder = [f for f in os.listdir(
        image_folder + "/GRANULE/") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}/GRANULE/{subfolder}/IMG_DATA/R{resolution}m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    # retrieve jp2 image file
    selected_file = [im for im in image_files if im.split("_")[2] == band][0]
    return image_folder_path + "/" + selected_file


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
    print("Retrieved dataframe.\n")
    print(f"Number of images between {date1} and {date2}: {len(images_df)}")
    return images_df

def minimize_dataframe(df):
    """Creates a score for the dataframe using a weighted average of
        cloud cover, vegetation cover, and water presence.
        More vegetation is better, and less water and cloud is better.

    Args:
        df (dataframe): dataframe containing all the information

    Returns:
        df: dataframe with a score for each image. Lower is better.
    """
    # the coefficients are arbitrary but allow for
    # giving more importance to the vegeation presence
    coeffs = [2, 0.1, 4]
    key_columns = ["cloudcoverpercentage",
                   "vegetationpercentage",
                   "waterpercentage"]

    print("Minimizing dataframe...\n")
    df_min = df.copy()
    score = (df_min[key_columns[0]] * coeffs[0]) +\
            (df_min[key_columns[1]] * coeffs[1]) +\
            (df_min[key_columns[2]] * coeffs[2]) / sum(coeffs)

    df_min["score"] = score
    df_min.sort_values(by="score", inplace=True, ascending=True)
    return df_min


def convert_size(df):
    """Converts the size of the images from GB to MB.

    Args:
        df (dataframe): dataframe

    Returns:
        df: dataframe with the size in MB
    """
    # we convert the data types
    df = df.convert_dtypes()

    # the "size" column is of type string
    # if the unit of "size" is 'GB', we convert it to 'MB'
    cond = df["size"].apply(lambda x: x.split(" ")[1]) == 'GB'
    df["size"] = np.where(
        cond,
        df["size"].apply(lambda x: float(x.split(" ")[0]) * 1024),
        df["size"].apply(lambda x: float(x.split(" ")[0]))
    )
    return df


def get_uuid_title(df):
    """Returns the uuid and title of the best image.

    Args:
        df (dataframe): dataframe containing all the information

    Returns:
        uuid: uuid of the best image
        title: title of the best image
    """
    df = convert_size(df)
    # we drop the images with low vegetation
    print("Dropping images with low vegetation.\n")
    df = df[df["vegetationpercentage"] >= 45.]

    # we retrieve the image with the best score as long as its size is
    # large enough, since images with no data are smaller
    i = 0
    size = df["size"].values[i]
    while size < 900. and i <= df.shape[0]:
        i += 1
        size = df["size"].values[i]

    uuid = df.iloc[i]["uuid"]
    title = df.iloc[i]["title"]
    print(
        f"Retrieved best uuid and title from the dataframe on row {i + 1}.\n"
    )
    print(f"uuid: {uuid}, title: {title}")

    key_cols = ["cloudcoverpercentage",
                "vegetationpercentage",
                "waterpercentage", "score",
                "ingestiondate", "size"]
    print(df.loc[uuid][key_cols])
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

    # the name of the downloaded file is 'title + .SAFE'
    if (title + '.SAFE') not in dirs_safe:
        print("Downloading image from the API.")
        api.download(uuid, path)

    # check that the zip file has been downloaded
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
        

def get_image(api, wildfire_date, observation_interval,
              path, when=None, **kwargs):
    """Get the image from the API.

    Args:
        api (SentinelAPI): API object
        wildfire_date (string): date of the wildfire
        observation_interval (string): interval of the observation
        path (string): path to save the image. Defaults to './data/'
        when (string): name of the tiff file, either 'before' or 'after'
        **kwargs: keyword arguments are:
            - geojson_path (string): path to the geojson file
            - cloud_threshold (int): threshold for cloud cover. Defaults to 40.
            - band1 (string): band to be extracted. Can be 'B01', 'B02', for example
            - band2 (string): band to be extracted. Can be 'B01', 'B02', for example
            - resolution (int): Resolution of the image. Defaults to 10.
            - output_folder (string): path to save the tiff file. Defaults to 'output/'
            - name (string): name of the tiff file.

    Returns:
        opened tiff file
    """
    if when not in ['before', 'after']:
        raise ValueError(
            f"{when} is not a valid value. It should be 'before' or 'after'."
        )

    if when == 'before':
        # create dates around the wildfire
        before_date = wildfire_date - dt.timedelta(days=1)
        before_date_one_week_ago = wildfire_date - \
            dt.timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api, before_date_one_week_ago, before_date,
            geojson_path=kwargs['geojson_path'],
            cloud_threshold=kwargs['cloud_threshold'],
        )

    elif when == 'after':
        last_observation_date = wildfire_date + \
            dt.timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api, wildfire_date, last_observation_date,
            geojson_path=kwargs['geojson_path'],
            cloud_threshold=kwargs['cloud_threshold']
        )

    # retrieve the uuid of the best image then download it
    df = minimize_dataframe(df)
    uuid, title = get_uuid_title(df)
    download_from_api(api, uuid, title)
    image_folder = path + title + ".SAFE"

    return create_ndvi_tiff_image(
        path=image_folder, when=when,
        fire_name=kwargs['fire_name'],
        output_folder=kwargs['output_folder']
    )
    
def get_before_after_images(**kwargs):
    """Returns the images before and after the wildfire date.
    Multiple keyword arguments are required that ares passed
    to the 'get_image' function.

    Args:
        api (SentinelAPI): API object
        path (string): path to save the downloaded files. Default is 'data/'
        wildfire_date (date): date of the wildfire
        geojsont_path (string): path to the geojson file
        observation_interval (int): interval between observations in days
        output_folder (string): path to the output folder. Default is 'output/'
        band1 (string): band to be used for the NDVI calculation. Default is 'B04'
        band2 (string): band to be used for the NDVI calculation. Default is 'B08'
        cloud_threshold (int): threshold for cloud coverage. Default is 40
        resolution (int): resolution of the images. Default is 10

    Returns:
        before_image: image before the wildfire
        after_image: image after the wildfire
    """
    before_image = get_image(
        when='before', **kwargs
    )
    print("Created image from before the fire.\n")

    print("-" * 30)
    after_image = get_image(
        when='after', **kwargs
    )
    print("Created image from after the fire.\n")

    return before_image, after_image


def get_tci_file_path(image_folder):
    """Get the path to the tci file.
       Gives more context and information on the selected zone.

    Args:
        image_folder (path): path to the image folder

    Returns:
        path: path of the tci file
    """
    subfolder = [f for f in os.listdir(
        image_folder + "GRANULE") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}GRANULE/{subfolder}/IMG_DATA/R10m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    selected_file = [im for im in image_files if im.split("_")[2] == "TCI"][0]
    path = f"{image_folder_path}/{selected_file}"
    return path