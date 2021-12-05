import utm
from sentinelsat import read_geojson, geojson_to_wkt
import numpy as np
import os
import cv2
import datetime as dt
import itertools
import zipfile
import json

import rasterio
from rasterio import plot
import matplotlib.pyplot as plt


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


def get_band(image_folder, band, resolution):
    """Returns an image opened with rasterio with the given band and resolution.

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


def create_tiff_image(path, when, band1, band2, name, output_folder='output/', resolution=10):
    """Create the tiff image from the uuid.
        This function creates the tiff image.
        Be mindful of the size of the file.

    Args:
        path (string): path to save the tiff file
        when (string): name of the tiff file. Usually 'before' or 'after'

    Returns:
        image: opened image with the given uuid
    """
    image_path_b1 = get_band(path, band1, resolution=resolution)
    image_path_b2 = get_band(path, band2, resolution=resolution)

    print("First image selected for NDVI:", image_path_b1.split("/")[-1])
    print("Second image selected for NDVI:", image_path_b2.split("/")[-1])

    first_band = rasterio.open(image_path_b1, driver='JP2OpenJPEG')
    second_band = rasterio.open(image_path_b2, driver='JP2OpenJPEG')

    red = first_band.read(1).astype('float64')
    nir = second_band.read(1).astype('float64')

    ndvi = calculate_ndvi(red, nir)

    # create the tiff file
    ndvi_img = rasterio.open(
        fp=f'{output_folder}{when}_{name}.tiff',
        mode='w', driver='GTiff',
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

    return rasterio.open(f'{output_folder}{when}_{name}.tiff').read(1)


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
            f"{when} is not a valid value. It should be 'before' or 'after'"
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
    return create_tiff_image(
        path=image_folder, when=when, band1=kwargs['band1'],
        band2=kwargs['band2'], name=kwargs['name'],
        output_folder=kwargs['output_folder'],
        resolution=kwargs['resolution']
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


def get_tci_file_path(image_folder):
    subfolder = [f for f in os.listdir(
        image_folder + "GRANULE") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}GRANULE/{subfolder}/IMG_DATA/R10m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    selected_file = [im for im in image_files if im.split("_")[2] == "TCI"][0]

    return f"{image_folder_path}/{selected_file}"
