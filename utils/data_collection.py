import datetime as dt
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from sentinelsat import geojson_to_wkt, read_geojson
from utils.image_processing import imshow


def get_band(image_folder, band, resolution):
    """Returns a JP2 image path for the given band and resolution.

    The relevant images are inside 'GRANULE/.../IMG_DATA/'

    Args:
        image_folder (str): path to the folder containing the image
        band (string): band to be extracted. Can be `'B01'`, `'B02'`,
        `B03` for example
        resolution (int): resolution of the image. Default is 10. Other
        choices are 20 and 60

    Returns:
        path to the JP2 image (str)
    """
    subfolder = [f for f in os.listdir(image_folder + "/GRANULE/")
                 if f[0] == "L"][0]
    image_folder_path = image_folder + "/GRANULE/" + subfolder +\
        "/IMG_DATA/" + f"R{resolution}m"
    image_files = [im for im in os.listdir(image_folder_path)
                   if im[-4:] == ".jp2"]

    # retrieve path to JP2 image file
    selected_file = [im for im in image_files
                     if im.split("_")[2] == band][0]
    return image_folder_path + "/" + selected_file


def get_dataframe_between_dates(api, date1, date2, geojson_path,
                                cloud_threshold=40):
    """Retrieve a dataframe containing all the information
    between two dates.

    Args:
        api (SentinelAPI): API object
        date1 (datetime): date of the first observation
        date2 (datetime): date of the second observation
        geojson_path (string): path to the geojson file
        cloud_threshold (float): threshold for cloud coverage. Default is 40

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
    print("Retrieved dataframe.")
    print(f"Number of images between {date1} and {date2}: {len(images_df)}.")
    return images_df


def minimize_dataframe(df):
    """Creates a score for the dataframe using a weighted average of
    cloud cover, vegetation cover, and water presence.

    More vegetation is better, and less water presence and
    fewer clouds is better.

    Note that the coefficients are arbitrary but allow for
    giving more importance to the vegetation.

    Args:
        df (dataframe): dataframe containing all the information

    Returns:
        df_min: dataframe with a score for each image/row. Lower is better.
    """
    coeffs = [2, 0.1, 4]
    key_columns = ["cloudcoverpercentage",
                   "vegetationpercentage",
                   "waterpercentage"]

    print("Minimizing dataframe.")
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
        df (dataframe): dataframe containing a `size` column

    Returns:
        df: dataframe with the size in MB
    """
    # we convert the data types
    df = df.convert_dtypes()

    # the "size" column is of type string, for example "1.1 GB" or "980 MB"
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
    print("Dropping images with low vegetation.")
    df = df[df["vegetationpercentage"] >= 45.]

    # we retrieve the image with the best score as long as its size is
    # large enough, since images with no data are smaller
    i = 0
    size = df["size"].values[i]
    while size < 980. and i <= df.shape[0]:
        i += 1
        size = df["size"].values[i]

    uuid = df.iloc[i]["uuid"]
    title = df.iloc[i]["title"]

    print('Retrieved best uuid and title from the dataframe:')
    print(f"uuid: {uuid}\ntitle: {title}\n")

    key_cols = ["cloudcoverpercentage",
                "vegetationpercentage",
                "waterpercentage", "score",
                "ingestiondate", "size"]
    print(df.loc[uuid][key_cols])
    print("\n")

    return uuid, title


def download_from_api(api, uuid, title, path='data/'):
    """Download the image from the API, unzips the folder,
    and deletes the zip file.

    Args:
        api (SentinelAPI): API object
        uuid (string): uuid of the image
        title (string): title of the image (named column
        in the dataframe)
        path (string): path to save the image. Default is '`data/'`
    """
    if not os.path.exists(path):
        os.makedirs(path)
    dirs = os.listdir(path)
    dirs_safe = [safe for safe in dirs if safe[-4:] == "SAFE"]

    # the name of the downloaded file is 'title + .SAFE'
    img_folder = title + '.SAFE'
    if img_folder not in dirs_safe:
        print("Attempting to download image from the API.")
        api.download(uuid, path)
    else:
        print("Image already downloaded.")

    # check that the zip file has been downloaded
    dirs = os.listdir(path)
    zips = any(".zip" in dr for dr in dirs)
    if zips:
        # unzip the file with zipfile
        path_to_zip = path + title + ".zip"
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(path)
        print("Unzipped file.")

        os.remove(path_to_zip)
        print("Deleted zip file.")


def open_rasterio(image_path, driver='JP2OpenJPEG'):
    """Opens the given image with rasterio.

    Args:
        image_path (string): path to the JP2 image

    Returns:
        img: image opened with rasterio
    """
    with rasterio.open(image_path, driver=driver) as infile:
        img = infile.read(1).astype('float64')
    return img


def calculate_ndvi(red_band, nir_band):
    """Calculates the NDVI of a given image.

    The NDVI is calculated as:
        `(nir - red) / (nir + red)`
        or `(B08 - B04) / (B08 + B04)`
    where `B04` is the red band, and `B08` is the near-infrared band.

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
    """Create the TIFF image from the uuid.

    This function creates the TIFF image. Be mindful of the size of the file.

    Args:
        path (string): path to the JP2 files
        when (string): name of the TIFF file. It defaults to `'before'` or
        `'after'` according to the date retrieved from the API
        fire_name (string): name of the fire
        output_folder (string): path to save the TIFF file.
        Default is `'output/'`

    Returns:
        image: opened image with the given uuid
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = f'{output_folder}{when}_{fire_name}.tiff'

    image_path_b4 = get_band(path, 'B04', resolution=10)
    image_path_b8 = get_band(path, 'B08', resolution=10)
    temp_band = rasterio.open(image_path_b4, driver='JP2OpenJPEG')

    red = open_rasterio(image_path_b4)
    nir = open_rasterio(image_path_b8)
    ndvi = calculate_ndvi(red, nir)

    # create the TIFF file
    with rasterio.open(fp=output_file,
                       mode='w', driver='GTiff',
                       width=temp_band.width,
                       height=temp_band.height,
                       count=1,
                       crs=temp_band.crs,
                       transform=temp_band.transform,
                       dtype='float64') as ndvi_img:
        # we only need one band which corresponds to the NDVI
        ndvi_img.write(ndvi, 1)
    temp_band.close()
    print(f'Created image from {when} the fire.')


def get_image(api, wildfire_date, observation_interval,
              path, when=None, **kwargs):
    """Get the image from the API and create the TIFF file.

    Args:
        api (SentinelAPI): API object
        wildfire_date (datetime): date of the wildfire
        observation_interval (int): interval of observation in days
        path (string): path to save the image. Defaults to `'./data/'`
        when (string): name of the TIFF file, either `'before'` or `'after'`

        **kwargs: keyword arguments are:
            geojson_path (string): path to the geojson file
            cloud_threshold (int): threshold for cloud cover.
            Default is 40
            output_folder (string): path to save the TIFF file.
            Default is `'output/'`
            fire_name (string): name of the TIFF file.
    """
    if when not in ['before', 'after']:
        raise ValueError(
            f"{when} is not a valid value. It should be 'before' or 'after'."
        )

    if when == 'before':
        # create dates around the wildfire date
        before_date_delta = wildfire_date - \
            dt.timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api,
            before_date_delta,
            wildfire_date,
            geojson_path=kwargs['geojson_path'],
            cloud_threshold=kwargs['cloud_threshold'],
        )

    elif when == 'after':
        last_observation_date = wildfire_date + \
            dt.timedelta(days=observation_interval)

        df = get_dataframe_between_dates(
            api,
            wildfire_date,
            last_observation_date,
            geojson_path=kwargs['geojson_path'],
            cloud_threshold=kwargs['cloud_threshold']
        )

    df = minimize_dataframe(df)
    uuid, title = get_uuid_title(df)
    download_from_api(api, uuid, title, path)
    image_folder = path + title + ".SAFE"

    create_ndvi_tiff_image(
        path=image_folder,
        when=when,
        fire_name=kwargs['fire_name'],
        output_folder=kwargs['output_folder']
    )


def get_before_after_images(**kwargs):
    """Returns the images before and after the wildfire date.

    Multiple keyword arguments are required that are passed
    to the `'get_image'` function.

    Please refer to the `'get_image'` function for more details.
    """
    get_image(when='before', **kwargs)
    print("-" * 30)
    get_image(when='after', **kwargs)


def plot_ndvi_difference(output_folder, fire_name,
                         save_diff=False, figsize=(12, 8)):
    """Plots and returns the NDVI difference between the images.

    Args:
        output_folder (str): path to the folder where the images are stored
        fire_name (str): name of the fire
        save_diff (bool): whether to save the difference image. Defaults to `False`
        figsize (tuple): size of the figure

    Returns:
        difference: NDVI difference between the images
    """
    before = rasterio.open(f'{output_folder}before_{fire_name}.tiff').read(1)
    after = rasterio.open(f'{output_folder}after_{fire_name}.tiff').read(1)
    difference = before - after

    if save_diff:
        # save the difference image
        output_file = f'{output_folder}ndvi_difference_{fire_name}.tiff'
        if not os.path.exists(output_file):
            with rasterio.open(fp=output_file,
                               mode='w', driver='GTiff',
                               width=before.shape[1],
                               height=before.shape[0],
                               count=1,
                               crs=before.crs,
                               transform=before.transform,
                               dtype='float64') as diff_img:
                diff_img.write(difference, 1)

    plt.figure(figsize=figsize)
    imshow(difference, "NDVI Difference")
    return difference


def check_downloaded_data(path, output_folder, fire_name):
    """Check if the data has already been downloaded.

    Args:
        output_folder (str): path to the folder where the images are stored

    Returns:
        bool: `True` if the data is downloaded, `False` otherwise
    """
    dirs = os.listdir(path)
    dirs_safe = [safe for safe in dirs if safe[-4:] == "SAFE"]
    before_exists = os.path.exists(f'{output_folder}before_{fire_name}.tiff')
    after_exists = os.path.exists(f'{output_folder}after_{fire_name}.tiff')

    if len(dirs_safe) == 2 or before_exists and after_exists:
        print("Data has already been downloaded.")
        return True
    return False
