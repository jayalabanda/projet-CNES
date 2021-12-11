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


def imshow(img, title, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, **kwargs)
    plt.title(title)
    plt.show()


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
    return np.concatenate((images[0], images[1]), axis=1)


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
    if n_images != len(images):
        raise ValueError(
            "Number of images must be equal to the length of the image array.")

    if n_images == 2:
        return merge_two_images(images)
    elif n_images == 4:
        return merge_four_images(images)
    else:
        raise ValueError("Number of images must be 2 or 4.")


def create_sample_coordinates(image, seed, p=0.01):
    np.random.seed(seed)
    rand_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] != 0.:
                rand_image[i, j] = np.random.choice([1, 0], p=[p, 1 - p])
    return rand_image


def get_tci_file_path(image_folder):
    subfolder = [f for f in os.listdir(
        image_folder + "GRANULE") if f[0] == "L"][0]
    image_folder_path = f"{image_folder}GRANULE/{subfolder}/IMG_DATA/R10m"
    image_files = [im for im in os.listdir(
        image_folder_path) if im[-4:] == ".jp2"]
    selected_file = [im for im in image_files if im.split("_")[2] == "TCI"][0]

    return f"{image_folder_path}/{selected_file}"


def get_coordinates_from_pixels(img, h, v, img_folder):
    coords = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0.:
                coords.append((v + i, h + j))

    #pylint: disable=no-member
    tci_file_path = get_tci_file_path(img_folder)
    transform = rasterio.open(tci_file_path, driver='JP2OpenJPEG').transform
    zone_number = int(tci_file_path.split("/")[-1][1:3])
    zone_letter = tci_file_path.split("/")[-1][0]
    utm_x, utm_y = transform[2], transform[5]
    coords_data = []

    for coord in coords:
        x, y = coord
        east = utm_x + x * 10
        north = utm_y + y * - 10
        latitude, longitude = utm.to_latlon(
            east, north, zone_number, zone_letter)
        coords_data.append((latitude, longitude))
    return coords_data


def get_land_cover_data(coords_data, samples, seed):
    # Import the MODIS land cover collection.
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')
    scale = 1000

    for choose in samples:
        np.random.seed(seed)
        cover_data = []
        random_idxs = np.sort(np.random.choice(
            range(len(coords_data)), size=choose, replace=False))

        for i in random_idxs:
            u_lat = coords_data.iloc[i]['latitude']
            u_lon = coords_data.iloc[i]['longitude']
            u_poi = ee.Geometry.Point(u_lon, u_lat)

            try:
                lc_urban_point = lc.first().sample(
                    u_poi, scale).first().get('land_cover_data1').getInfo()
                cover_data.append(lc_urban_point)
            except:
                print('Error with Earth Engine. Land cover could not be retrieved.')

        if None in cover_data:
            cover_data = [i for i in cover_data if i]  # remove None values
        cover_data = dict(Counter(cover_data))
        cover_data = dict(sorted(cover_data.items(), key=lambda x: x[0]))
    return cover_data


def get_labels_colors(cover_data, land_cover_data):
    labels = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Description'].values[0]
        for i in cover_data
    ]
    labels = [i.split(':')[0] for i in labels]

    colors = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Color'].values[0]
        for i in cover_data
    ]
    colors = [mplcols.to_rgb(i) for i in colors]
    return labels, colors


def plot_pie_chart(cover_data, land_cover_data, output_folder, save_fig=True):
    """Plots a pie chart with the data and labels.

    Args:
        data (list): data to be plotted
        labels (list): labels to be plotted
        colors (list): colors to be used for the plot
        output_folder (string): path to the output folder
        save_fig (bool): whether to save the figure or not. Default is True

    Returns:
        None
    """
    labels, colors = get_labels_colors(cover_data, land_cover_data)
    choose = len(cover_data)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_aspect('equal')
    wedges, texts, autotexts = ax.pie(cover_data.values(),
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      textprops=dict(color='w'))
    ax.legend(wedges, labels, title='Land Cover Type', loc='best',
              bbox_to_anchor=(0.9, 0, 0.5, 1),
              prop={'size': 8},
              labelspacing=0.3)
    plt.setp(autotexts, size=6, weight='bold')
    plt.title(f'N = {choose}')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{output_folder}pie_{choose}.png')
    plt.show()


def make_gif(file_path, output_folder, **kwargs):
    """Makes a gif from the images in the file_path.
    The images must be in format 'pie_*.png'
    where '*' is the number of points sampled from the fire.

    Args:
        file_path (string): path to the folder with the images
        output_folder (string): path to the output folder.
            Saves the image to 'output_folder/pie_chart/'

    Returns:
        None
    """
    files = glob.glob(file_path)
    files = [f.split('_')[2].split('.')[0] for f in files]
    files = sorted(files, key=int)
    output = output_folder + 'pie_chart/'

    img, *imgs = [Image.open(f'{output}pie_{f}.png')
                  for f in files]
    img.save(fp=output_folder, format='GIF', append_images=imgs, **kwargs)


def split_image(image, fragment_count):
    n = range(fragment_count)
    frag_size = int(image.shape[0] / fragment_count)
    split_image = {}

    for y, x in itertools.product(n, n):
        split_image[(x, y)] = image[y * frag_size: (y + 1) * frag_size,
                                    x * frag_size: (x + 1) * frag_size]
    return split_image


def plot_split_image(split_image, fragment_count):
    n = range(fragment_count)
    fig, axs = plt.subplots(fragment_count, fragment_count, figsize=(10, 10))
    for y, x in itertools.product(n, n):
        axs[y, x].imshow(split_image[(x, y)])
        axs[y, x].axis('off')
    plt.tight_layout()
    plt.show()
