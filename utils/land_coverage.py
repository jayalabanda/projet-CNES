import glob
import os
from collections import Counter

import ee
import matplotlib.colors as mplcols
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import utm
from PIL import Image


def get_tci_file_path(image_folder):
    """Get the path to the True Color Image (TCI) file.

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
    return f"{image_folder_path}/{selected_file}"


def create_sample_coordinates(image, seed=None, p=0.01):
    """Creates a sample of coordinates from the image.

    The number of points equals approximately `n * m * p` where `n`
    is the number of rows, `m` is the number of columns, and `p` is
    the percentage of points to be selected.

    Args:
        image (image): image from which the coordinates are sampled
        seed (int, optional): seed for the random number generator. Defaults to `None`.
        p (float, optional): Sampling rate. Defaults to 0.01.

    Returns:
        rand_image: image with the randomly selected points
    """
    np.random.seed(seed)
    rand_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] != 0.:
                rand_image[i, j] = np.random.choice([1, 0], p=[p, 1 - p])
    return rand_image


def get_coordinates_from_pixels(img, h, v, img_folder):
    """Retreives the latitude and longitude coordinates from the pixels.

     Args:
         img (image): image from which the coordinates are retrieved
         h, v (int): horizontal and vertical offsets
         img_folder (string): path to the folder where the image is retrieved from

     Returns:
         coordinates: list of coordinate tuples (latitude, longitude)
     """
    coords = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0.:
                coords.append((v + j, h + i))

    tci_file_path = get_tci_file_path(img_folder)
    transform = rasterio.open(tci_file_path, driver='JP2OpenJPEG').transform
    zone_number = int(tci_file_path.split("/")[-1][1:3])
    zone_letter = tci_file_path.split("/")[-1][0]
    utm_x, utm_y = transform[2], transform[5]

    coordinates = []
    for coord in coords:
        x, y = coord
        east = utm_x + x * 10
        north = utm_y + y * - 10
        latitude, longitude = utm.to_latlon(
            east, north, zone_number, zone_letter)
        coordinates.append((latitude, longitude))
    return coordinates


def select_land_cover_data():
    while True:
        try:
            print(
                """Please select the land cover data you want to use.
                The choices are:
                1. MODIS Land Cover (2018, 500m)
                2. ESA World Cover (2020, 10m)
                3. Copernicus Global Land Service (2019, 100m)
                4. Copernicus Corine Land Cover (2018, 100m)"""
            )
            choice = int(input('Select land cover data: '))
            if choice == 1:
                return ee.ImageCollection('MODIS/006/MCD12Q1')
            elif choice == 2:
                return ee.ImageCollection("ESA/WorldCover/v100")
            elif choice == 3:
                return ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
            elif choice == 4:
                return ee.ImageCollection("COPERNICUS/CORINE/V20/100m/2018")
        except ValueError:
            print('Invalid input. Please try again.')


def get_land_cover_data(coords_data, size, seed=None):
    """Retrieves the land cover data from the coordinates.

    The keys in the dictionary are the labels and the values are the
    number of points in each land cover type.

    Args:
        coords_data (list): list of coordinate tuples (latitude, longitude)
        size (int): number of points to be sampled from the fire
        seed (int, optional): seed for the random number generator. Defaults to `None`.

    Returns:
        cover_data (dict): dictionary with the land cover data"""
    # Import the MODIS land cover collection.
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')
    scale = 1000

    np.random.seed(seed)
    cover_data = []
    random_idxs = np.sort(np.random.choice(
        range(len(coords_data)), size=size, replace=False))

    for i in random_idxs:
        u_lat = coords_data.iloc[i]['latitude']
        u_lon = coords_data.iloc[i]['longitude']
        u_poi = ee.Geometry.Point(u_lon, u_lat)

        try:
            # access the earth engine API
            lc_urban_point = lc.first().sample(
                u_poi, scale).first().get('LC_Type1').getInfo()
            cover_data.append(lc_urban_point)
        except:
            print('Error with Earth Engine. Land cover could not be retrieved.')

    if None in cover_data:
        cover_data = [i for i in cover_data if i]  # remove None values

    cover_data = dict(Counter(cover_data))
    # sort by keys so that the output plots have the same structure
    cover_data = dict(sorted(cover_data.items(), key=lambda x: x[0]))
    return cover_data


def get_labels_colors(cover_data, land_cover_data):
    """Retrieves the labels and colors for the pie chart.

    The land cover data is a dataframe with columns `Value`, `Color`, and `Description`.

    Args:
        cover_data (dict): dictionary with the land cover data
        land_cover_data (dataframe): dictionary with the colors by type of land coverage

    Returns:
        labels: labels for the pie chart
        colors: colors for the pie chart
    """
    labels = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Description'].values[0]
        for i in cover_data
    ]
    # The text after ':' in the description gives more detals but is not needed
    labels = [i.split(':')[0] for i in labels]

    colors = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Color'].values[0]
        for i in cover_data
    ]
    colors = [mplcols.to_rgb(i) for i in colors]
    return labels, colors


def plot_pie_chart(cover_data, labels, colors, size, fire_name,
                   out_folder=None, save_fig=True):
    """Plots a pie chart with the given data and labels.

    Args:
        data (list): data to be plotted
        labels (list): labels to be plotted
        colors (list): colors to be used for the plot
        output_folder (string): path to the output folder
        save_fig (bool): whether to save the figure or not. Default is `True`
    """
    _, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_aspect('equal')
    wedges, _, autotexts = ax.pie(
        cover_data.values(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops=dict(color='w')
    )
    ax.legend(
        wedges,
        labels,
        title='Land Cover Type',
        loc='best',
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        prop={'size': 8},
        labelspacing=0.3
    )
    plt.setp(autotexts, size=6, weight='bold')
    plt.title(f'N = {size}')
    plt.tight_layout()
    if save_fig:
        if out_folder is None:
            out_folder = 'output/pie_chart_' + fire_name + '/'
        plt.savefig(f'{out_folder}pie_{size}.png')
    plt.show()


def create_plots(samples, coordinates, land_cover_data, seed=None, **kwargs):
    """Creates the plots for the given samples.

    Args:
        samples (list): list of samples to be drawn
        coordinates (dataframe): dataframe of coordinates
        land_cover_data (dataframe): dataframe with colors and descriptions of land coverage
        seed (int): seed for the random number generator. Defaults to `None`.
        **kwargs: keyword arguments for the `plot_pie_chart` function
    """
    for size in samples:
        print(f'Retrieving {size} samples...')
        cover_data = get_land_cover_data(coordinates, size, seed)
        print(cover_data)
        labels, colors = get_labels_colors(cover_data, land_cover_data)
        plot_pie_chart(cover_data, labels, colors, size,
                       fire_name=kwargs['fire_name'],
                       out_folder=kwargs['out_folder'],
                       save_fig=kwargs['save_fig'])


def make_pie_chart_gif(fire_name, file_path=None, **kwargs):
    """Makes a gif from the images in the file_path.

    The filenames must be in format 'pie_*.png'
    where '*' is the number of points sampled from the fire.

    This assumes the PNG files are saved using the `save_fig` parameter.

    Args:
        file_path (string): path to the folder with the images
        output_folder (string): path to the output folder.
    """
    if file_path is None:
        file_path = 'output/pie_chart_' + fire_name + '/'

    files = glob.glob(file_path + 'pie_*.png')
    files = [f.split('_')[-1].split('.')[0] for f in files]
    files = sorted(files, key=int)
    out_file = file_path + fire_name + '.gif'

    img, *imgs = [Image.open(f'{file_path}pie_{f}.png')
                  for f in files]
    img.save(fp=out_file, format='GIF', append_images=imgs, **kwargs)