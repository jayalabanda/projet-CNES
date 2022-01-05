import glob
import os
import webbrowser
from collections import Counter

import ee
import matplotlib.colors as mplcols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import utm
from PIL import Image


def get_tci_file_path(image_folder):
    """Get the path to the True Color Image (TCI) file.
    It gives more context and information on the selected zone.

    Args:
        image_folder (path): path to the image folder

    Returns:
        path: path of the tci file
    """
    subfolder = [f for f in os.listdir(image_folder + "GRANULE")
                 if f[0] == "L"][0]
    image_folder_path = f"{image_folder}GRANULE/{subfolder}/IMG_DATA/R10m"
    image_files = [im for im in os.listdir(image_folder_path)
                   if im[-4:] == ".jp2"]
    selected_file = [im for im in image_files
                     if im.split("_")[2] == "TCI"][0]
    return f"{image_folder_path}/{selected_file}"


def create_sample_coordinates(image, seed=None, p=0.01):
    """Creates a sample of coordinates from the image.

    The number of points equals approximately `n * m * p` where `n`
    is the number of rows, `m` is the number of columns, and `p` is
    the percentage of points to be selected.

    Args:
        image (image): image from which the coordinates are sampled
        seed (int, optional): seed for the random number generator.
        Default is `None`
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


def get_coordinates_from_pixels(img, h, v, img_folder, fire_name):
    """Retrieves the latitude and longitude coordinates from the pixels,
    then saves the coordinates in a CSV file.

    Args:
        img (image): image from which the coordinates are retrieved
        h, v (int): horizontal and vertical offsets
        img_folder (str): path to the folder where
        the JP2 file is stored
        fire_name (str): name of the fire

    Returns:
        coordinates: dataframe of coordinate tuples (latitude, longitude)
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

    coordinates = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    if not os.path.exists('data/coordinates_files/'):
        os.makedirs('data/coordinates_files/')

    coordinates.to_csv(f'data/coordinates_files/{fire_name}.csv', index=False)
    return coordinates


def get_choice():
    """Get the user's choice for the land cover data.
    Returns the choice as an integer in [1, 2, 3, 4]."""
    while True:
        try:
            print(
                """Please select the land cover data you want to use.
                The choices are:
                1. MODIS Land Cover (2018, 500m)
                2. ESA World Cover (2020, 10m)
                3. Copernicus Global Land Service (2019, 100m)
                4. Copernicus CORINE Land Cover (2018, 100m)"""
            )
            choice = int(input('Select land cover data: '))
            if choice not in {1, 2, 3, 4}:
                raise ValueError
            return choice
        except ValueError:
            print('Invalid input. Please try again.')


def select_land_cover_data(choice):
    """Select the land cover data to be used."""
    if choice == 1:
        return ee.ImageCollection('MODIS/006/MCD12Q1')
    elif choice == 2:
        return ee.ImageCollection("ESA/WorldCover/v100")
    elif choice == 3:
        return ee.Image(
            "COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select(
                'discrete_classification')
    elif choice == 4:
        return ee.Image("COPERNICUS/CORINE/V20/100m/2018")


def get_lc_urban_point(lc, choice, u_poi, scale):
    """Get the land cover type for the given point.

    Args:
        lc (image): land cover image
        choice (int): choice of the land cover data
        u_poi (list): point coordinates
        scale (int): scale of the land cover data
    """
    # MODIS Land Cover
    if choice == 1:
        return lc.first().sample(u_poi, scale).first().get(
            'LC_Type1').getInfo()
    # ESA World Cover
    elif choice == 2:
        return lc.first().sample(u_poi, scale).first().getInfo()[
            'properties']['Map']
    # Copernicus Global Land Service
    elif choice == 3:
        return lc.sample(u_poi, scale).first().getInfo()[
            'properties']['discrete_classification']
    # Copernicus CORINE Land Cover
    elif choice == 4:
        return lc.sample(u_poi, scale).first().getInfo()[
            'properties']['landcover']


def get_land_cover_dataframe(choice):
    """Get the land cover dataframe with
    the land cover value, description, and color.
    """
    if choice == 1:
        return pd.read_csv(
            'data/MODIS_LandCover_Type1.csv')
    elif choice == 2:
        return pd.read_csv(
            'data/ESA_WorldCover_10m_v100.csv')
    elif choice == 3:
        return pd.read_csv(
            'data/Copernicus_Landcover_100m_Proba-V-C3_Global.csv')
    elif choice == 4:
        return pd.read_csv(
            'data/Copernicus_CORINE_Land_Cover.csv')


def get_land_cover_data(coords_data, choice, size, seed=None):
    """Retrieves the land cover data from the coordinates.

    The keys in the dictionary are the labels and the values are the
    number of points in each land cover type.

    Args:
        coords_data (list): list of coordinate tuples (latitude, longitude)
        size (int): number of points to be sampled from the fire
        seed (int): seed for the random number generator.
        Default is `None`

    Returns:
        cover_data (dict): dictionary with the land cover data. The keys
        are the labels and the values are the number of points in each
        land cover type.
    """
    lc = select_land_cover_data(choice)
    scale = 1000

    np.random.seed(seed)
    cover_data = []
    random_idxs = np.random.choice(
        range(len(coords_data)), size=size, replace=False)

    for i in random_idxs:
        u_lat = coords_data.iloc[i]['latitude']
        u_lon = coords_data.iloc[i]['longitude']
        u_poi = ee.Geometry.Point(u_lon, u_lat)

        try:
            # access the earth engine API
            lc_urban_point = get_lc_urban_point(lc, choice, u_poi, scale)
            cover_data.append(lc_urban_point)
        except:
            print('Error with Earth Engine. Land cover could not be found.')

    if None in cover_data:
        cover_data = [i for i in cover_data if i]  # remove None values

    cover_data = dict(Counter(cover_data))
    # sort by keys so that the output plots have the same structure
    cover_data = dict(sorted(cover_data.items(), key=lambda x: x[0]))
    return cover_data


def get_labels_colors(cover_data, land_cover_data):
    """Retrieves the labels and colors for the pie chart.

    The land cover data is a dataframe with columns `Value`,
    `Color`, and `Description`.

    Args:
        cover_data (dict): dictionary with the land cover data
        land_cover_data (dataframe): dataframe with the description and
        colors by type of land coverage

    Returns:
        labels: labels for the pie chart
        colors: colors for the pie chart
    """
    labels = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Description'].values[0]
        for i in cover_data
    ]

    colors = [
        land_cover_data.loc[
            land_cover_data['Value'] == i]['Color'].values[0]
        for i in cover_data
    ]
    colors = [mplcols.to_rgb(i) for i in colors]
    return labels, colors


def plot_pie_chart(cover_data, labels, colors, points, fire_name,
                   out_folder=None, save_fig=True):
    """Plots a pie chart with the given data and labels.

    Args:
        cover_data (dict): data to be plotted
        labels (list): labels to be plotted
        colors (list): colors to be used for the plot
        points (int): number of points that were sampled
        fire_name (str): name of the fire
        out_folder (str): path to the output folder
        save_fig (bool): whether to save the figure or not. Default is `True`
    """
    percent = [100 * cover_data[i] / sum(cover_data.values())
               for i in cover_data]
    labels = ['{} ({:.2f}%)'.format(i, j) for i, j in zip(labels, percent)]
    _, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.set_aspect('equal')
    wedges, _, autotexts = ax.pie(
        cover_data.values(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops=dict(color='black'),
        pctdistance=1.1
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
    plt.setp(autotexts, size=6)
    title = f'{fire_name.title()} affected land cover types\n' + \
        f'Number of points: {points}'
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        if out_folder is None:
            out_folder = f'output/{fire_name}/pie_charts/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        plt.savefig(f'{out_folder}pie_{points}.png')


def create_plots(samples, coordinates, choice, seed=None, **kwargs):
    """Creates the plots for the given samples.

    Args:
        samples (list): list of samples to be drawn
        coordinates (dataframe): dataframe of coordinates
        choice (int): choice of the land cover data
        seed (int): seed for the random number generator.
        Default is `None`
        **kwargs: keyword arguments for the `plot_pie_chart` function
    """
    land_cover_dataframe = get_land_cover_dataframe(choice)
    for size in samples:
        print(f'Retrieving {size} samples...')
        cover_data = get_land_cover_data(
            coordinates, choice, size, seed)
        print(cover_data)
        labels, colors = get_labels_colors(cover_data, land_cover_dataframe)
        plot_pie_chart(
            cover_data=cover_data,
            labels=labels,
            colors=colors,
            points=size,
            fire_name=kwargs['fire_name'],
            out_folder=kwargs['out_folder'],
            save_fig=kwargs['save_fig']
        )
    # plt.show()


def make_pie_chart_gif(fire_name, file_path=None, **kwargs):
    """Makes a gif from the PNG images in the file_path.

    The filenames must be in format 'pie_*.png'
    where '*' is the number of points sampled from the fire using the
    `create_plots` function.

    This assumes that the PNG files are saved using the `save_fig` parameter
    of the `plot_pie_chart` function.

    Args:
        fire_name (str): name of the fire
        file_path (str): path to the folder where the PNG files are saved
    """
    if file_path is None:
        file_path = f'output/{fire_name}/pie_charts/'

    files = glob.glob(file_path + 'pie_*.png')
    files = [f.split('_')[-1].split('.')[0] for f in files]
    files = sorted(files, key=int)
    out_file = file_path + fire_name + '.gif'

    img, *imgs = [Image.open(f'{file_path}pie_{f}.png')
                  for f in files]
    img.save(fp=out_file, format='GIF', append_images=imgs, **kwargs)


def open_gif(fire_name, output_folder):
    """Opens the gif created with the `make_pie_chart_gif` function.

    Args:
        fire_name (str): name of the fire
        output_folder (str): path to the output folder
    """
    file_path = output_folder + fire_name + '.gif'
    webbrowser.open_new_tab('file://' + os.path.realpath(file_path))
