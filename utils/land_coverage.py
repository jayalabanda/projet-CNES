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

def create_sample_coordinates(image, seed, p=0.01):
    """Creates a sample of coordinates from the image.
       The number of points equals n * m * p where n 
       is the number of rows and m is the number of columns.

    Args:
        image (image): image from which the coordinates are sampled
        seed (int): seed for the random number generator
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
   """Retreives the coordinates from the pixels.

    Args:
        img (image): image from which the coordinates are retrieved
        h,v (int): horizontal and vertical offsets
        img_folder (string): path to the folder where the image is retrieved from 

    Returns:
        coords_data: list of coordinates
    """
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
    """Retrieves the labels and colors for the pie chart.

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