import datetime as dt
import json
import os

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentinelsat import SentinelAPI
from skimage.morphology import area_closing

import utils.data_collection as dc
import utils.image_processing as ip
import utils.land_coverage as lc


while True:
    authenticated = input(
        "Authenticate with Earth Engine? (y/n): ").lower().strip()
    if authenticated == "y":
        ee.Authenticate()
        break
    elif authenticated != "n":
        print("Please enter 'y' or 'n'.")
    else:
        break
ee.Initialize()

# Path to the GeoJSON file
GEOJSON_PATH = "data/geojson_files/istres.geojson"
# Path to the JSON file where the Sentinel API credentials are stored
CREDENTIALS_PATH = "secrets/sentinel_api_credentials.json"
# Path to the CSV land cover data
LAND_COVER_DATA = pd.read_csv('data/MODIS_LandCover_Type1.csv')

# Name of the place where the fire is located
FIRE_NAME = 'istres'
# Date of the fire
WILDFIRE_DATE = dt.date(2020, 8, 24)
# Number of days both before and after the fire to get images
OBSERVATION_INTERVAL = 15
# Folder where the JP2 images will be stored
PATH = 'data/' + FIRE_NAME + '/'
# Path to the folder where the TIFF, PNG, and GIF images will be stored
OUTPUT = 'output/'
OUTPUT_FOLDER = OUTPUT + FIRE_NAME + '/'

# Resolution of the images (10m, 20m, or 60m)
RESOLUTION = 10
# Threshold for the cloud cover (between 0 and 100)
CLOUD_THRESHOLD = 40
# Split the image into smaller chunks
# FRAG_COUNT = 15
# Coordinates of the fire
LATITUDE, LONGITUDE = 43.453228, 4.980225
# Actual area in hectares that burned. We retrieved the area on the news
TRUE_AREA = 319.6900
# Seed for random number generator (for reproducibility)
SEED = 42


def main():
    with open(CREDENTIALS_PATH, 'r') as infile:
        credentials = json.load(infile)

    api = SentinelAPI(
        credentials["username"],
        credentials["password"]
    )

    is_empty = not any(os.scandir(PATH))
    exists = os.path.exists(PATH)

    if exists and is_empty or not exists:
        dc.get_before_after_images(
            api=api,
            wildfire_date=WILDFIRE_DATE,
            geojson_path=GEOJSON_PATH,
            observation_interval=OBSERVATION_INTERVAL,
            path=PATH,
            fire_name=FIRE_NAME,
            output_folder=OUTPUT_FOLDER,
            resolution=RESOLUTION,
            cloud_threshold=CLOUD_THRESHOLD
        )

    ip.plot_downloaded_images(FIRE_NAME, OUTPUT_FOLDER)

    img_folder = PATH + os.listdir(PATH)[1] + '/'
    pixel_column, pixel_row = ip.get_fire_pixels(
        img_folder,
        LATITUDE, LONGITUDE
    )

    diff = dc.plot_ndvi_difference(OUTPUT_FOLDER, FIRE_NAME, figsize=(10, 10))
    plt.plot(pixel_column, pixel_row, 'ro',
             markersize=3, label='Fire location')
    plt.legend()
    plt.show()

    # split = ip.split_image(diff, FRAG_COUNT)
    # ip.plot_split_image(split, FRAG_COUNT)
    # plt.tight_layout()
    # plt.show()

    # fire = ip.merge_images(
    #     2, [split[(8, 11)], split[(8, 12)]], horizontal=False)
    # plt.figure(figsize=(8, 6))
    # ip.imshow(fire, 'Istres Fire')
    # plt.show()

    VLINE_1 = 5800
    VLINE_2 = 6550
    DELTA_V = VLINE_2 - VLINE_1
    HLINE_1 = 8400
    HLINE_2 = HLINE_1 + DELTA_V

    plt.figure(figsize=(12, 12))
    ip.imshow(diff, 'NDVI Difference')
    plt.vlines(VLINE_1, ymin=0, ymax=diff.shape[0],
               color='r', linestyle='dashed', linewidth=1)
    plt.vlines(VLINE_2, ymin=0, ymax=diff.shape[0],
               color='r', linestyle='dashed', linewidth=1)
    plt.hlines(HLINE_1, xmin=0, xmax=diff.shape[1],
               color='r', linestyle='dashed', linewidth=1)
    plt.hlines(HLINE_2, xmin=0, xmax=diff.shape[1],
               color='r', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    plt.show()

    fire = diff[HLINE_1:HLINE_2, VLINE_1:VLINE_2]
    ip.imshow(fire, 'Istres Fire')
    plt.axis('off')
    plt.show()

    THRESHOLDS = np.arange(0.2, 0.41, 0.02)
    areas = []
    for thr in THRESHOLDS:
        tmp = ip.threshold_filter(fire, thr)
        area = round(ip.calculate_area(tmp, diff) * 100, 4)
        areas.append(area)

    plt.figure(figsize=(8, 6))
    with sns.axes_style('darkgrid'):
        plt.plot(THRESHOLDS, areas)
        plt.hlines(TRUE_AREA, THRESHOLDS[0], THRESHOLDS[-1],
                   colors='r', linestyles='dashed')
        plt.xlabel('Threshold')
        plt.ylabel('Burnt Area (ha)')
        plt.title('Istres Fire')
        plt.legend(['Calculated Area', 'True Value'])
        plt.show()

    while True:
        threshold = input('Enter a threshold value between -1 and 1: ')
        try:
            threshold = float(threshold)
            if threshold > -1 and threshold < 1:
                break
            else:
                print('Please enter a value between -1 and 1.')
        except ValueError:
            print('Please enter a valid number.')

    tmp = ip.threshold_filter(fire, threshold)
    plt.figure(figsize=(8, 6))
    ip.imshow(tmp, 'Istres Fire')
    plt.show()

    print(
        'Area:',
        round(ip.calculate_area(tmp, diff) * 100, 4), 'ha'
    )

    dilated = area_closing(tmp)
    ip.plot_comparison(tmp, dilated, 'Area Closing')

    print(
        'Area after morphology:',
        round(ip.calculate_area(dilated, diff) * 100, 4), 'ha'
    )

    fire = ip.threshold_filter(fire, 0.28)
    ip.imshow(fire, 'Thresholded Istres Fire')
    plt.show()

    rand_image = lc.create_sample_coordinates(fire, 42, 0.05)
    ip.imshow(rand_image, 'Sampled Coordinates')
    plt.show()

    coordinates = lc.get_coordinates_from_pixels(
        rand_image, HLINE_1, VLINE_1, img_folder)
    coordinates = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    coordinates.to_csv(
        'data/coordinates_files/coords_utm_istres.csv',
        index=False
    )

    output_folder = OUTPUT + 'pie_chart_' + FIRE_NAME + '/'
    is_empty = not any(os.scandir(output_folder))
    exists = os.path.exists(output_folder)

    if exists and is_empty or not exists:
        SAMPLES = np.arange(50, 1001, 50)
        lc.create_plots(
            samples=SAMPLES,
            coordinates=coordinates,
            seed=SEED,
            land_cover_data=LAND_COVER_DATA,
            fire_name=FIRE_NAME,
            out_folder=output_folder,
            save_fig=True
        )

        lc.make_pie_chart_gif(
            fire_name=FIRE_NAME,
            file_path=OUTPUT_FOLDER,
            save_all=True,
            duration=500,
            loop=0
        )


if __name__ == '__main__':
    main()
