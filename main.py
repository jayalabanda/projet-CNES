import datetime as dt
import json
import os
import webbrowser

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sentinelsat import SentinelAPI
from skimage.morphology import area_closing

import utils.data_collection as dc
import utils.image_processing as ip
import utils.land_coverage as land_c
import utils.plot_folium as pf

###############################################################################
# SETUP
###############################################################################

while True:
    try:
        authenticated = input("Authenticate with Earth Engine? (y/n): ")
        if authenticated == "y":
            ee.Authenticate()
            break
        elif authenticated == "n":
            break
        else:
            raise ValueError("Invalid input")
    except ValueError:
        print("Please enter 'y' or 'n'.")
ee.Initialize()

# Name of the place where the fire is located
try:
    FIRE_NAME = input("Name of the fire: ").lower()
except ValueError:
    print("Please enter a valid name.")

# Folder where the JP2 images will be stored
PATH = f'data/{FIRE_NAME}/'
# Path to the folders where the TIFF, PNG, and GIF files will be stored
OUTPUT_FOLDER = f"output/{FIRE_NAME}/"

with open(f"data/info_{FIRE_NAME}.json") as f:
    fire_info = json.load(f)

# Date of the fire
WILDFIRE_DATE = dt.datetime.strptime(fire_info["wildfire_date"], "%Y-%m-%d")
# Coordinates of the fire
LATITUDE, LONGITUDE = fire_info["latitude"], fire_info["longitude"]
# Actual area in hectares that burned. We retrieved the area on the news
TRUE_AREA = fire_info["true_area"]

# Path to the GeoJSON file
GEOJSON_PATH = f"data/geojson_files/{FIRE_NAME}.geojson"
# Path to the JSON file where the Sentinel API credentials are stored
CREDENTIALS_PATH = "secrets/sentinel_api_credentials.json"

# Number of days both before and after the fire to get images
OBSERVATION_INTERVAL = 15
# Resolution of the images (10m, 20m, or 60m)
RESOLUTION = 10
# Threshold for the cloud cover (between 0 and 100)
CLOUD_THRESHOLD = 40
# Split the image into smaller chunks
# FRAG_COUNT = 15
# Seed for random number generator (for reproductibility)
SEED = 42

# Threshold values for calculating the area that burned
THRESHOLDS = np.arange(0.1, 0.41, 0.02)
# Number of coordinates to use for the pie charts
SAMPLES = np.arange(50, 251, 50)


###############################################################################
# DATA COLLECTION
###############################################################################

with open(CREDENTIALS_PATH, 'r') as infile:
    credentials = json.load(infile)

api = SentinelAPI(
    credentials["username"],
    credentials["password"]
)

exists = os.path.exists(PATH)
if not exists:
    os.makedirs(PATH)

try:
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
except Exception as e:
    print(e)
    exit()

ip.plot_downloaded_images(FIRE_NAME, OUTPUT_FOLDER)

img_folder = PATH + os.listdir(PATH)[1] + '/'
pixel_column, pixel_row = ip.get_fire_pixels(
    img_folder,
    LATITUDE, LONGITUDE
)

diff = dc.plot_ndvi_difference(OUTPUT_FOLDER, FIRE_NAME, figsize=(10, 10))
plt.plot(pixel_column, pixel_row, 'ro',
         markersize=3, label='Fire Location')
plt.legend()
plt.show()


###############################################################################
# IMAGE PROCESSING
###############################################################################


print(f'The fire is located at pixels ({pixel_column}, {pixel_row})\n')
fire, hline_1, vline_1 = ip.retrieve_fire_area(
    diff, pixel_column, pixel_row, 'Fire Area'
)

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
    plt.title('Fire Area')
    plt.legend(['Calculated Area', 'True Value'])
    plt.show()

while True:
    threshold = float(input('Enter a threshold value between -1 and 1: '))
    try:
        if -1 <= threshold <= 1:
            break
        else:
            print('Please enter a value between -1 and 1.')
    except ValueError:
        print('Please enter a valid number.')

fire = ip.threshold_filter(fire, threshold)
plt.figure(figsize=(8, 6))
ip.imshow(tmp, 'Thresholded Fire')
plt.show()

print(
    'Area:',
    round(ip.calculate_area(fire, diff) * 100, 4), 'ha'
)

closed = area_closing(fire)
ip.plot_comparison(fire, closed, 'Area Closing')

print(
    'Area after morphology:',
    round(ip.calculate_area(closed, diff) * 100, 4), 'ha'
)


###############################################################################
# CLASSIFICATION
###############################################################################

while True:
    try:
        p = input(
            r'Enter sample percentage to use for land cover classification: ')
        p = float(p) / 100
        if 0 < p <= 1:
            break
        else:
            print('Please enter a value between 0 and 100.')
    except ValueError:
        print('Please enter a valid number.')

rand_image = land_c.create_sample_coordinates(fire, 42, p)
ip.imshow(rand_image, 'Sampled Coordinates')
plt.show()

coordinates = land_c.get_coordinates_from_pixels(
    rand_image, hline_1, vline_1, img_folder, FIRE_NAME
)

output_folder = f"output/pie_chart_{FIRE_NAME}/"
exists = os.path.exists(output_folder)
if not exists:
    os.makedirs(output_folder)
is_empty = not any(os.scandir(output_folder))

choice = land_c.get_choice()

if exists and is_empty or not exists:
    land_c.create_plots(
        samples=SAMPLES,
        coordinates=coordinates,
        choice=choice,
        seed=SEED,
        fire_name=FIRE_NAME,
        out_folder=output_folder,
        save_fig=True
    )

land_c.make_pie_chart_gif(
    fire_name=FIRE_NAME,
    file_path=output_folder,
    save_all=True,
    duration=500,
    loop=0
)

# Create Folium map
while True:
    try:
        prob = float(input('Value between 0 and 100: ')) / 100
        if 0 < p <= 1:
            break
        else:
            raise ValueError
    except ValueError:
        print('Please enter a valid number.')

fire_map = pf.create_map(FIRE_NAME, prob, SEED, choice)

output_maps = "output/maps/"
if not os.path.exists(output_maps):
    os.makedirs(output_maps)

pf.save_map(fire_map, FIRE_NAME, output_maps)
pf.open_map(FIRE_NAME, output_maps)

# Finally, open the produced gif
# Note that it opens the file with your default program for opening images,
# not necessarily your default web browser.
land_c.open_gif(FIRE_NAME, output_folder)

###############################################################################
# WIND DATA
###############################################################################
