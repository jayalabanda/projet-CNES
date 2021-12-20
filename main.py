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
import utils.plot_folium as pf


authenticated = input("Authenticate with Earth Engine? (y/n): ")
if authenticated == "y":
    ee.Authenticate()
elif authenticated != "n":
    print("Please enter 'y' or 'n'.")
    raise ValueError("Please enter 'y' or 'n'.")
ee.Initialize()

# Name of the place where the fire is located
FIRE_NAME = input("Name of the fire: ").lower()

# Path to the GeoJSON file
GEOJSON_PATH = f"data/geojson_files/{FIRE_NAME}.geojson"
# Path to the JSON file where the Sentinel API credentials are stored
CREDENTIALS_PATH = "secrets/sentinel_api_credentials.json"
# Path to the CSV land cover data
LAND_COVER_DATA = pd.read_csv('data/MODIS_LandCover_Type1.csv')

# Date of the fire
WILDFIRE_DATE = dt.date(2020, 7, 28)
# Number of days both before and after the fire to get images
OBSERVATION_INTERVAL = 15
# Folder where the JP2 images will be stored
PATH = f'data/{FIRE_NAME}/'
# Path to the folder where the TIFF, PNG, and GIF images will be stored
OUTPUT = 'output/'
OUTPUT_FOLDER = f"{OUTPUT}{FIRE_NAME}/"

# Resolution of the images (10m, 20m, or 60m)
RESOLUTION = 10
# Threshold for the cloud cover (between 0 and 100)
CLOUD_THRESHOLD = 40
# Split the image into smaller chunks
# FRAG_COUNT = 15
# Coordinates of the fire
LATITUDE, LONGITUDE = 44.456801, -0.571638
# Actual area in hectares that burned. We retrieved the area on the news
TRUE_AREA = 250.0
# Seed for random number generator (for reproductibility)
SEED = 42

# Number of coordinates to use for the pie charts
THRESHOLDS = np.arange(0.1, 0.41, 0.02)


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
    print("Please try again later.")
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

# split = ip.split_image(diff, FRAG_COUNT)
# ip.plot_split_image(split, FRAG_COUNT)
# plt.tight_layout()
# plt.show()

# fire = ip.merge_images(
#     2, [split[(8, 11)], split[(8, 12)]], horizontal=False)
# plt.figure(figsize=(8, 6))
# ip.imshow(fire, 'Istres Fire')
# plt.show()

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

dilated = area_closing(fire)
ip.plot_comparison(fire, dilated, 'Area Closing')

print(
    'Area after morphology:',
    round(ip.calculate_area(dilated, diff) * 100, 4), 'ha'
)

while True:
    try:
        p = float(input(
            r'Enter sample percentage between 0% and 100%: ')) / 100
        if 0 < p <= 1:
            break
        else:
            print('Please enter a value between 0 and 100.')
    except ValueError:
        print('Please enter a valid number.')


###############################################################################
# CLASSIFICATION
###############################################################################

rand_image = lc.create_sample_coordinates(fire, 42, p)
ip.imshow(rand_image, 'Sampled Coordinates')
plt.show()

coordinates = lc.get_coordinates_from_pixels(
    rand_image, hline_1, vline_1, img_folder)
coordinates = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
coordinates.to_csv(
    f'data/coordinates_files/coords_utm_{FIRE_NAME}.csv',
    index=False
)

output_folder = f"{OUTPUT}pie_chart_{FIRE_NAME}/"
exists = os.path.exists(output_folder)
if not exists:
    os.makedirs(output_folder)
is_empty = not any(os.scandir(output_folder))

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
        file_path=output_folder,
        save_all=True,
        duration=500,
        loop=0
    )

# Create Folium map
p = float(input('Value between 0 and 100: ')) / 100
fire_map = pf.create_map(FIRE_NAME, p, SEED)

pf.save_map(fire_map, FIRE_NAME, 'output/maps/')
pf.open_map(FIRE_NAME, 'output/maps/')

###############################################################################
# WIND DATA
###############################################################################
