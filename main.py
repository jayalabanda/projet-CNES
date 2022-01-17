import argparse
import datetime as dt
import json
import os
import sys

import ee
import matplotlib.pyplot as plt
import numpy as np
from ipyleaflet import basemap_to_tiles, basemaps
from sentinelsat import SentinelAPI
from skimage.morphology import area_closing

import utils.image_processing as ip
import utils.land_coverage as land_c
import utils.wind_data as wind
from utils.data_collection import (check_downloaded_data,
                                   get_before_after_images)
from utils.plot_map import create_map, open_map, save_map
from utils.user_inputs import get_fire_name, get_percentage

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fire_name',
    help='Name of the fire to study',
    type=str)
parser.add_argument(
    '--fire_percentage',
    help='Percentage of the fire to study, between 0.0 and 1.0',
    type=float)
parser.add_argument(
    '--land_percentage',
    help='Percentage of the land to study, between 0.0 and 1.0',
    type=float)
args = parser.parse_args()


def clear_screen():
    """Clears the terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


###############################################################################
# SETUP
###############################################################################

print('Welcome to the Sentinel data collection and analysis tool.')
input('Please press enter to continue.')
clear_screen()
print('1. Setup')

# If you have never authenticated, this will raise an exception
# and prompt you to authenticate. You only need to do this once.
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Name of the place where the fire is located
FIRE_NAME = args.fire_name if len(sys.argv) > 1 else get_fire_name()

# Folder where the JP2 images will be stored
PATH = f'data/{FIRE_NAME}/'
# Path to the folders where the TIFF, PNG, HTML, and GIF files will be stored
OUTPUT_FOLDER = f'output/{FIRE_NAME}/'
OUTPUT_MAPS = OUTPUT_FOLDER + 'maps/'
OUTPUT_PLOTS = OUTPUT_FOLDER + 'plots/'

for path in [PATH, OUTPUT_PLOTS, OUTPUT_FOLDER, OUTPUT_MAPS]:
    if not os.path.exists(path):
        os.makedirs(path)

input('Press enter to proceed.')
clear_screen()

###############################################################################
# WILDFIRE INFORMATION
###############################################################################

print('2. Wildfire Information')

with open(f'data/info_fires/info_{FIRE_NAME}.json') as f:
    fire_info = json.load(f)

print('Retrieving date, coordinates, and burnt area from JSON file.')

# Date of the fire
WILDFIRE_DATE = dt.datetime.strptime(fire_info['wildfire_date'], '%Y-%m-%d')
# Coordinates of the fire
LATITUDE, LONGITUDE = fire_info['latitude'], fire_info['longitude']
# Actual area in hectares that burned. We retrieved the area on the news
TRUE_AREA = fire_info['true_area']

print('Done.')
print('Accessing GeoJSON file and API credentials.')

# Path to the GeoJSON file
GEOJSON_PATH = f'data/geojson_files/{FIRE_NAME}.geojson'
# Path to the JSON file where the Sentinel API credentials are stored
CREDENTIALS_PATH = 'secrets/sentinel_api_credentials.json'

print('Done.')
input('Press enter to proceed.')
clear_screen()

###############################################################################
# FILE CONSTANTS
###############################################################################

print('3. File Constants')

# Number of days both before and after the fire to get images
OBSERVATION_INTERVAL = 15
print(
    f'Number of days both before and after the fire: {OBSERVATION_INTERVAL}.')

# Resolution of the images (10, 20, or 60 m)
RESOLUTION = 10
print(f'Resolution of images: {RESOLUTION} m.')

# Threshold for the cloud cover (between 0 and 100)
CLOUD_THRESHOLD = 40
print(f'Threshold for cloud cover: {CLOUD_THRESHOLD} %.')

# Seed for random number generator (for reproductibility)
SEED = 42

# Number of coordinates to use for the pie charts
SAMPLES = np.arange(50, 1001, 50)

input('Press enter to continue.')
clear_screen()

###############################################################################
# DATA COLLECTION
###############################################################################

print('4. Data Collection')

with open(CREDENTIALS_PATH, 'r') as infile:
    credentials = json.load(infile)

api = SentinelAPI(
    credentials['username'],
    credentials['password']
)

if not check_downloaded_data(PATH, OUTPUT_FOLDER, FIRE_NAME):
    try:
        get_before_after_images(
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

input('Press enter to continue.')
clear_screen()

###############################################################################
# IMAGE PROCESSING
###############################################################################

print('5. Image Processing')

print('Plotting NDVI images for comparison.')
ip.plot_downloaded_images(FIRE_NAME, OUTPUT_FOLDER, save=True)

# The necessary information is stored in the following folder:
img_folder = PATH + os.listdir(PATH)[1] + '/'

pixel_column, pixel_row = ip.get_fire_pixels(
    img_folder, LATITUDE, LONGITUDE
)

print('Plotting NDVI difference.')
diff = ip.get_ndvi_difference(
    OUTPUT_FOLDER, FIRE_NAME, save_diff=False
)
_ = ip.imshow(diff, figsize=(10, 10), title='NDVI Difference')
plt.savefig(f'{OUTPUT_PLOTS}ndvi_difference.png', dpi=200)
plt.show()

print('Plotting NDVI difference with wildfire location.')
title = 'NDVI Difference with Wildfire Location'
ax = ip.imshow(diff, figsize=(10, 10), title=title)
ip.plot_location(ax, pixel_column, pixel_row)
plt.savefig(f'{OUTPUT_PLOTS}ndvi_difference_w_fire.png', dpi=200)
plt.show()

print(f'The fire is located at pixels ({pixel_column}, {pixel_row}).\n')

fire, hline_1, vline_1 = ip.retrieve_fire_area(
    diff, pixel_column, pixel_row,
    figsize=(10, 10), title='Fire Area'
)
plt.savefig(f'{OUTPUT_PLOTS}fire_area.png', dpi=200)
plt.show()

input('Press enter to continue.')
clear_screen()

###############################################################################
# WILDFIRE AREA
###############################################################################

print('6. Wildfire Area')

thresholds, areas = ip.get_thresholds_areas(fire, RESOLUTION)

print('Plotting the calculated area by threshold.')
ip.plot_area_vs_threshold(thresholds, areas, TRUE_AREA)
plt.savefig(f'{OUTPUT_PLOTS}fire_area_thresholds.png', dpi=200)
plt.show()

threshold = ip.get_threshold(thresholds, areas, TRUE_AREA)
print('Selected threshold:', threshold)

tmp = ip.threshold_filter(fire, threshold)
title = f'Thresholded Fire\nwith threshold = {threshold}'
ax = ip.imshow(tmp, figsize=(10, 10), title=title)
plt.savefig(f'{OUTPUT_PLOTS}thresholded_fire.png', dpi=200)
plt.show()

print('Calculated area:', round(ip.calculate_area(tmp) * 100, 4), 'ha.')
print(f'The true area that burned is {TRUE_AREA} hectares.\n')

input('Press enter to continue.')
clear_screen()

###############################################################################
# MORPHOLOGY
###############################################################################

print('7. Morphology')

print('Plotting original vs. filtered image.')
closed = area_closing(tmp, connectivity=2)
ip.plot_comparison(tmp, closed, 'Area Closing')
plt.savefig(f'{OUTPUT_PLOTS}area_closing.png', dpi=200)
plt.show()

print('Area after morphology:',
      round(ip.calculate_area(closed) * 100, 4), 'ha.')

fire = closed.copy()
del tmp, closed

input('Press enter to continue.')
clear_screen()

###############################################################################
# LAND COVER CLASSIFICATION
###############################################################################

print('8. Land Cover Classification')

choice = land_c.get_choice()
lc_dataframe = land_c.get_land_cover_dataframe(choice)

prob = args.fire_percentage if len(
    sys.argv) > 1 else get_percentage(case='land use')

rand_image = land_c.create_sample_coordinates(fire, SEED, prob)
land_c.plot_sampled_coordinates(
    rand_image, prob, figsize=(8, 6), cmap='hot'
)
plt.savefig(f'{OUTPUT_PLOTS}sampled_coordinates.png', dpi=200)
plt.show()

coordinates = land_c.get_coordinates_from_pixels(
    rand_image, hline_1, vline_1, img_folder, FIRE_NAME
)

output_folder = f'output/{FIRE_NAME}/pie_charts/'
exists = os.path.exists(output_folder)
if not exists:
    os.makedirs(output_folder)
is_empty = not any(os.scandir(output_folder))

if exists and is_empty or not exists:
    print('Creating pie charts.')
    land_c.create_plots(
        samples=SAMPLES,
        coordinates=coordinates,
        choice=choice,
        seed=SEED,
        fire_name=FIRE_NAME,
        out_folder=output_folder,
        save_fig=True
    )

print('Done.\nPlotting pie charts.')
land_c.make_pie_chart_gif(
    fire_name=FIRE_NAME,
    file_path=output_folder,
    save_all=True,
    duration=500,
    loop=0
)

land_c.open_gif(FIRE_NAME, output_folder)

input('Done.\nPress enter to continue.')
clear_screen()

###############################################################################
# CREATE GEEMAP MAP
###############################################################################

print('9. Coordinates Map')

prob = args.land_percentage if len(
    sys.argv) > 1 else get_percentage(case='map')

fire_map = create_map(
    FIRE_NAME, prob, choice,
    seed=SEED,
    zoom=5,
    cluster=True,
    minimap=False
)

save_map(fire_map, FIRE_NAME, OUTPUT_MAPS, wind=False)
open_map(OUTPUT_MAPS, wind=False)

###############################################################################
# WIND DATA
###############################################################################

print('10. Wind Map')

year = WILDFIRE_DATE.strftime('%Y')
month = WILDFIRE_DATE.strftime('%m')
day = WILDFIRE_DATE.strftime('%d')
hours = ['12:00']
center = (LATITUDE, LONGITUDE)

output_file = wind.retrieve_wind_data(FIRE_NAME, year, month, day, hours)
print('Output file:', output_file)

ds = wind.open_nc_data(output_file)
print('Wind data:\n', ds)
ds = wind.reshape_data(ds)

wind_map = wind.create_map(
    ds, center, choice,
    zoom=5,
    # basemap=basemaps.CartoDB.DarkMatter,
    basemap=basemaps.Esri.WorldImagery
)

wind_map.add_layer(basemap_to_tiles(basemaps.CartoDB.DarkMatter))

save_map(wind_map, FIRE_NAME, OUTPUT_MAPS, wind=True)
open_map(OUTPUT_MAPS, wind=True)

input('Press enter to continue.')
clear_screen()

print('Reached the end of the wildfire monitoring program.')
