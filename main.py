import datetime as dt
import json
import os

import ee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ipyleaflet import basemap_to_tiles, basemaps
from sentinelsat import SentinelAPI
from skimage.morphology import area_closing

import utils.data_collection as dc
import utils.image_processing as ip
import utils.land_coverage as land_c
import utils.plot_map as pm
import utils.wind_data as wind


def clear_screen():
    """Clears the terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


###############################################################################
# SETUP
###############################################################################

print("Welcome to the Sentinel data collection and analysis tool!")
input("Please press enter to continue.")
clear_screen()
print("1. Setup")

# If you have never authenticated, this will raise an exception
# and prompt you to authenticate. You only need to do this once.
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
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

input("Press enter to proceed.")
clear_screen()


###############################################################################
# WILDFIRE INFORMATION
###############################################################################

print("2. Wildfire Information")

with open(f"data/info_fires/info_{FIRE_NAME}.json") as f:
    fire_info = json.load(f)

print("Retrieving date, coordinates and burnt area from JSON file.", end='')

# Date of the fire
WILDFIRE_DATE = dt.datetime.strptime(fire_info["wildfire_date"], "%Y-%m-%d")
# Coordinates of the fire
LATITUDE, LONGITUDE = fire_info["latitude"], fire_info["longitude"]
# Actual area in hectares that burned. We retrieved the area on the news
TRUE_AREA = fire_info["true_area"]

print("Done!")
print("Accessing GeoJSON file and API credentials...", end='')

# Path to the GeoJSON file
GEOJSON_PATH = f"data/geojson_files/{FIRE_NAME}.geojson"
# Path to the JSON file where the Sentinel API credentials are stored
CREDENTIALS_PATH = "secrets/sentinel_api_credentials.json"

print("Done!")
input("Press enter to proceed.")
clear_screen()


###############################################################################
# FILE CONSTANTS
###############################################################################

print("3. File Constants")

# Number of days both before and after the fire to get images
OBSERVATION_INTERVAL = 15
print(f"Number of days both before and after the fire: {OBSERVATION_INTERVAL}")

# Resolution of the images (10m, 20m, or 60m)
RESOLUTION = 10
print(f"Resolution of the images: {RESOLUTION} m")
# Threshold for the cloud cover (between 0 and 100)
CLOUD_THRESHOLD = 40
print(f"Threshold for the cloud cover: {CLOUD_THRESHOLD}%")

# Seed for random number generator (for reproductibility)
SEED = 42
# Threshold values for calculating the area that burned
THRESHOLDS = np.arange(0.1, 0.41, 0.02)
# Number of coordinates to use for the pie charts
SAMPLES = np.arange(50, 1001, 50)

print("If you are not satisfied with the values, you can change them here.")
input("Press enter to continue.")
clear_screen()


###############################################################################
# DATA COLLECTION
###############################################################################

with open(CREDENTIALS_PATH, 'r') as infile:
    credentials = json.load(infile)

api = SentinelAPI(
    credentials["username"],
    credentials["password"]
)

if not os.path.exists(PATH):
    os.makedirs(PATH)

if not dc.check_downloaded_data(PATH, OUTPUT_FOLDER, FIRE_NAME):
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


###############################################################################
# IMAGE PROCESSING
###############################################################################

ip.plot_downloaded_images(FIRE_NAME, OUTPUT_FOLDER)

img_folder = PATH + os.listdir(PATH)[1] + '/'
print(img_folder)

pixel_column, pixel_row = ip.get_fire_pixels(
    img_folder, LATITUDE, LONGITUDE
)
print(f'The fire is located at pixels ({pixel_column}, {pixel_row}).\n')

diff = dc.plot_ndvi_difference(
    OUTPUT_FOLDER, FIRE_NAME, save_diff=False, figsize=(10, 10)
)
plt.plot(pixel_column, pixel_row, 'ro',
         markersize=3, label='Fire Location')
plt.legend()
plt.show()

fire, hline_1, vline_1 = ip.retrieve_fire_area(
    diff, pixel_column, pixel_row, 'Fire Area'
)


###############################################################################
# WILDFIRE AREA
###############################################################################

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


tmp = ip.threshold_filter(fire, threshold)
plt.figure(figsize=(8, 6))
ip.imshow(tmp, 'Thresholded Fire')
plt.show()

print('Calculated area:', round(ip.calculate_area(tmp, diff) * 100, 4), 'ha.')
print(f'The true area that burned is {TRUE_AREA} hectares.\n')


###############################################################################
# MORPHOLOGY
###############################################################################

closed = area_closing(tmp)
ip.plot_comparison(tmp, closed, 'Area Closing')

print('Area after morphology:',
      round(ip.calculate_area(closed, diff) * 100, 4), 'ha.')

fire = closed.copy()
del tmp, closed


###############################################################################
# LAND COVER CLASSIFICATION
###############################################################################

choice = land_c.get_choice()
lc_dataframe = land_c.get_land_cover_dataframe(choice)

while True:
    try:
        prob = input(
            r'Enter sample percentage to use for land cover classification: '
        )
        prob = float(prob) / 100
        if 0 < prob <= 1:
            break
        else:
            print('Please enter a value between 0 and 100.')
    except ValueError:
        print('Please enter a valid number.')

rand_image = land_c.create_sample_coordinates(fire, SEED, prob)
ip.imshow(rand_image, 'Sampled Coordinates', cmap='hot')
plt.show()

coordinates = land_c.get_coordinates_from_pixels(
    rand_image, hline_1, vline_1, img_folder, FIRE_NAME
)

output_folder = f"output/pie_chart_{FIRE_NAME}/"
exists = os.path.exists(output_folder)
if not exists:
    os.makedirs(output_folder)
is_empty = not any(os.scandir(output_folder))

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

land_c.open_gif(FIRE_NAME, output_folder)

# import base64
# from IPython import display

# def show_gif(fname):
#     with open(fname, 'rb') as fd:
#         b64 = base64.b64encode(fd.read()).decode('ascii')
#     return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')

# show_gif('output/pie_chart_' + FIRE_NAME + '/' + FIRE_NAME + '.gif')


###############################################################################
# CREATE GEEMAP MAP
###############################################################################

while True:
    try:
        print("\nEnter the percentage of points to add to the map: ")
        prob = float(input()) / 100
        if 0 < prob <= 1:
            break
        else:
            raise ValueError
    except ValueError:
        print('Please enter a valid number.')

fire_map = pm.create_map(FIRE_NAME, prob, SEED, choice)

output_maps = "output/maps/"
if not os.path.exists(output_maps):
    os.makedirs(output_maps)

pm.save_map(fire_map, FIRE_NAME, output_maps)
pm.open_map(FIRE_NAME, output_maps)


###############################################################################
# WIND DATA
###############################################################################

year = WILDFIRE_DATE.strftime('%Y')
month = WILDFIRE_DATE.strftime('%m')
day = WILDFIRE_DATE.strftime('%d')
hours = ['12:00']
center = (LATITUDE, LONGITUDE)

output_file = wind.retrieve_wind_data(FIRE_NAME, year, month, day, hours)

ds = wind.open_nc_data(output_file)
ds = wind.reshape_data(ds)

m = wind.create_map(
    ds, center, zoom=5,
    # basemap=basemaps.CartoDB.DarkMatter,
    basemap=basemaps.Esri.WorldImagery,
    add_zoom_slider=True,
    add_layers_control=True,
    add_full_screen=True
)

m.add_layer(basemap_to_tiles(basemaps.CartoDB.DarkMatter))

wind.save_map(m, FIRE_NAME)
pm.open_map(FIRE_NAME, output_maps)
