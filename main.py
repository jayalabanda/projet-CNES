import datetime as dt
import json

import ee
from sentinelsat import SentinelAPI

import utils.data_collection as dc
import utils.image_processing as ip
import utils.land_coverage as lc

ee.Authenticate()
ee.Initialize()


geojson_path = "data/geojson_files/montguers.geojson"
credentials_path = "secrets/sentinel_api_credentials.json"

fire_name = 'montguers'
wildfire_date = dt.date(2020, 8, 26)
observation_interval = 16
path = 'data/' + fire_name + '/'
output_folder = 'output/' + fire_name + '/'

with open(credentials_path, 'r') as infile:
    credentials = json.load(infile)

api = SentinelAPI(
    credentials["username"],
    credentials["password"]
)

dc.get_before_after_images(
    api=api,
    wildfire_date=wildfire_date,
    geojson_path=geojson_path,
    observation_interval=observation_interval,
    path=path,
    fire_name=fire_name,
    output_folder=output_folder,
    resolution=10,
    cloud_threshold=40
)

diff = dc.plot_ndvi_difference(output_folder, fire_name, figsize=(10, 10))
