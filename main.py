import datetime as dt
import json

import ee
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

import utils

ee.Authenticate()
ee.Initialize()


def main():
    geojson_path = "data/montguers.geojson"
    credentials_path = "secrets/sentinel_api_credentials.json"
    wildfire_date = dt.date(2020, 8, 26)
    observation_interval = 16
    path = 'data/'
    output_folder = 'output/'
    fire_name = 'Montguers'

    with open(credentials_path, 'r') as infile:
        credentials = json.load(infile)

    api = SentinelAPI(
        credentials["username"],
        credentials["password"]
    )

    before_image, after_image = utils.get_before_after_images(
        api=api, wildfire_date=wildfire_date, geojson_path=geojson_path,
        observation_interval=observation_interval, path=path,
        bands=['B04', 'B08'], fire_name=fire_name, output_folder=output_folder,
        resolution=10, cloud_threshold=40
    )
