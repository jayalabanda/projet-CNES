import requests
import json
import pandas as pd
import numpy as np
import time


credentials_path = "../secrets/weather_api_credentials.json"
with open(credentials_path, 'r') as infile:
    credentials = json.load(infile)

key = credentials['weather_key']
date1 = '2021-08-16'
date2 = '2021-08-17'

coordinates_data_path = "../data/coords_utm_var.csv"
coords_data = pd.read_csv(coordinates_data_path)
np.random.seed(0)
choose = 10
random_idxs = np.random.choice(
    coords_data.shape[0], size=choose, replace=False)
as_json = True

for i in random_idxs:
    lat = coords_data.iloc[i, 0]
    lon = coords_data.iloc[i, 1]

    # build the URL
    url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    url = url + str(lat) + '%2C' + str(lon)
    url = url + '/' + date1 + '/' + date2
    url = url + '?key=' + key
    url += '&elements=windspeed,winddir&include=hours'

    # check if the type should be JSON or CSV
    url += '&contentType=csv' if not as_json else '&contentType=json'

    try:
        api_request = requests.get(url)
        time.sleep(1)  # we don't want to be too fast

        if as_json:
            with open('data/weather_api.json', 'w') as outfile:
                json.dump(api, outfile, indent=4, sort_keys=True)
        else:
            with open(f'temp_data{i}.csv', 'w') as outfile:
                outfile.write(api_request.text)

    except Exception as e:
        api = 'Error getting request.'
