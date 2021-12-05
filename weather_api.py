import requests
import json


credentials_path = "secrets/weather_api_credentials.json"
with open(credentials_path, 'r') as infile:
    credentials = json.load(infile)

url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
key = credentials['weather_key']
date1 = '2021-08-17'
date2 = '2021-08-19'
lat, lon = 43.1295646351788, 6.515786622283026
url = url + str(lat) + '%2C' + str(lon)
url = url + '/' + date1 + '/' + date2
url = url + '?key=' + key
url += '&elements=windspeed,winddir'

# print(url)

try:
    api_request = requests.get(url)
    api = json.loads(api_request.content)
    print(json.dumps(api, indent=4, sort_keys=True))
except Exception as e:
    api = 'Error getting request.'