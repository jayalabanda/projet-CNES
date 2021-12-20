import os
import webbrowser

import ee
import folium
import numpy as np
import pandas as pd
from folium import plugins

ee.Initialize()

fire_name = 'istres'
coordinates = pd.read_csv(
    f'data/coordinates_files/coords_utm_{fire_name}.csv')
size = int(0.1 * coordinates.shape[0])

np.random.seed(0)
random_idxs = np.random.choice(len(coordinates), size=size, replace=False)
random_coords = coordinates.iloc[random_idxs]
location_list = random_coords.values.tolist()

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=True,
        control=True
    ),
    'Google Satellite': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Google Terrain': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Terrain',
        overlay=True,
        control=True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=True,
        control=True
    )
}


# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    try:
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):
            folium.GeoJson(
                data=ee_object.getInfo(),
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)

    except:
        print("Could not display {}".format(name))


# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer


def select_land_cover_data():
    while True:
        try:
            print(
                """
                Please select the land cover data you want to use.
                The choices are:
                    1. MODIS Land Cover (2018, 500m)
                    2. ESA World Cover (2020, 10m)
                    3. Copernicus Global Land Service (2019, 100m)
                    4. Copernicus Corine Land Cover (2018, 100m)
                """
            )
            choice = int(input('Select land cover data: '))
            if choice == 1:
                dat = ee.ImageCollection('MODIS/006/MCD12Q1')
                return dat.select('LC_Type1'), choice
            elif choice == 2:
                return ee.ImageCollection("ESA/WorldCover/v100").first(), choice
            elif choice == 3:
                return ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification'), choice
            elif choice == 4:
                dat = ee.Image("COPERNICUS/CORINE/V20/100m/2018")
                return dat.select('landcover'), choice
        except ValueError:
            print('Invalid input. Please try again.')


def add_to_map(map_, dataset, choice):
    if choice == 1:
        map_.add_ee_layer(
            dataset,
            vis_params={
                'min': 1.0, 'max': 17.0,
                'palette': ['05450a', '086a10', '54a708', '78d203', '009900', 'c6b044', 'dcd159',
                            'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5', 'ff6d4c',
                            '69fff8', 'f9ffa4', '1c0dff']},
            name='MODIS Land Cover')
    elif choice == 2:
        map_.add_ee_layer(dataset, {}, 'ESA World Cover')
    elif choice == 3:
        map_.add_ee_layer(dataset, {}, 'Copernicus Global Land Service')
    elif choice == 4:
        map_.add_ee_layer(dataset, {}, 'Copernicus Corine Land Cover')


center = coordinates.mean(axis=0).to_list()
my_map = folium.Map(location=center, zoom_start=12)

for point in range(len(location_list)):
    folium.Marker(location_list[point]).add_to(my_map)

# Add basemaps
basemaps['Google Maps'].add_to(my_map)
basemaps['Google Satellite Hybrid'].add_to(my_map)

# Add minimap
minimap = plugins.MiniMap()
my_map.add_child(minimap)

# Add selected land cover
dataset, choice = select_land_cover_data()
add_to_map(my_map, dataset, choice)

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Add fullscreen button
plugins.Fullscreen().add_to(my_map)

# Display the map.
# my_map

# Save the map
if not os.path.exists('output/maps/'):
    os.makedirs('output/maps/')

filename = f'output/maps/map_{fire_name}.html'
my_map.save(filename)

webbrowser.open_new_tab('file://' + os.path.realpath(filename))
