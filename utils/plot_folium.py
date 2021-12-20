import os
import webbrowser

import ee
import folium
import numpy as np
import pandas as pd
from folium import plugins


def get_coordinates(fire_name):
    return pd.read_csv(
        f'data/coordinates_files/coords_utm_{fire_name}.csv')


def get_location_list(fire_name, p, seed):
    coordinates = get_coordinates(fire_name)
    size = int(p * coordinates.shape[0])
    np.random.seed(seed)
    random_idxs = np.random.choice(len(coordinates), size=size, replace=False)
    random_coords = coordinates.iloc[random_idxs]
    return random_coords.values.tolist()


# Add custom basemaps to folium
def create_basemaps():
    return {
        'Google Maps': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=True,
            control=True,
        ),
        'Google Satellite': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True,
        ),
        'Google Terrain': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Terrain',
            overlay=True,
            control=True,
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True,
        ),
        'Esri Satellite': folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=True,
            control=True,
        ),
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
        print(f"Could not display {name}.")


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
            {'min': 1.0, 'max': 17.0,
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


def create_map(fire_name, p, seed):
    """Create a folium map of the burnt area using `p` percent of the number of coordinates
    in the fire CSV.

    A Google Satellite and Google Maps layer are added to the map, along with the selected
    land cover layer used in the `select_land_cover_data` function.

    The map also has a minimap and a layer control option.

    Args:
        fire_name (str): name of the fire
        p (float): percentage of coordinates to use in the map
        seed (int): seed used to generate the sampled coordinates.
    """
    # Add EE drawing method to folium.
    folium.Map.add_ee_layer = add_ee_layer

    coordinates = get_coordinates(fire_name)
    center = coordinates.mean(axis=0).to_list()
    my_map = folium.Map(location=center, zoom_start=12)

    # Add markers
    location_list = get_location_list(fire_name, p, seed)
    for point in range(len(location_list)):
        folium.Circle(location_list[point], radius=40).add_to(my_map)

    # Add basemaps
    basemaps = create_basemaps()
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
    return my_map


def save_map(map_, fire_name, output_folder):
    """Save the map to an HTML file.

    Args:
        map_ (folium.Map): map to save
        fire_name (str): name of the fire
        output_folder (str): folder to save the map to
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = f'{output_folder}map_{fire_name}.html'
    map_.save(filename)


def open_map(fire_name, output_folder):
    """Open the HTML file from the map in a web browser.

    Args:
        fire_name (str): name of the fire
        output_folder (str): folder where the HTML file is saved
    """
    filename = f'{output_folder}map_{fire_name}.html'
    webbrowser.open_new_tab('file://' + os.path.realpath(filename))
