import os
import webbrowser

import ee
import folium
import geemap
import numpy as np
import pandas as pd
from ipyleaflet import Marker, MarkerCluster
from ipyleaflet.leaflet import LayerGroup


def get_coordinates(fire_name):
    """Get the coordinates of the fire.

    Args:
        fire_name (str): name of the fire

    Returns:
        coordinates (pandas.DataFrame): coordinates of the fire
    """
    return pd.read_csv(
        f'data/coordinates_files/{fire_name}.csv')


def get_location_list(fire_name, p, seed):
    """Get the coordinates of the fire to add to the geemap map.

    Args:
        fire_name (str): name of the fire
        p (float): probability of a fire pixel being selected
        seed (int): seed for the random number generator

    Returns:
        location_list (list): list of coordinates of the fire
    """
    coordinates = get_coordinates(fire_name)
    size = int(p * coordinates.shape[0])
    np.random.seed(seed)
    random_idxs = np.random.choice(len(coordinates), size=size, replace=False)
    random_coords = coordinates.iloc[random_idxs]
    return random_coords.values.tolist()


def create_basemaps():
    """Add custom basemaps to folium map."""
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
            tiles='https://server.arcgisonline.com/ArcGIS/rest/\
                services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=True,
            control=True,
        ),
    }


def add_ee_layer(self, ee_object, vis_params, name):
    """Define a method for displaying Earth Engine image tiles
    on a folium map.
    """
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


def select_land_cover_data(choice):
    """Select the land cover data to be displayed on the map.

    Args:
        choice (int): choice of land cover data to be displayed on the map

    Returns:
        ee_object (ee.Image or ee.ImageCollection): land cover data to be
        displayed on the map
    """
    if choice == 1:
        dat = ee.ImageCollection('MODIS/006/MCD12Q1')
        return dat.select('LC_Type1')
    elif choice == 2:
        return ee.ImageCollection("ESA/WorldCover/v100").first()
    elif choice == 3:
        return ee.Image(
            "COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select(
                'discrete_classification')
    elif choice == 4:
        dat = ee.Image("COPERNICUS/CORINE/V20/100m/2018")
        return dat.select('landcover')


def add_to_map(map_, dataset, choice):
    """Add the land cover data to the map. If the map is a geemap map,
    also add a legend.

    Args:
        map_ (folium.Map or geemap.Map): map to add the land cover data to
        dataset (ee.Image or ee.ImageCollection): land cover data
        choice (int): choice of land cover data
    """
    if choice == 1:
        palette = ['05450a', '086a10', '54a708', '78d203',
                   '009900', 'c6b044', 'dcd159', 'dade48',
                   'fbff13', 'b6ff05', '27ff87', 'c24f44',
                   'a5a5a5', 'ff6d4c', '69fff8', 'f9ffa4', '1c0dff']
        map_.addLayer(
            dataset,
            {'min': 1.0, 'max': 17.0, 'palette': palette},
            'MODIS Land Cover'
        )
        map_.add_legend(
            legend_title='MODIS Land Cover',
            builtin_legend='MODIS/006/MCD12Q1')
    elif choice == 2:
        map_.addLayer(dataset, {'bands': ['Map']}, 'ESA World Cover')
        map_.add_legend(
            legend_title='ESA World Cover',
            builtin_legend='ESA_WorldCover')
    elif choice == 3:
        map_.addLayer(dataset, {}, 'Copernicus Global Land Service')
        map_.add_legend(
            legend_title='Copernicus Global Land Service',
            builtin_legend='COPERNICUS/Landcover/100m/Proba-V/Global')
    elif choice == 4:
        map_.addLayer(dataset, {}, 'Copernicus CORINE Land Cover')
        map_.add_legend(
            legend_title='Copernicus CORINE Land Cover',
            builtin_legend='COPERNICUS/CORINE/V20/100m')


def get_legend(choice):
    """Get the legend for the land cover data."""
    # Image legends are hosted online
    if choice == 1:
        return 'https://i.ibb.co/w0Jhsck/legend-MODIS.png'
    elif choice == 2:
        return 'https://i.ibb.co/BGS690y/legend-ESA.png'
    elif choice == 3:
        return 'https://i.ibb.co/hDyGSxT/legend-CGLS.png'
    elif choice == 4:
        return 'https://i.ibb.co/1sFzhf8/legend-CORINE.png'


def create_map(fire_name, prob, seed, choice,
               zoom=5, cluster=False, minimap=False):
    """Create a geemap map of the burnt area using `prob` percent of
    the number of coordinates in the fire CSV.

    A Google Satellite and Google Maps layers are added to the map,
    along with the selected land cover layer used in the
    `select_land_cover_data` function.

    The map also has a cluster and minimap options.

    For more details on the geemap map, see https://geemap.org/get-started/

    Args:
        fire_name (str): name of the fire
        prob (float): percentage of coordinates to use in the map
        seed (int): seed used to generate the sampled coordinates
        choice (int): choice of land cover data
        zoom (int): zoom level of the map. Default is 5
        cluster (bool): whether to cluster the coordinates.
        Default is `False`
        minimap (bool): whether to add a minimap to the map.
        Default is `False`

    Returns:
        geemap.Map: map of the burnt area
    """
    coordinates = get_coordinates(fire_name)
    center = coordinates.mean(axis=0).to_list()
    my_map = geemap.Map(
        location=center, zoom=zoom,
        scroll_wheel_zoom=True
    )

    # Add basemaps
    my_map.add_basemap('SATELLITE')
    my_map.add_basemap('HYBRID')

    # Add minimap
    if minimap:
        my_map.add_minimap(zoom=6)

    # Add selected land cover
    dataset = select_land_cover_data(choice)
    add_to_map(my_map, dataset, choice)

    location_list = get_location_list(fire_name, prob, seed)
    if cluster:
        # Create marker cluster
        marker_cluster = MarkerCluster(
            markers=[Marker(location=loc, draggable=False)
                     for loc in location_list],
            name='Wildfire Location',
        )
        my_map.add_layer(marker_cluster)
    else:
        # Add markers to the map
        layer_group = LayerGroup(
            name='Wildfire Location',
            layers=[Marker(location=loc, draggable=False)
                    for loc in location_list]
        )
        my_map.add_layer(layer_group)

    # Add a layer control panel to the map.
    my_map.addLayerControl()

    return my_map


def save_map(map_, fire_name, output_folder, wind=False):
    """Save the map to an HTML file.

    Args:
        map_ (geemap.Map): map to save
        fire_name (str): name of the fire
        output_folder (str): folder to save the map to
        wind (bool): whether the map contains wind data. Default is `False`
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if wind:
        filename = f'{output_folder}wind_map_{fire_name}.html'
        title = f'Wind Map for {fire_name}'
    else:
        filename = f'{output_folder}map_{fire_name}.html'
        title = f'Coordinates Map for {fire_name}'
    map_.to_html(outfile=filename, title=title)


def open_map(fire_name, output_folder, wind=False):
    """Open the HTML file from the map in a web browser.

    Args:
        fire_name (str): name of the fire
        output_folder (str): folder where the HTML file is saved
        wind (bool): whether the map contains wind data. Default is `False`
    """
    if wind:
        filename = f'{output_folder}wind_map_{fire_name}.html'
    else:
        filename = f'{output_folder}map_{fire_name}.html'
    webbrowser.open_new_tab('file://' + os.path.realpath(filename))
