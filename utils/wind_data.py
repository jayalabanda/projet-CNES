import os

import cdsapi
import xarray as xr
from ipyleaflet import (FullScreenControl, LayersControl, Map, Marker,
                        WidgetControl, basemaps)
from ipyleaflet.velocity import Velocity
from ipywidgets import IntSlider, jslink


def retrieve_wind_data(fire_name, year, month, day, hours,
                       output_path='data/nc_files/'):
    """Retrieve wind data from Climate Data Store and save to netCDF file.

    Args:
        year (str): year of data to retrieve
        month (str): month of data to retrieve
        day (str): day of data to retrieve
        hours (list): hours of data to retrieve.
            Can be a single hour or a list of hours.
        output_path (str): path to save the file. Default is `data/nc_files/`
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f"{output_path}wind_data_{fire_name}_{year}_{month}_{day}.nc"

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': hours,
            'grid': "2.5/2.5",
        },
        output_file,
    )
    return output_file


def open_nc_data(path, **kwargs):
    """Open netCDF file and return data as xarray Dataset.

    Args:
        path (str): path to netCDF file
        **kwargs: keyword arguments to pass to `xarray.open_dataset`
    """
    return xr.open_dataset(path, **kwargs)


def reshape_data(ds):
    """Reshape data to be compatible with `ipyleaflet.Velocity`.

    Args:
        ds (xarray.Dataset): dataset containing wind data
    """
    ds = ds.isel(time=0)
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    return ds


def create_map(ds, center, zoom=5,
               basemap=basemaps.CartoDB.DarkMatter,
               add_zoom_slider=True,
               add_layers_control=True,
               add_full_screen=True,):
    """Create map with wind velocity data.

    The map can be fully customized. Please refer to the
    ipyleaflet documentation for more details:
    https://ipyleaflet.readthedocs.io/en/latest/index.html

    Args:
        ds (xarray.Dataset): dataset containing wind data
        center (list): center of map in [lat, lon] format
        zoom (int): zoom level of map
        basemap (ipyleaflet.basemaps): basemap to use

    Returns:
        ipyleaflet.Map: map with wind velocity data
    """
    m = Map(center=center,
            zoom=zoom,
            interpolation='nearest',
            basemap=basemap)

    display_options = {
        'velocityType': 'Global Wind',
        'displayPosition': 'bottomleft',
        'displayEmptyString': 'No wind data'
    }

    # Add winds to the map
    wind = Velocity(data=ds,
                    zonal_speed='u10',
                    meridional_speed='v10',
                    latitude_dimension='lat',
                    longitude_dimension='lon',
                    velocity_scale=0.01,
                    max_velocity=20,
                    display_options=display_options,
                    name='Winds')
    m.add_layer(wind)

    # Add marker to indicate wildfire location
    marker = Marker(location=center, draggable=False,
                    name="Location of Wildfire")
    m.add_layer(marker)

    # Add zoom slider
    if add_zoom_slider:
        zoom_slider = IntSlider(description='Zoom level:',
                                min=0, max=15, value=zoom)
        jslink((zoom_slider, 'value'), (m, 'zoom'))
        widget_control1 = WidgetControl(
            widget=zoom_slider, position='topright')
        m.add_control(widget_control1)

    # Add fullscreen control
    if add_full_screen:
        m.add_control(FullScreenControl())

    # Add layer control
    if add_layers_control:
        m.add_control(LayersControl(position='topright'))

    return m


def save_map(map_, fire_name, output_path='output/maps/'):
    """Save map to html file.

    Args:
        map_ (ipyleaflet.Map): map to save
        output_path (str): path to save map to
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    map_.save(output_path + 'wind_map_' + fire_name + '.html')
