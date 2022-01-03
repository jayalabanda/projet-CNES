import os

import cdsapi
import geemap
import xarray as xr
from ipyleaflet import Marker, basemaps
from ipyleaflet.velocity import Velocity

import utils.plot_map as pm


def retrieve_wind_data(fire_name, year, month, day, hours,
                       output_path='data/nc_files/'):
    """Retrieve wind data from Climate Data Store and save to netCDF file.

    Args:
        fire_name (str): name of wildfire
        year (str): year of data to retrieve
        month (str): month of data to retrieve
        day (str): day of data to retrieve
        hours (list): hours of data to retrieve.
            Can be a single hour or a list of hours.
        output_path (str): path to save the file. Default is `data/nc_files/`
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out_file = f"{output_path}wind_data_{fire_name}_{year}_{month}_{day}.nc"

    if not os.path.exists(out_file):
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
            out_file,
        )
    else:
        print('File already exists.')
    return out_file


def open_nc_data(path, **kwargs):
    """Open netCDF file and return data as xarray Dataset.

    Args:
        path (str): path to netCDF file
        **kwargs: keyword arguments to pass to `xarray.open_dataset`

    Returns:
        xarray.Dataset: opened dataset containing wind data
    """
    return xr.open_dataset(path, **kwargs)


def reshape_data(ds):
    """Reshape data to be compatible with `ipyleaflet.Velocity`.

    Args:
        ds (xarray.Dataset): dataset containing wind data

    Returns:
        xarray.Dataset: reshaped dataset
    """
    if ds['time'].shape != (1,):
        vals = list(ds['time'].values.astype(str))
        vals = [v[11:13] for v in vals]
        if '12' in vals:
            ds = ds.isel(time=12)
            return ds
    ds = ds.isel(time=0)
    return ds


def create_map(ds, center, choice,
               zoom=5,
               basemap=basemaps.CartoDB.DarkMatter):
    """Create map with wind velocity data.

    The map can be fully customized. Please refer to the
    ipyleaflet documentation for more details:
    https://ipyleaflet.readthedocs.io/en/latest/index.html

    Args:
        ds (xarray.Dataset): dataset containing wind data
        center (list): center of map in [lat, lon] format
        choice (int): choice of land cover layer to add to map
        zoom (int): zoom level of map. Default is 5.
        basemap (ipyleaflet.basemaps): basemap to use.
            Default is `basemaps.CartoDB.DarkMatter`
        add_zoom_slider (bool): whether to add zoom slider.
            Default is `True`
        add_layers_control (bool): whether to add layers control.
            Default is `True`

    Returns:
        geemap.Map: map with wind velocity data
    """
    my_map = geemap.Map(center=center,
                        zoom=zoom,
                        interpolation='nearest',
                        basemap=basemap,
                        scroll_wheel_zoom=True)

    display_options = {
        'velocityType': 'Global Wind',
        'displayPosition': 'bottomleft',
        'displayEmptyString': 'No wind data'
    }

    # Add winds to the map
    wind_data = Velocity(data=ds,
                         zonal_speed='u10',
                         meridional_speed='v10',
                         latitude_dimension='latitude',
                         longitude_dimension='longitude',
                         velocity_scale=0.02,
                         max_velocity=20,
                         display_options=display_options,
                         name='Winds')
    my_map.add_layer(wind_data)

    # Add marker to indicate wildfire location
    marker = Marker(location=center, draggable=False,
                    name="Location of Wildfire")
    my_map.add_layer(marker)

    # Add selected land cover
    lc_layer = pm.select_land_cover_data(choice)
    pm.add_to_map(my_map, lc_layer, choice)

    # Add a layer control panel to the map.
    my_map.add_layer_control()

    return my_map
