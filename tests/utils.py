import numpy as np
import pandas as pd
import xarray as xr


def _make_dataset(size, variable_name='VHI', lonmin=-180.0, lonmax=180.0,
                  latmin=-55.152, latmax=75.024, add_times=True):

    lat_len, lon_len = size
    # create the vector
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)

    dims = ['lat', 'lon']
    coords = {'lat': latitudes,
              'lon': longitudes}

    if add_times:
        times = pd.date_range('2000-01-01', '2001-12-31', name='time')
        size = (len(times), size[0], size[1])
        dims.insert(0, 'time')
        print(type(times[0]))
        coords['time'] = times
    var = np.random.randint(100, size=size)

    ds = xr.Dataset({variable_name: (dims, var)}, coords=coords)

    return ds, (lonmin, lonmax), (latmin, latmax)
