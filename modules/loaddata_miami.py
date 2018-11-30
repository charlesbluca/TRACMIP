import numpy as np
import xarray as xr
import xesmf as xe
import times

# months, models, experiments, variables covered
month   = np.arange(1, 13)
mod     = np.array(['AM2', 'CAM3', 'CAM4', 'CNRM-AM6-DIA-v2', 'CaltechGray',
                    'ECHAM-6.1', 'ECHAM-6.3', 'IPSL-CM5A', 'MIROC5', 'MPAS',
                    'MetUM-GA6-CTL', 'MetUM-GA6-ENT', 'NorESM2'])
exp     = np.array(['Aqua4xCO2', 'AquaControl', 'Land4xCO2', 'LandControl', 'LandOrbit'])
var     = np.array(['clt', 'hfls', 'hfss', 'pr', 'rlds', 'rldscs', 'rlus', 'rluscs',
                    'rsds', 'rsdscs', 'rsus', 'rsuscs', 'ts'])
var_lev = np.array(['ua', 'va', 'zg'])

# standard latitude, longitude, pressure
lat  = np.linspace(-89.5, 89.5, 180)
lon  = np.linspace(-179.0, 179.0, 180)
plev = np.array([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400,
                 500, 600, 700, 850, 925, 1000]) * 100

# missing AM2 lat/lon values
lat_AM2 = np.array([-89.49438, -87.97753, -85.95506, -83.93259, -81.91011, -79.88764,
                    -77.86517, -75.8427, -73.82022, -71.79775, -69.77528, -67.75281,
                    -65.73034, -63.70787, -61.68539, -59.66292, -57.64045, -55.61798,
                    -53.5955, -51.57303, -49.55056, -47.52809, -45.50562, -43.48315,
                    -41.46067, -39.4382, -37.41573, -35.39326, -33.37078, -31.34831,
                    -29.32584, -27.30337, -25.2809, -23.25843, -21.23595, -19.21348,
                    -17.19101, -15.16854, -13.14607, -11.1236, -9.101124, -7.078652,
                    -5.05618, -3.033708, -1.011236])
lat_AM2 = np.concatenate((lat_AM2, np.flip(-lat_AM2, axis=0)))
lon_AM2 = np.arange(1.25, 360.00, 2.50)

# missing MPAS plev values
plev_MPAS = np.array([3.544638, 7.3888135, 13.967214, 23.944625,  37.23029, 53.114605,
                      70.05915, 85.439115, 100.514695, 118.250335, 139.115395, 163.66207,
                      192.539935, 226.513265, 266.481155, 313.501265, 368.81798, 433.895225,
                      510.455255, 600.5242, 696.79629, 787.70206, 867.16076, 929.648875,
                      970.55483, 992.5561]) * 100

# grid we want lat/lon to be interpolated to
ds_out = xr.Dataset({'lat': (['lat'], lat),
                     'lon': (['lon'], lon)})

def get_experiments(variable, lev):
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda e : get_models(e, variable, lev).values, exp)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[exp, mod, month, plev, lat, lon],
                          dims=['exp', 'model', 'month', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[exp, mod, month, lat, lon],
                          dims=['exp', 'model', 'month', 'lat', 'lon'])
    dr.name = variable
    return dr

def get_models(experiment, variable, lev):
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda m : regrid_data(m, experiment, variable, lev).values, mod)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[mod, month, plev, lat, lon],
                          dims=['model', 'month', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[mod, month, lat, lon],
                          dims=['model', 'month', 'lat', 'lon'])
    return dr

def regrid_data(model, experiment, variable, lev):
    try:
        varname = get_varname(model, variable)
        ds = fetch_data(model, experiment, varname).isel(time=slice(0,24))
        # fill in blank data for AM2
        if model == 'AM2':
            ds['lat'] = lat_AM2
            ds['lon'] = lon_AM2
        # get monthly means
        if lev:
            data = np.zeros((12, ds.shape[1], ds.shape[2], ds.shape[3]))
            for i in range(12):
                data[i, :, :, :] = np.nanmean(ds[i:-1:12, :, :, :], axis=0)
            ds_mm = xr.DataArray(data, coords=[range(1,13), ds[get_levname(model)], ds.lat, ds.lon],
                                 dims=['month', 'plev', 'lat', 'lon'])
        else:
            data = np.zeros((12, ds.shape[1], ds.shape[2]))
            for i in range(12):
                data[i, :, :] = np.nanmean(ds[i:-1:12, :, :], axis=0)
            ds_mm = xr.DataArray(data, coords=[range(1,13), ds.lat, ds.lon],
                                 dims=['month', 'lat', 'lon'])
        # make sure latitude from data is ascending
        if ds_mm['lat'].values[0] > ds_mm['lat'].values[1]:
            ds_mm = ds_mm.sel(lat=slice(None, None, -1))
        # for all models except IPSL lon is from [0, 360], but we want data on [-180, 180]
        if model != 'IPSL-CM5A':
            ds_mm = ds_mm.roll(lon=(ds_mm['lon'].size//2))
            auxlon = ds_mm['lon'].values
            auxlon[0:ds_mm['lon'].size//2] -= 360
            ds_mm['lon'] = auxlon
        # regrid data
        regridder = xe.Regridder(ds_mm, ds_out, 'bilinear', 
                                 periodic=True, reuse_weights=True)
        dr = regridder(ds_mm)
        # make sure pressure is ascending and interpolate if necessary
        if lev:
            if dr['plev'].values[0] > dr['plev'].values[1]:
                dr = dr.sel(plev=slice(None, None, -1))
            if dr['plev'].size != 17:
                if model in ['MIROC5', 'NorESM2']:
                    dr['plev'] *= 100000
                elif model == 'MPAS':
                    dr['plev'] = plev_MPAS
                else:
                    dr['plev'] *= 100
                dr = dr.interp(plev=plev)
        print(('\x1b[1;30;42m' +
               'regridding data for %s, %s, %s... success.' +
               '\x1b[0m') % (model, experiment, variable))
    # if we get an invalid URL/dataset return appropriate blank DataArray
    except (FileNotFoundError, UnboundLocalError, KeyError) as e:
        if lev:
            dr = xr.DataArray(np.zeros((month.size, plev.size, lat.size, lon.size)) + np.nan)
        else:
            dr = xr.DataArray(np.zeros((month.size, lat.size, lon.size)) + np.nan)
        print(('\x1b[1;30;41m' +
               'regridding data for %s, %s, %s... ' + str(e) +
               '\x1b[0m') % (model, experiment, variable))
    return dr

def fetch_data(model, experiment, variable, **kwargs):
    # open corresponding csv files
    lines = open('../csv/' + model + '_' + experiment + '.csv').readlines()
    # handle models with data in single file
    if model in ['AM2', 'CAM3', 'CAM4', 'CaltechGray', 'MPAS']:
        line = lines[1]
    # handle cmorized data, iterate to find variable
    else:
        for l in lines:
            if variable == l.split("_")[0]:
                line = l
                break
    # generate url
    url = ('https://weather.rsmas.miami.edu/repository/opendap/' + 
           line.split(',')[1] + 
           '/entry.das')
    # fetch dataset (IPSL has unique variable names)
    if model == 'IPSL-CM5A' and variable == 'cl' and experiment == 'AquaControl':
        ds = xr.open_dataset(url, decode_times=False, **kwargs)['rneb']
    else:
        ds = xr.open_dataset(url, decode_times=False, **kwargs)[variable]
    return ds

def get_varname(model, variable):
    var_names = dict.fromkeys(mod)
    if variable == 'cl':
        for m in var_names.keys():
            var_names[m] = 'cl'
        for m in ['CAM3', 'CAM4', 'MPAS']:
            var_names[m] = 'CLOUD'
        for m in ['CNRM-AM6-DIA-v2', 'CaltechGray']:
            var_names[m] = '?'    
    return var_names[model]

def get_levname(model):
    lev_names = dict.fromkeys(mod)
    for m in lev_names.keys():
        lev_names[m] = 'lev'
    for m in ['ECHAM-6.1', 'ECHAM-6.3', 'IPSL-CM5A', 'MetUM-GA6-CTL', 'MetUM-GA6-ENT']:
        lev_names[m] = 'plev'
    lev_names['MIROC5'] = 'HETA40'
    return lev_names[model]
