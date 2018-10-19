import numpy as np
import xarray as xr
import xesmf as xe
from requests import get

# months, models, experiments, variables covered
month   = np.arange('2006-01', '2007-01', dtype='datetime64[M]').astype('datetime64[ns]')
mod     = np.array(['AM2', 'CAM3', 'CAM4', 'CNRM-AM6-DIA-v2', 'CaltechGray',
                    'ECHAM-6.1', 'ECHAM-6.3', 'IPSL-CM5A', 'MIROC5', 'MPAS',
                    'MetUM-GA6-CTL', 'MetUM-GA6-ENT', 'NorESM2'])
exp     = np.array(['Aqua4xCO2', 'AquaControl', 'Land4xCO2', 'LandControl', 'LandOrbit'])
var     = np.array(['clt', 'FLDS', 'FLDSC', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLUT', 'FLUTC', 'FSDS',
                    'FSDSC', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'hfls', 'hfss', 'pr', 'prc', 'prsn',
                    'prw', 'ps', 'psl', 'rlds', 'rldscs', 'rldt', 'rldtcs', 'rlus', 'rluscs', 'rlut',
                    'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsdtcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs',
                    'snow', 'ts'])
var_lev = np.array(['hur', 'hus', 'ua', 'va', 'wap', 'zg'])

# standard latitude, longitude, pressure
lat  = np.linspace(-89.5, 89.5, 180)
lon  = np.linspace(-179.0, 179.0, 180)
plev = np.array([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400,
                 500, 600, 700, 850, 925, 1000]) * 100

# missing MPAS zeta pressure levels
plev_MPAS = np.array([3.544638, 7.388813, 13.967214, 23.944626, 37.23029, 53.114605,
                      70.05915, 85.43912, 100.514694, 118.250336, 139.1154, 163.66206,
                      192.53993, 226.51326, 266.48114, 313.50125, 368.818, 433.89523,
                      510.45526, 600.5242, 696.79626, 787.7021, 867.16077, 929.64886,
                      970.5548, 992.5561]) * 100

# grid we want lat/lon to be interpolated to
ds_out = xr.Dataset({'lat': (['lat'], lat),
                     'lon': (['lon'], lon)})

def get_variables():
    # fetch data for all variables
    ds = xr.merge(list(map(lambda v : get_experiments(v, False), var)))
    ds_plev = xr.merge(list(map(lambda v : get_experiments(v, True), var_lev)))
    # adjust precip, clouds, humidity, pressure
    ds['pr'].values *= 86400
    for i in range(13):
        if ds['clt'].values[0, i, 0, 0, 0] > 1:
            ds['clt'].values[:, i, :, :, :] *= 1.0 / 100.0
        if ds_plev['hur'].values[0, i, 0, -2, 0, 0] > 1:
            ds_plev['hur'].values[:, i, :, :, :, :] *= 1.0 / 100.0
        if i not in [0, 8]:
            ds['psl'].values[:, i, :, :, :] *= 1.0 / 100.0
            ds['ps'].values[:, i, :, :, :] *= 1.0 / 100.0
    ds.to_netcdf('../data/master')
    print('data saved to /data/master')
    ds_plev.to_netcdf('../data/master_plev')
    print('plev data saved to /data/master_plev')
    return ds, ds_plev
    
def get_experiments(variable, lev):
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda e : get_models(e, variable, lev).values, exp)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[exp, mod, month, plev, lat, lon],
                          dims=['exp', 'model', 'time', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[exp, mod, month, lat, lon],
                          dims=['exp', 'model', 'time', 'lat', 'lon'])
    dr.name = variable
    return dr

def get_models(experiment, variable, lev):
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda m : regrid_data(m, experiment, variable, lev).values, mod)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[mod, month, plev, lat, lon],
                          dims=['model', 'time', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[mod, month, lat, lon],
                          dims=['model', 'time', 'lat', 'lon'])
    return dr
        
def regrid_data(model, experiment, variable, lev):
    try:
        ds = fetch_data(model, experiment, variable)
        # make sure latitude from data is ascending
        if ds['lat'].values[0] > ds['lat'].values[1]:
            ds = ds.sel(lat=slice(None, None, -1))
        # make sure pressure is ascending and interpolate if necessary
        if lev:
            if ds['plev'].values[0] > ds['plev'].values[1]:
                ds = ds.sel(plev=slice(None, None, -1))
            if ds['plev'].size != 17:
                if 'MPAS' == 'MPAS':
                    ds['plev'] = plev_MPAS
                else:
                    ds['plev'] *= 100
                ds = ds.interp(plev=plev, kwargs={'fill_value': 'extrapolate'})
        # for all models except IPSL lon is from [0, 360], but we want to data on [-180, 180]
        if model != 'IPSL-CM5A':
            ds = ds.roll(lon=(ds['lon'].size//2))
            auxlon = ds['lon'].values
            auxlon[0:ds['lon'].size//2] -= 360
            ds['lon'] = auxlon
        # regrid data
        regridder = xe.Regridder(ds, ds_out, 'bilinear', 
                                 periodic=True, reuse_weights=True)
        dr = regridder(ds[variable])
        print(('\x1b[1;30;42m' +
               'regridding data for %s, %s, %s... success.' +
               '\x1b[0m') % (model, experiment, variable))
    # if we get an invalid URL/dataset return appropriate blank DataArray
    except (ValueError, IOError, KeyError) as e:
        if lev:
            dr = xr.DataArray(np.zeros((month.size, plev.size, lat.size, lon.size)) + np.nan)
        else:
            dr = xr.DataArray(np.zeros((month.size, lat.size, lon.size)) + np.nan)
        print(('\x1b[1;30;41m' +
               'regridding data for %s, %s, %s... ' + str(e) +
               '\x1b[0m') % (model, experiment, variable))
    return dr

def fetch_data(model, experiment, variable):
    # build up url string
    url = ('http://fletcher.ldeo.columbia.edu:81/home/OTHER/biasutti/netcdf/TRACMIP/AmonClimAug2nd2016/PP/'
           + model + '/' + experiment + '/' + variable)
    # use time fixed data when available
    if (model == 'CaltechGray' and 'Land' in experiment) or variable.isupper():
        url += '.nc/dods'
    else:
        url += '_tf.nc/dods'
    # check for valid URL and fetch dataset
    if get(url).status_code == 404:
        raise IOError('model/variable/experiment does not exist')
    else:
        ds = xr.open_dataset(url, decode_times=False)
    return ds