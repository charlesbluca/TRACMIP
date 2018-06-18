import numpy as np
import xarray as xr
import xesmf as xe

# months, models, experiments, variables covered
month   = np.arange('2005-01', '2006-01', dtype='datetime64[M]').astype('datetime64[ns]')
model   = np.array(['AM2', 'CaltechGray', 'CAM3', 'CAM4', 'CNRM-AM6-DIA-v2', 'ECHAM-6.1',
                    'ECHAM-6.3', 'NorESM2', 'IPSL-CM5A', 'MetUM-GA6-CTL',  'MetUM-GA6-ENT', 
                    'MIROC5', 'MPAS'])
exp     = np.array(['AquaControl', 'Aqua4xCO2', 'LandControl', 'Land4xCO2', 'LandOrbit'])
var     = np.array(['pr', 'clt', 'ts', 'hfls', 'hfss', 'rlds', 'rlus', 'rldscs', 'rluscs',
                    'rsds', 'rsus', 'rsdscs', 'rsuscs'])
var_lev = np.array(['ua', 'va', 'zg'])

# standard latitude, longitude, pressure
lat  = np.linspace(-89.5, 89.5, 180)
lon  = np.linspace(-179.0, 179.0, 180)
plev = np.array([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400,
                 500, 600, 700, 850, 925, 1000]) * 100

# grid we want lat/lon to be interpolated to
ds_out = xr.Dataset({'lat': (['lat'], lat),
                     'lon': (['lon'], lon)})

def main():
    # fetch data for plev and non-plev variables
    ds = xr.merge(list(map(lambda v : get_experiments(v), var)))
    ds_plev = xr.merge(list(map(lambda v : get_experiments(v), var_lev)))
    # adjust precipitation and cloud data
    ds['pr'].values *= 86400
    for i in range(13):
        if ds['clt'].values[1, i, 1, 1, 1] > 1:
            ds['clt'].values[:, i, :, :, :] *= 1.0 / 100.0
    # save datasets to master files
    ds.to_netcdf('nc/master')
    print('data saved to nc/master')
    ds_plev.to_netcdf('nc/master_plev')
    print('plev data saved to nc/master_plev')
    return ds, ds_plev 
    
def get_experiments(var):
    # check if we need pressure levels
    lev = False
    if var in var_lev:
        lev = True
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda e : get_models(e, var, lev).values, exp)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[exp, model, month, plev, lat, lon],
                          dims=['exp', 'model', 'time', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[exp, model, month, lat, lon],
                          dims=['exp', 'model', 'time', 'lat', 'lon'])
    dr.name = var
    return dr

def get_models(exp, var, lev):
    # lambda function to fetch/regrid data by model
    data = np.array(list(map(lambda m : regrid_data(m, exp, var, lev).values, model)))
    # create large data array (based on ndim)
    if lev:
        dr = xr.DataArray(data, coords=[model, month, plev, lat, lon],
                          dims=['model', 'time', 'plev', 'lat', 'lon'])
    else:
        dr = xr.DataArray(data, coords=[model, month, lat, lon],
                          dims=['model', 'time', 'lat', 'lon'])
    return dr
        
def regrid_data(model, exp, var, lev):
    try:
        ds = fetch_data(model, exp, var)
        # make sure latitude from data is ascending
        if ds['lat'].values[0] > ds['lat'].values[1]:
            ds = ds.sel(lat=slice(None, None, -1))
        # make sure pressure from data is properly formatted and ascending
        if lev and ds['plev'].size != 17:
            raise ValueError('Irregular pressure levels')
        if lev and ds['plev'].values[0] > ds['plev'].values[1]:
            ds = ds.sel(plev=slice(None, None, -1))
        # for all models except IPSL lon is from [0, 360], but we want to data on [-180, 180]
        if model != 'IPSL-CM5A':
            ds = ds.roll(lon=(ds['lon'].size//2))
            auxlon = ds['lon'].values
            auxlon[0:ds['lon'].size//2] -= 360
            ds['lon'] = auxlon
        # regrid data
        regridder = xe.Regridder(ds, ds_out, 'bilinear', 
                                 periodic=True, reuse_weights=True)
        dr = regridder(ds[var])
        print(('\x1b[1;30;42m' +
               'regridding data for %s, %s, %s... success.' +
               '\x1b[0m') % (model, exp, var))
    # if we get an invalid URL/dataset return appropriate blank DataArray
    except (ValueError, IOError, KeyError) as e:
        if lev:
            dr = xr.DataArray(np.zeros((month.size, plev.size, lat.size, lon.size)) + np.nan)
        else:
            dr = xr.DataArray(np.zeros((month.size, lat.size, lon.size)) + np.nan)
        print(('\x1b[1;30;41m' +
               'regridding data for %s, %s, %s... ' + str(e) +
               '\x1b[0m') % (model, exp, var))
    return dr

def fetch_data(model, exp, var):
    # build up url string
    url = ('http://kage.ldeo.columbia.edu:81/home/OTHER/biasutti/netcdf/TRACMIP/AmonClimAug2nd2016/PP/'
           + model + '/'
           + exp + '/'
           + var + '_tf.nc/dods')
    # fetch xarray data array in 5 attempts
    for attempt in range(5):
        try:
            ds = xr.open_dataset(url, decode_times=False)
            break
        except IOError:
            if attempt == 4:
                raise
            else:
                continue
    return ds

if __name__ == '__main__':
    main()
