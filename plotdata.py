# load modules
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

# load own modules
import tracmipmodels
import atmosphere as atm

def contourmap_allexperiments(z, anom=True, plotname=None, **kwargs):
    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')

    # plot model median
    ax = plt.subplot(5, 3, 1)
    if anom:
        ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                               edgecolor='black', linewidth=1))
    c = z.median(dim='model').plot(robust=False, **kwargs)
    ax.set_title('Model median')
        
    # plot all models
    for i in range(13):
        ax = plt.subplot(5, 3, i + 3)
        if anom:
            ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                                   edgecolor='black', linewidth=1))
        z.isel(model=i).plot(robust=False, **kwargs)
        ax.set_title(z.model.values[i])

    # layout and save plot
    plt.subplots_adjust(wspace=0.04, hspace=0.05)
    plt.tight_layout()
    if plotname != None:
        plt.savefig('figs/' + plotname + '.pdf')
        print('saved to figs/' + plotname + '.pdf')
    plt.show()

def quivermap_allexperiments(step, plev=92500, anom=True, plotname=None, **kwargs):
    # fetch wind data
    ds_plev = xr.open_dataset('nc/master_plev')
    
    u = ds_plev['ua'].mean(dim='time', skipna=True).sel(plev=plev)
    v = ds_plev['va'].mean(dim='time', skipna=True).sel(plev=plev)
    
    if anom:
        u_i = u.sel(exp='LandControl') - u.sel(exp='AquaControl')
        v_i = v.sel(exp='LandControl') - v.sel(exp='AquaControl')
    else:
        u_i = u.sel(exp='AquaControl')
        v_i = v.sel(exp='AquaControl')

    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')

    # plot model median
    ax = plt.subplot(5, 3, 1)
    if anom:
        ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                               edgecolor='black', linewidth=1))
    u.sel(exp='AquaControl').median(dim='model').plot(robust=True, **kwargs)
    plt.quiver(u.lon[::step], u.lat[::step], 
               u_i.median(dim='model')[::step, ::step],
               v_i.median(dim='model')[::step, ::step],
               units='width', pivot='mid')
    ax.set_title('Model median')

    # plot all models
    for i in range(13):
        ax = plt.subplot(5, 3, i + 3)
        if anom:
            ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                                   edgecolor='black', linewidth=1))
        u.sel(exp='AquaControl').isel(model=i).plot(robust=True, **kwargs)
        plt.quiver(u.lon[::step], u.lat[::step], 
                   u_i.isel(model=i)[::step, ::step],
                   v_i.isel(model=i)[::step, ::step],
               units='width', pivot='mid', edgecolors='w')
        ax.set_title(u.model.values[i])

    # layout and save plot
    plt.subplots_adjust(wspace=0.04, hspace=0.05)
    plt.tight_layout()
    if plotname != None:
        plt.savefig('figs/' + plotname + '.pdf')
        print('saved to figs/' + plotname + '.pdf')
    plt.show()

def hovmoller_allexperiments(var, unit, vmin, vmax, ocean=True,
                             line=False, plotname=None, **kwargs):

    # load month mean data
    lat, lon, aqct, ldct = get_latlondata(var, 'month')
    sinlat = np.sin(lat*np.pi/180)
    month  = np.arange(1, 13)

    # get model names and numbers
    modelnames    = tracmipmodels.get_modelnames()
    modelsubplots = tracmipmodels.get_modelsubplots()

    # separate data over land and ocean
    land_ldct = np.nanmedian(ldct[:, :, :, 90:112], axis=3)
    land_aqct = np.nanmedian(aqct[:, :, :, 90:112], axis=3)
    ocea_ldct = np.nanmedian(np.concatenate((ldct[:, :, :, 0:90], ldct[:, :, :, 112:]), axis=3), axis=3)
    ocea_aqct = np.nanmedian(np.concatenate((aqct[:, :, :, 0:90], aqct[:, :, :, 112:]), axis=3), axis=3)

    # calculate ITCZ line for ocean/land specified
    if line:
        itcz_land, itcz_ocea = get_itcz(14, 12, lat)
        w = itcz_land
        if ocean:
            w = itcz_ocea

    # compute Homoller over ocean or land if specified
    z = land_ldct - land_aqct
    if ocean:
        z = ocea_ldct - ocea_aqct

    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')
    clev = MaxNLocator(nbins=13).tick_values(vmin, vmax)

    # plot model median
    ax, c = make_hovmoller(month, sinlat, np.nanmedian(z, axis=0),
                           clev, 1, 'Model median', **kwargs)
    if line:
        plt.plot(month, np.sin(np.nanmedian(w, axis=0) * np.pi/180), 'k', linewidth=3)
    ax.yaxis.set_ticklabels(['30S', 'Eq', '30N'], fontsize=11)

    # plot colorbar
    ax = plt.subplot(5, 3, 2)
    ax.axis('off')
    cbar = plt.colorbar(c, orientation='horizontal', aspect=30)
    cbar.ax.tick_params(labelsize=10)
    plt.text(1, -0.17, unit, fontsize=11, ha='right')

    # plot all models except GISS
    for m in range(13):
        msubplot = modelsubplots[m]
        ax, c = make_hovmoller(month, sinlat, z[m, :, :],
                               clev, msubplot, modelnames[m], **kwargs)
        if line:
            plt.plot(month, np.sin(w[m, :] * np.pi/180), 'k', linewidth=3)
        if (msubplot == 13) or (msubplot == 14) or (msubplot == 15):
            ax.xaxis.set_ticklabels(['Jan', '', '', 'Apr', '', '', 'Jul', '', '' ,'Oct', '', ''], fontsize=10)
        if msubplot in [1, 4, 7, 10, 13]:
            ax.yaxis.set_ticklabels(['30S', 'Eq', '30N'], fontsize=11)

    # layout and save plot
    plt.subplots_adjust(wspace=0.04, hspace=0.05)
    plt.tight_layout
    if plotname != None:
        print('saved to plot/figs/' + plotname + '.pdf')
        plt.savefig('plot/figs/' + plotname + '.pdf')
    plt.show()

def get_latlondata(var, time):

    # load annual or monthly mean file
    file    = np.load('calc/npz/' + var + '_latlon' + time + 'mean.npz')
    lat     = file['lat']
    lon     = file['lon']
    aqct = file[var + '_aqct']
    ldct = file[var + '_ldct']

    return lat, lon, aqct, ldct

def get_itcz(nmod, nmonth, lat):

    # load pr data to calculate ITCZ
    _ ,  _ , pr_aqct, pr_ldct = get_latlondata('pr', 'month')

    # separate data over land and ocean
    pr_land_ldct = np.nanmedian(pr_ldct[:, :, :, 90:112], axis=3)
    pr_ocea_ldct = np.nanmedian(np.concatenate((pr_ldct[:, :, :, 0:90], pr_ldct[:, :, :, 112:]), axis=3), axis=3)

    # itcz position for precip averaged over land and ocean
    itcz_land = np.zeros((nmod, nmonth)) + np.nan
    itcz_ocea = np.zeros((nmod, nmonth)) + np.nan
    for m in range(nmod):
        for i in range(nmonth):
            itcz_land[m, i] = atm.get_itczposition(pr_land_ldct[m, i, :], lat, 30.0, 0.1)
            itcz_ocea[m, i] = atm.get_itczposition(pr_ocea_ldct[m, i, :], lat, 30.0, 0.1)

    return itcz_land, itcz_ocea

def make_contourmap(x, y, z, levels, msubplot, name, **kwargs):

    ax = plt.subplot(5, 3, msubplot)
    c = plt.contourf(x, y, z, levels=levels, extend='both', **kwargs)
    ax.add_patch(Rectangle((0, -0.5), 45, 1, alpha=1, facecolor='none',
                           edgecolor='gray', linewidth=2))
    plt.plot([-200, 200], [0, 0], 'k--')
    make_map(ax, name)

    return ax, c

def make_quivermap(x, y, u, v, msubplot, name):

    ax = plt.subplot(5, 3, msubplot)
    c = np.hypot(u, v)
    q = plt.quiver(x, y, u, v, c, units='width', pivot='mid')
    ax.add_patch(Rectangle((0, -0.5), 45, 1, alpha=1, facecolor='none',
                           edgecolor='gray', linewidth=2))
    plt.plot([-200, 200], [0, 0], 'k--')
    make_map(ax, name)

    return ax, q

def make_hovmoller(x, y, z, levels, msubplot, name, **kwargs):

    ax = plt.subplot(5, 3, msubplot)
    c = plt.contourf(x, y, np.transpose(z), levels=levels, extend='both', **kwargs)
    plt.plot([-200, 200], [0, 0], 'k--')
    make_seasonalcycle(ax, name)

    return ax, c

def make_map(ax, modelname):

    plt.xlim(-178, 178), plt.ylim(-0.98, 0.98)
    ax.xaxis.set_ticks([-120, -60, 0, 60, 120])
    ax.xaxis.set_ticklabels([''],fontsize=10)
    ax.yaxis.set_ticks([-0.866, -0.5, 0, 0.5, 0.866])
    ax.yaxis.set_ticklabels([''], fontsize=10)
    plt.text(0.03, 0.93, modelname, fontsize=15, ha='left', va='center', \
             transform=ax.transAxes, backgroundcolor='white')

def make_seasonalcycle(ax, modelname):

    plt.xlim(1, 12), plt.ylim(-0.6, 0.6)
    ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.xaxis.set_ticklabels([''], fontsize=10)
    ax.yaxis.set_ticks([-0.5, 0, 0.5])
    ax.yaxis.set_ticklabels([''], fontsize=10)
    plt.text(0.02, 0.92, modelname, fontsize=14, ha='left', va='center', \
             transform=ax.transAxes, backgroundcolor='white')
