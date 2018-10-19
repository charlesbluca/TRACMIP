# load modules
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

badmodels = ['AM2', 'CaltechGray', 'ECHAM-6.3', 'MetUM-GA6-CTL',
             'MetUM-GA6-ENT']

def plot_allmodels(z, units=None, box=True, plotname=None, **kwargs):
    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')
    # plot model median of only the correct models
    ax = plt.subplot(5, 3, 1)
    if box:
        ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                               edgecolor='black', linewidth=1))
    z.sel(model=list(set(z.model.values) - set(badmodels))).median(dim='model').plot(**kwargs)
    if units is not None:
        plt.text(1.225, 0.75, units, va='center', rotation=90, transform=ax.transAxes)
    ax.set_title('Model median (corrected)')
    # plot all models
    for i, modname in list(enumerate(z.model.values)):
        ax = plt.subplot(5, 3, i + 2)
        if box:
            ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                                   edgecolor='black', linewidth=1))
        z.isel(model=i).plot(**kwargs)
        if modname in badmodels:
            ax.set_title(modname, color='r')
        else:
            ax.set_title(modname)
    # layout and save plot
    saveplot(plotname)
    
def plot_allvariables(z, box=True, plotname=None, variables=None, **kwargs):
    # work out rows of plots needed
    data_vars = z.data_vars
    if variables is not None:
        data_vars = variables
    row = len(data_vars) // 5 + 1
    # plotting
    plt.figure(figsize=(30, 4 * row), dpi=80, facecolor='w', edgecolor='k')
    # plot specified variables for model
    for i, var in list(enumerate(data_vars)):
        ax = plt.subplot(row, 5, i + 1)
        if box:
            ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                                   edgecolor='black', linewidth=1))
        # default to filled contour plot
        try:
            z[var].plot.contourf(**kwargs)
        except ValueError:
            z[var].plot(**kwargs)
    # layout and save plot
    saveplot(plotname)

def quivermap_allexperiments(u, v, z=None, units=None,
                             step=5, box=True, plotname=None, **kwargs):
    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')
    # plot model median
    ax = plt.subplot(5, 3, 1)
    if box:
        ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                               edgecolor='black', linewidth=1))
    if z is not None:
        z.sel(model=list(set(z.model.values) - set(badmodels))).median(dim='model').plot.contourf(**kwargs)
        plt.text(1.225, 0.5, units, va='center', rotation=90, transform=ax.transAxes)
    q = plt.quiver(u.lon[::step], u.lat[::step], 
                   u.sel(model=list(set(z.model.values) - set(badmodels))).median(dim='model')[::step, ::step],
                   v.sel(model=list(set(z.model.values) - set(badmodels))).median(dim='model')[::step, ::step],
                   units='width', pivot='mid')
    qk = plt.quiverkey(q, .9, 1.04, 3, '3 m/s', labelpos='E',
                       coordinates='axes')
    ax.set_title('Model median (corrected)')
    # plot all models
    for i, modname in list(enumerate(u.model.values)):
        ax = plt.subplot(5, 3, i + 2)
        if box:
            ax.add_patch(Rectangle((0, -30), 45, 60, alpha=1, facecolor='none',
                                   edgecolor='black', linewidth=1))
        if z is not None:
            z.isel(model=i).plot.contourf(**kwargs)
        q = plt.quiver(u.lon[::step], u.lat[::step], 
                       u.isel(model=i)[::step, ::step],
                       v.isel(model=i)[::step, ::step],
                       units='width', pivot='mid')
        qk = plt.quiverkey(q, .9, 1.04, 3, '3 m/s', labelpos='E',
                           coordinates='axes')
        if modname in badmodels:
            ax.set_title(modname, color='r')
        else:
            ax.set_title(modname)
    # layout and save plot
    saveplot(plotname)
    
def quivermap_plev_allexperiments(u, v, x=None, y=None, z=None, units=None,
                                  step=5, box=True, plotname=None, **kwargs):
    # plotting
    plt.figure(figsize=(18, 20), dpi=80, facecolor='w', edgecolor='k')
    # plot model median
    ax = plt.subplot(5, 3, 1)
    if z is not None:
        z.median(dim='model').plot.contourf(x='lon', add_labels=False, **kwargs)
        plt.text(1.225, 0.5, units, va='center', rotation=90, transform=ax.transAxes)
    q = plt.quiver(u.lon[::step], u.plev, 
                   u.median(dim='model')[::step],
                   v.median(dim='model')[::step],
                   units='width', pivot='mid')
    ax.set_title('Model median')
    # plot all models
    for i in range(13):
        ax = plt.subplot(5, 3, i + 3)
        if z is not None:
            z.isel(model=i).plot.contourf(x='lon', add_labels=False, **kwargs)
        q = plt.quiver(u.lon[::step], u.plev, 
                       u.isel(model=i)[::step],
                       v.isel(model=i)[::step],
                       units='width', pivot='mid')
        ax.set_title(u.model.values[i])
    # layout and save plot
    saveplot(plotname)

def saveplot(plotname):
    plt.subplots_adjust(wspace=0.04, hspace=0.05)
    plt.tight_layout()
    if plotname != None:
        plt.savefig('figs/' + plotname)
        print('saved to figs/' + plotname)
    plt.show()
