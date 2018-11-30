def to_pandas(Tgrid):
    """
    Parse the time grid of a Dataset and replace by a pandas time grid.
    """
    import pandas as pd
    from datetime import date, timedelta

    # get reference year from units
    words = Tgrid.units.split()
    ref_year = int(words[2][0:4])
    # get first time grid value
    if words[0] == 'months':
        first_time = Tgrid.values[0] - 0.5
        datetime = enso2date(first_time, ref_year)
    elif words[0] == 'days':
        days = Tgrid.values[0] - 15
        start = date(ref_year, 1, 1) 
        delta = timedelta(days)
        datetime = start + delta
    elif words[0] == 'hours':
        days = Tgrid.values[0] / 24
        start = date(ref_year, 1, 1)
        delta = timedelta(days)
        datetime = start + delta
    else:
        print('Unrecognized time grid')
        return
    return pd.date_range(datetime, periods=Tgrid.shape[0], freq='MS').shift(15, freq='D')

def to_enso(start_time,nt=1):
    """
    Parse the time grid of a Dataset and replace by a enso time grid.
    """
    import numpy as np
    # first get the reference year from start_time
    ryear,rmonth,rday = start_time[0:10].split('-')
    return (int(ryear)-1960)*12 + int(rmonth) - 0.5 + np.arange(0,nt)

def enso2date(T0,ryear=1960,leap=True):
    """
    Print the date corresponding to an enso-time (months since 1960). The reference year can be changed.
    """
    norm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    iy = ryear + int(T0/12)
    if T0 < 0:
        iy = iy - 1
    res = T0 - (iy - ryear)*12
    im = int(res) + 1
    if im == 13:
        im = 1
        iy = iy + 1
    if leap & (im == 2) &  (iy % 4 == 0 ):   
        id = 1 + int(29 * (res - int(res)))
    else:
        id = 1 + int(norm[im-1] * (res - int(res)))
    return str(iy)+'/'+str(im)+'/'+str(id)