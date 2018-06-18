import numpy as np

# load data from netcdf file
def load_netcdfdata(path, fname, var):
    from netCDF4 import Dataset
    file = Dataset(path + fname, 'r')
    if var in file.variables:   
       data = np.squeeze(np.array(file.variables[var]))
    else:
       print('WARNING in atmosphere.load_netcdfdata: field %s missing in %s!'%(var, fname))
    return data
    
def get_verticalintegral(data,lev,lat):
    # do vertical integral over pressure/g=mass
    # inputs: zonal mean data in (lev,lat) format, lev is pressure levels in Pa

    nlev=np.size(lev)
    nlat=np.size(lat)
 
    # make sure that layout of data is nlevxnlat
    if np.shape(data) != (nlev,nlat):
       data=np.transpose(data)   
       
    # make sure that level 0 is top of atmosphere
    # if not reorder vertical direction
    if lev[0]>lev[1]:
        #print('reordering in the vertical')
        lev =lev[::-1]
        data=data[::-1,:]
    
    # calculate level thickness    
    dlev=0*lev
    dlev[0]=0.5*(lev[0]+lev[1])
    for k in range(1,nlev-1):
        dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
    dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1]) 
    #print(dlev)
  
    # vertical integral of data
    out=np.zeros(nlat)+np.NaN
    for j in range(0,nlat):
        out[j]=1/9.81*np.sum(data[:,j]*dlev)
  
    return out        
    
def get_dxdy(x, y):
    ny=np.size(y)
    dy=0*y
    dy[0]=0.5*(y[0]+y[1])
    for k in range(1,ny-1):
        dy[k]=0.5*(y[k]+y[k+1])-0.5*(y[k-1]+y[k])
    dy[ny-1]=y[ny-1]-0.5*(y[ny-2]+y[ny-1])
    dxdy=0*y    
    for k in range(1,ny-2):    
        dxdy[k]=1/(2*dy[k])*(x[k+1]-x[k-1])
    return dxdy      

# calculate itcz position based on precip centroid between 30n and 30s
def get_itczposition(pr, lat, latboundary, dlat):
  area  = np.cos(lat*np.pi/180)  
  xi    = np.arange(-latboundary, latboundary, dlat)
  yi    = np.interp(xi, lat, pr)
  areai = np.interp(xi, lat, area)
  # area-integrated precip (up to constant factor)
  itcz = np.NaN
  for j in range(len(xi)):
     if sum(np.multiply(yi[0:j], areai[0:j])) >= 0.5*sum(np.multiply(yi, areai)): # find median latitude of area-integrated precip
        itcz = xi[j]
        break
  return itcz

# calculate meridional atmosphere energy transport from atmosphere energy budget
def get_atmenergytransport(atm,lat):
  # lat must be from South Pole to North Pole
  nlat=len(lat)
  
  # construct area of latband
  # sum of area must be Earth surface, so normalize
  area=np.cos(lat*np.pi/180) 
  area=area*4*np.pi*np.power(6371e3,2)/sum(area)

  # set output to NaN  
  tra = 0*atm+np.NaN
  
  # integrate from South Pole to North Pole
  s2n   =0*atm+np.NaN
  s2n[0]=atm[0]*area[0]
  for j in range(1,nlat):
     s2n[j]=atm[j]*area[j]+s2n[j-1]
  # integrate from North Pole to South Pole
  n2s   =0*atm+np.NaN
  n2s[nlat-1]=-atm[nlat-1]*area[nlat-1]
  for j in range(nlat-2,-1,-1):
     n2s[j]=-atm[j]*area[j]+n2s[j+1]
  
  # transport is average of the two integration directions
  # and convert to units of PW
  tra=0.5*(n2s+s2n)
  tra=1e-15*tra
 
  return tra
  
# get mean over between two pressure boundaries p1 and p2
def get_mean_over_plevels(data,lev,p1,p2):
   # lev has to be in units of Pa
   if max(lev)<0.9e4:
       print('-----------------------------------------------------------------')
       print('error in get_mean_over_plevels: it seems that lev is not in Pa but hPa')
       print('-----------------------------------------------------------------')
 
   nlev=np.size(lev)
   # make sure that layout of data is nlevx1
   data=np.reshape(data,(nlev,1))
   # make sure that layout of lev is nlevx1
   lev=np.reshape(lev,(nlev,1))
  
   #test if first level is top of atmosphere, if not reorder the lev and in
   if lev[0]>lev[1]:
       lev  = lev[::-1]
       data = data[::-1]

   # calculate pressure thickness of levels, level 1 is top of atmosphere
   dlev=0*lev;
   dlev[0]=0.5*(lev[0]+lev[1])
   for k in range(1,nlev-1):
       dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
   dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1])
   
   # make vertical mean between p1 and p2 (for example, 300e2 and 900e2 Pa)
   indlev=np.where((lev>=p1) & (lev<=p2))
   mean=sum(data[indlev]*dlev[indlev])/sum(dlev[indlev])

   return mean    
   
# get mean over between two pressure boundaries p1 and p2
def get_nanmean_over_plevels(data,lev,p1,p2):
   # lev has to be in units of Pa
   if max(lev)<0.9e4:
       print('-----------------------------------------------------------------')
       print('error in get_mean_over_plevels: it seems lev in not in Pa but hPa')
       print('-----------------------------------------------------------------')
 
   nlev=np.size(lev)
   # make sure that layout of data is nlevx1
   data=np.reshape(data,(nlev,1))
   # make sure that layout of lev is nlevx1
   lev=np.reshape(lev,(nlev,1))
  
   #test if first level is top of atmosphere, if not reorder the lev and in
   if lev[0]>lev[1]:
       lev  = lev[::-1]
       data = data[::-1]

   # calculate pressure thickness of levels, level 1 is top of atmosphere
   dlev=0*lev;
   dlev[0]=0.5*(lev[0]+lev[1])
   for k in range(1,nlev-1):
       dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
   dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1]);
   
   # if data is nan, then set dlev to nan as well
   for k in range(0,nlev):
       if np.isnan(data[k]): dlev[k]=np.nan    

   # make vertical mean between p1 and p2 (for example, 300e2 and 900e2 Pa)
   indlev=np.where((lev>=p1) & (lev<=p2))
   mean=np.nansum(data[indlev]*dlev[indlev])/np.nansum(dlev[indlev])

   return mean   

# get mean over a pressure and latitude box 
def get_mean_over_plevellatituderegion(data,lev,lev1,lev2,lat,lat1,lat2,area):
    nlev=np.size(lev)
    nlat=np.size(lat)
 
    # make sure that layout of data is nlevxnlat
    if np.shape(data) != (nlev,nlat):
        data=np.transpose(data) 

    # make sure that layout of lev is nlevx1
    lev=np.reshape(lev,(nlev,))        
    # make sure that layout of lat nlatx1
    lat=np.reshape(lat,(nlat,))   
    # make sure that layout of area is nlatx1
    area=np.reshape(area,(nlat,))
        
    # test if first level is top of atmosphere, if not reorder lev and data
    if lev[0]>lev[1]:
        lev=lev[::-1]
        data=data[::-1,:]
     
    # calculate pressure thickness of levels, level 1 is top of atmosphere
    dlev=0*lev;
    dlev[0]=0.5*(lev[0]+lev[1])
    for k in range(1,nlev-1):
        dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
    dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1]);
    #print(lev)
    #print(dlev)
    
    # find indices of latitudes over which we average
    indlat=np.squeeze(np.where((lat>=lat1) & (lat<=lat2)))
    # find indices of levels over which we average (between p1 and p2, i.e., 300e2 and 900e2 Pa)
    indlev=np.squeeze(np.where((lev>=lev1) & (lev<=lev2)))
    
    # do the mean: first over lev, then over lat    
    aux=np.zeros(lat.shape)
    for j in range(0,nlat):
        aux[j]=np.sum(data[indlev,j]*dlev[indlev])/np.sum(dlev[indlev])
    mean=np.sum(aux[indlat]*area[indlat])/np.sum(area[indlat])
 
    return mean    
    
    
# get mean over a pressure and latitude box 
def get_nanmean_over_plevellatituderegion(data,lev,lev1,lev2,lat,lat1,lat2,area):
    nlev=np.size(lev)
    nlat=np.size(lat)
 
    # make sure that layout of data is nlevxnlat
    if np.shape(data) != (nlev,nlat):
        data=np.transpose(data) 

    # make sure that layout of lev is nlevx1
    lev=np.reshape(lev,(nlev,))        
    # make sure that layout of lat nlatx1
    lat=np.reshape(lat,(nlat,))   
    # make sure that layout of area is nlatx1
    area=np.reshape(area,(nlat,))
        
    # test if first level is top of atmosphere, if not reorder lev and data
    if lev[0]>lev[1]:
        lev=lev[::-1]
        data=data[::-1,:]
     
    # calculate pressure thickness of levels, level 1 is top of atmosphere
    dlev=0*lev;
    dlev[0]=0.5*(lev[0]+lev[1])
    for k in range(1,nlev-1):
        dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
    dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1]);
    #print(lev)
    #print(dlev)
    
    # find indices of latitudes over which we average
    indlat=np.squeeze(np.where((lat>=lat1) & (lat<=lat2)))
    # find indices of levels over which we average (between p1 and p2, i.e., 300e2 and 900e2 Pa)
    indlev=np.squeeze(np.where((lev>=lev1) & (lev<=lev2)))
    
    # do weighted mean: do not use nan values
    aux     = np.float(0.0)
    weights = np.float(0.0)
    for j in indlat:
        for k in indlev:
            if data[k,j] != np.nan:
                print(data[k,j])
                aux     = aux + data[k,j]*dlev[k]*area[j]
                weights = weights + dlev[k]*area[j]
    mean = np.nan
    if  weights > 0.0: mean = aux/weights
    print(weights)
 
    return mean        
     
  
# get global mean based on 1d-latitude data
def get_globalmean(data,lat,area):
   nlat=np.size(lat)
   # make sure that layout of data and area is nlatx1
   data=np.reshape(data,(nlat,1))
   area=np.reshape(area,(nlat,1))
   # do the average
   mean=np.sum(data*area)/sum(area)
   return mean
   
   
# get mean between two latitudes lat1 and lat2
def get_mean_over_latituderegion(data,lat,lat1,lat2,area):
   nlat=np.size(lat)
   # make sure that layout of data and area is nlatx1
   data=np.reshape(data,(nlat,1))
   area=np.reshape(area,(nlat,1))
   # find indices of latitudes over which we average
   indlat=np.where((lat>lat1) & (lat<lat2))
   # do the average
   mean=np.sum(data[indlat]*area[indlat])/sum(area[indlat])
   return mean       
    
    
# get symmetric and asymmetric component wrt equator
def get_sym_and_asym_component(data,lat):
   nlat = len(lat)
   sym  = np.zeros(nlat) + np.NaN
   asym = sym
  
   #nlat is even, so there is no latitude at the equator
   if nlat%2==0:
      for j in range(0,int(nlat/2)):
         sym[j]=0.5*(data[j]+data[nlat-j-1])
         sym[nlat-j-1]=sym[j]
   #nlat is odd, so there is no latitude at the equator      
   if nlat%2!=0:
      for j in range(0,int((nlat-1)/2)):
         sym[j]=0.5*(data[j]+data[nlat-j-1])
         sym[nlat-j-1]=sym[j]
      sym[int((nlat-1)/2)]=data[int((nlat-1)/2)]
   # asymmetric component
   asym=data-sym
  
   return sym, asym
   
# get average over northern and southern hemisphere
def get_hemispheric_mean(data,lat):
   area      = np.cos(lat*np.pi/180)
   indlat_nh = np.where(lat>0)
   indlat_sh = np.where(lat<0)
   nh = np.nansum(data[indlat_nh]*area[indlat_nh]) / np.nansum(area[indlat_nh])
   sh = np.nansum(data[indlat_sh]*area[indlat_sh]) / np.nansum(area[indlat_sh])
  
   return nh, sh

# get difference between nh and sh assuming data is on lat grid
def get_hemispheric_diff_1d(data, lat):
   nlat = lat.size
   area = np.zeros( nlat )
   for j in range(0, nlat):
       area[j] = np.cos( lat[j]*np.pi/180 )
   indlat_nh = np.where(lat>0)
   indlat_sh = np.where(lat<0)
   nh = np.nansum( data[indlat_nh] * area[indlat_nh] ) / \
        np.nansum( area[indlat_nh] )
   sh = np.nansum( data[indlat_sh] * area[indlat_sh] ) / \
        np.nansum( area[indlat_sh] )        
   return nh - sh 

   
# get difference between nh and sh assuming data is on latxlon grid
def get_hemispheric_diff_2d(data, lat):
   nlat = lat.size
   nlon = data[0, :].size
   area = np.zeros( (nlat, nlon) )
   for j in range(0, nlat):
       area[j, :] = np.cos( lat[j]*np.pi/180 )
   indlat_nh = np.where(lat>0)
   indlat_sh = np.where(lat<0)
   nh = np.nansum(np.ravel(data[indlat_nh, :])*np.ravel(area[indlat_nh, :])) / \
        np.nansum(np.ravel(area[indlat_nh, :]))
   sh = np.nansum(np.ravel(data[indlat_sh, :])*np.ravel(area[indlat_sh, :])) / \
        np.nansum(np.ravel(area[indlat_sh, :]))        
   return nh - sh 

# get difference between nh and sh assuming data is on ndim1xlatxlon grid
def get_hemispheric_diff_3d(data, lat):
   ndim1= data[:, 0, 0].size   
   nlat = lat.size
   nlon = data[0, 0, :].size
   area = np.zeros( (nlat, nlon) )
   for j in range(0, nlat):
       area[j, :] = np.cos( lat[j]*np.pi/180 )
   indlat_nh = np.where(lat>0)
   indlat_sh = np.where(lat<0)
   nh = np.zeros( (ndim1) ) + np.nan
   sh = np.zeros( (ndim1) ) + np.nan
   for d1 in range(0, ndim1):
       nh[d1] = np.nansum( np.ravel(data[d1, indlat_nh, :]) * np.ravel(area[indlat_nh, :]) ) / \
                np.nansum( np.ravel(area[indlat_nh, :]) )
       sh[d1] = np.nansum( np.ravel(data[d1, indlat_sh, :]) * np.ravel(area[indlat_sh, :]) ) / \
                np.nansum( np.ravel(area[indlat_sh, :]) )
   return nh - sh    

# get difference between nh and sh assuming data is on ndim1xndim2xlatxlon grid
def get_hemispheric_diff_4d(data, lat):
   ndim1= data[:, 0, 0, 0].size   
   ndim2= data[0, :, 0, 0].size   
   nlat = lat.size
   nlon = data[0, 0, 0, :].size
   area = np.zeros( (nlat, nlon) )
   for j in range(0, nlat):
       area[j, :] = np.cos( lat[j]*np.pi/180 )
   indlat_nh = np.where(lat>0)
   indlat_sh = np.where(lat<0)
   nh = np.zeros( (ndim1, ndim2) ) + np.nan
   sh = np.zeros( (ndim1, ndim2) ) + np.nan
   for d1 in range(0, ndim1):
       for d2 in range(0, ndim2):   
           nh[d1, d2] = np.nansum( np.ravel(data[d1, d2, indlat_nh, :]) * np.ravel(area[indlat_nh, :]) ) / \
                np.nansum( np.ravel(area[indlat_nh, :]) )
           sh[d1, d2] = np.nansum( np.ravel(data[d1, d2, indlat_sh, :]) * np.ravel(area[indlat_sh, :]) ) / \
                np.nansum( np.ravel(area[indlat_sh, :]) )
   return nh - sh


# hemispherid difference
def get_hemispheric_diff(data, lat):
   if data.ndim == 1: diff = get_hemispheric_diff_1d(data, lat)
   if data.ndim == 2: diff = get_hemispheric_diff_2d(data, lat)
   if data.ndim == 3: diff = get_hemispheric_diff_3d(data, lat)   
   if data.ndim == 4: diff = get_hemispheric_diff_4d(data, lat)    
   return diff


# get tropical between nh and sh assuming data is on lat grid
def get_tropical_diff_1d(data, lat):
   nlat = lat.size
   area = np.zeros( nlat )
   for j in range(0, nlat):
       area[j] = np.cos( lat[j]*np.pi/180 )
   indlat_nh = np.where( (lat>0) & (lat<= 30) )
   indlat_sh = np.where( (lat<0) & (lat>=-30) )
   nh = np.nansum( data[indlat_nh] * area[indlat_nh] ) / \
        np.nansum( area[indlat_nh] )
   sh = np.nansum( data[indlat_sh] * area[indlat_sh] ) / \
        np.nansum( area[indlat_sh] )        
   return nh - sh 

   
def func_fit_quadratic(x,p0,p1,p2):
    return p0+p1*x+p2*x**2
    
def get_eddyjetlat(u,lat):
    # calculate latitude of eddy-driven jet

    import scipy.optimize as spopt  
    
    # make sure that lat is ordered from SP to NP; otherwise
    # np.arange does not work to create latint
    if lat[0]>lat[1]:
        lat=lat[::-1]
        u  =u[::-1]
  
    if any(np.isnan(u)):
        jetlat_nh=np.NaN
        jetlat_sh=np.NaN
    else:
       #Northern hemisphere
       indlat_nh = np.squeeze(np.array(np.where((lat>25) & (lat<70))))
       maxlat = np.argmax(u[indlat_nh]) + indlat_nh[0]
  
       # do quadratic fit around the maximum
       latint=np.arange(lat[maxlat-2],lat[maxlat+2],0.01)
       uint  =np.interp(latint,lat[indlat_nh],u[indlat_nh])
       p,_   =spopt.curve_fit(func_fit_quadratic,latint,uint)
       ufit  =func_fit_quadratic(latint,p[0],p[1],p[2])
       jetlat_nh=latint[np.argmax(ufit)]
     
       #Southern hemisphere
       indlat_sh = np.squeeze(np.array(np.where((lat<-25) & (lat>-70))))
       maxlat = np.argmax(u[indlat_sh]) + indlat_sh[0]
  
       # do quadratic fit around the maximum
       latint=np.arange(lat[maxlat-2],lat[maxlat+2],0.01)
       uint  =np.interp(latint,lat[indlat_sh],u[indlat_sh])
       p,_   =spopt.curve_fit(func_fit_quadratic,latint,uint)
       ufit  =func_fit_quadratic(latint,p[0],p[1],p[2])
       jetlat_sh=latint[np.argmax(ufit)]
     
    return jetlat_nh, jetlat_sh 
    
    
def get_eddyjetmax(u,lat):
    # calculate strength of eddy-driven jet

    import scipy.optimize as spopt  
   
    # make sure that lat is ordered from SP to NP; otherwise
    # np.arange does not work to create latint
    if lat[0]>lat[1]:
        lat=lat[::-1]
        u  =u[::-1]
    
    if any(np.isnan(u)):
        jetmax_nh=np.NaN
        jetmax_sh=np.NaN
    else:
       #Northern hemisphere
       indlat_nh=np.where(lat>10)
       lat_nh=lat[indlat_nh]  
       u_nh=u[indlat_nh]
       maxlat=np.argmax(u_nh)
  
       # do quadratic fit around the maximum
       latint=np.arange(lat_nh[maxlat-5],lat_nh[maxlat+5],0.01)
       uint  =np.interp(latint,lat_nh,u_nh)
     
       p,_   =spopt.curve_fit(func_fit_quadratic,latint,uint)
       ufit  =func_fit_quadratic(latint,p[0],p[1],p[2])
       jetmax_nh=np.max(ufit)
     
       #Southern hemisphere
       indlat_sh=np.where(lat<-10)
       lat_sh=lat[indlat_sh]  
       u_sh=u[indlat_sh]
       maxlat=np.argmax(u_sh)
  
       # do quadratic fit around the maximum
       latint=np.arange(lat_sh[maxlat-5],lat_sh[maxlat+5],0.01)
       uint  =np.interp(latint,lat_sh,u_sh)
       p,_   =spopt.curve_fit(func_fit_quadratic,latint,uint)
       ufit  =func_fit_quadratic(latint,p[0],p[1],p[2])
       jetmax_sh=np.max(ufit)

    return jetmax_nh, jetmax_sh     
    

def get_massstreamfunction(v, lev, lat):
    #calculate mass stream function in units of 10^9 kg/s
    #inputs: zonal mean meridional wind in m/s, lev is pressure levels in Pa
    #with first level corresponding to top-of-atmosphere

    nlev = np.size(lev)
    nlat = np.size(lat)
 
    # make sure that layout of v is nlevxnlat
    if np.shape(v) != (nlev, nlat):
        v = np.transpose(v)        
        #print('error inget_masstreamfunction: layout of v-wind must be nlevxnlat')
        #return

    # make sure that layout of lev is nlevx1
    lev=np.reshape(lev,(nlev,1))        
    # make sure that layout of lat nlatx1
    lat=np.reshape(lat,(nlat,1))    
    
    # test if first level is top of atmosphere, if not reorder lev and v
    do_flipud=0    
    if lev[0]>lev[1]:
        lev=lev[::-1]
        v=v[::-1,:]
        do_flipud=1
        #print('flipping vertical levels')
    
    # calculate pressure thickness of levels, level 0 is top of atmosphere
    dlev=0*lev
    dlev[0]=0.5*(lev[0]+lev[1])
    for k in range(1,nlev-1):
        dlev[k]=0.5*(lev[k]+lev[k+1])-0.5*(lev[k-1]+lev[k])
    dlev[nlev-1]=lev[nlev-1]-0.5*(lev[nlev-2]+lev[nlev-1])
   
    # do the integral to get mass stream function
    msf=np.zeros((nlev,nlat)) + np.NaN    
    factor=2*np.pi*6371e3/9.81  #2*pi*rearth/g
    for j in range(0,nlat):
        # top layer
        msf[0,j]=factor*np.cos(lat[j]*np.pi/180.0)*v[0,j]*dlev[0]
        # now integrate downward        
        for k in range(1,nlev):
            msf[k,j]=factor*np.cos(lat[j]*np.pi/180.0)*v[k,j]*dlev[k]+msf[k-1,j]            
            
    # if we flipped before we need to flip back for msf here
    if do_flipud==1:
        msf=msf[::-1,:]
         
    #convert to units of 10^9 kg/s
    msf=msf/1e9
    
    return msf
    
    
def get_hcedge(msf,lev,lat):
    # find Hadley cell edge as subtropical latitude where mass stream function
    # at 500 hPa changes sign

    nlev=np.size(lev)
    nlat=np.size(lat)
 
    # make sure that layout of msf is nlevxnlat
    if np.shape(msf) != (nlev,nlat):
        msf=np.transpose(msf)   
        
    # make sure that layout of lev is nlevx0
    lev=np.reshape(lev,(nlev,))        
    # make sure that layout of lat nlatx0
    lat=np.reshape(lat,(nlat,))    
    
    # make sure that lat is ordered from SP to NP; otherwise
    # np.arange does not work to create latint
    if lat[0]>lat[1]:
        lat=lat[::-1]
        msf=msf[:,::-1]
  
    # find msf at 500hPa: note that levels are in Pa
    ilev  = (np.abs(lev-500e2)).argmin()
    msf500=msf[ilev,:]
    
    # Northern hemisphere
    # interpolate to find zero crossing
    latint=np.arange(20,40,0.01)
    msf500_int=np.interp(latint,lat,msf500)
    nhedge=latint[np.argmin(np.abs(msf500_int))]

    # Southern hemisphere
    # interpolate to find zero crossing
    latint=np.arange(-40,-20,0.01)
    msf500_int=np.interp(latint,lat,msf500)
    shedge=latint[np.argmin(np.abs(msf500_int))]

    return nhedge, shedge
 
    
def get_pottemp(t, lev, lat):
    # calculate potential temperature
    # inputs: zonal mean temperature in K, lev is pressure levels in Pa

    nlev=np.size(lev)
    nlat=np.size(lat)
 
    # make sure that layout of t is nlevxnlat
    do_transpose=0    
    if np.shape(t) != (nlev,nlat):
       t=np.transpose(t)   
       do_transpose=1
  
    # calculate potential temperature
    theta=0*t+np.NaN
    for k in range(0,nlev):
        theta[k,:]=t[k,:]*np.power(1e5/lev[k],0.286)   
       
    # if we transposed before we need to transpose back here
    if do_transpose==1:
       theta=np.transpose(theta)  
       
    return theta    
        
    
def get_bruntvaisala(t , z, lat, lev):
    # calculate brunt vaisala frequency
    # inputs: zonal mean temperature in K, geopotential height in m
    #         lat is latitude, lev is pressure in Pa
    # output: N = sqrt(g/theta * dtheta/dz)    
    
    g = 9.81

    nlev = np.size(lev)
    nlat = np.size(lat)
 
    # make sure that layout of t (and z) is nlevxnlat
    do_transpose = 0    
    if np.shape(t) != (nlev, nlat):
       t = np.transpose(t)   
       z = np.transpose(z)
       do_transpose = 1
  
    # calculate potential temperature
    theta = 0*np.copy(t) + np.nan
    theta = get_pottemp(t , lev, lat)
    
    # calculate dtheta/dz
    dthetadz = 0*np.copy(theta) + np.nan
    for j in range(0, nlat):
        dthetadz[:, j] = get_dxdy(theta[:, j], z[:, j])
    
    # brunt vaisala frequency
    N = 0*np.copy(t) + np.nan
    N = np.power(g/theta * dthetadz, 0.5)    
       
    # if we transposed before we need to transpose back here
    if do_transpose==1:
       N = np.transpose(N)  
       
    return N
    
    
def usstandardatmosphere1976():
    # taken from http://www.digitaldutch.com/atmoscalc/tableatmosphere.htm

    z=np.array([0.00000,500.000,1000.00,1500.00,2000.00,2500.00,3000.00,3500.00,
                4000.00,4500.00,5000.00,5500.00,6000.00,6500.00,7000.00,7500.00,
                8000.00,8500.00,9000.00,9500.00,10000.0,10500.0,11000.0,11500.0,
                12000.0,12500.0,13000.0,13500.0,14000.0,14500.0,15000.0,15500.0,
                16000.0,16500.0,17000.0,17500.0,18000.0,18500.0,19000.0,19500.0,
                20000.0,20500.0,21000.0,21500.0,22000.0,22500.0,23000.0,23500.0,
                24000.0,24500.0,25000.0,25500.0,26000.0,26500.0,27000.0,27500.0,
                28000.0,28500.0,29000.0,29500.0,30000.0,30500.0,31000.0,31500.0,
                32000.0,32500.0,33000.0,33500.0,34000.0,34500.0,35000.0,35500.0,
                36000.0,36500.0,37000.0,37500.0,38000.0,38500.0,39000.0,39500.0,
                40000.0])    
    
    temp=np.array([288.150,284.900,281.650,278.400,275.150,271.900,268.650,265.400,
                   262.150,258.900,255.650,252.400,249.150,245.900,242.650,239.400,
                   236.150,232.900,229.650,226.400,223.150,219.900,216.650,216.650,
                   216.650,216.650,216.650,216.650,216.650,216.650,216.650,216.650,
                   216.650,216.650,216.650,216.650,216.650,216.650,216.650,216.650,
                   216.650,217.150,217.650,218.150,218.650,219.150,219.650,220.150,
                   220.650,221.150,221.650,222.150,222.650,223.150,223.650,224.150,
                   224.650,225.150,225.650,226.150,226.650,227.150,227.650,228.150,
                   228.650,230.050,231.450,232.850,234.250,235.650,237.050,238.450,
                   239.850,241.250,242.650,244.050,245.450,246.850,248.250,249.650,
                   251.050])

    press=np.array([101325,95460.8,89874.6,84556.0,79495.2,74682.5,70108.5,65764.1,
                    61640.2,57728.3,54019.9,50506.8,47181.0,44034.8,41060.7,38251.4,
                    35599.8,33099.0,30742.5,28523.6,26436.3,24474.4,22632.1,20916.2,
                    19330.4,17864.8,16510.4,15258.7,14101.8,13032.7,12044.6,11131.4,
                    10287.5,9507.50,8786.68,8120.51,7504.84,6935.86,6410.01,5924.03,
                    5474.89,5060.26,4677.89,4325.18,3999.79,3699.54,3422.43,3166.65,
                    2930.49,2712.42,2511.02,2324.98,2153.09,1994.26,1847.46,1711.75,
                    1586.29,1470.27,1362.96,1263.70,1171.87,1086.88,1008.23,935.425,
                    868.019,805.719,748.228,695.150,646.122,600.814,558.924,520.175,
                    484.317,451.118,420.367,391.872,365.455,340.954,318.220,297.118,
                    277.522])
    
    return z,temp,press