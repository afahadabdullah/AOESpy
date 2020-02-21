#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:29:53 2018

@author: afahad (afahad@gmu.edu)
"""
## Python AOES library ! ##

# Loading necessary Libraries #
from numpy import *
from scipy import stats
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset as nc
import  mpl_toolkits.basemap
d=plt.show
from mpl_toolkits.basemap import shiftgrid
from tqdm.autonotebook import tqdm

#functions

## d(): displays plot based on matplotlib.pyplot

## shiftgrid:
	# dataout, newlon= shiftgrid(lon0, datain, lonsin, start=True, cyclic=360.0)

	#inputs: 	lon0=starting longitude for shifted grid (ending longitude if start=False)
	#			datain= original data with longitude the right-most dimension.
	#			lonsin= original longitudes
	#			start= if True, lon0 represents the starting longitude of the new grid. if False, lon0 is the ending longitude. Default True.
	#			cyclic=	width of periodic domain (default 360)

	#outputs:	dataout= shifted input data
	#			newlon= new shifted longitude array


## an output default figure size 

def f():
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=.05, bottom=.05, right=.95, top=.95)

## Read netcdf data
	#data=rnc(var,file)
	
	#inputs:	var= string variable name that needs to be read 
	#			file=string file path

	#outputs:	data= data read from the file

def rnc(var,file):
    f = nc(file)
    v = f.variables[var][:]
    f.close()
    return v

## linear Trend
	 #vart, varp=ltrend(var,lon,lat,time,sig=False)

	#inputs:	var= variable as 3D [time,lat,lon] or 2D [time,lat*lon]
				# lon=lon array
				# lat=lat array
				# time=time array
				# sig= alpha significance value (e.g. 0.05, 0.1), if given the output data
				#		will have nan values in insignificant points (default False)

	#outputs:	#vart= linear trend of the variable along time dimension
				#varp= P value of the trend		

def ltrend(var,lon,lat,time,sig=False):
    nlon=len(lon)
    nlat=len(lat)
    nt=len(time)
    vart=zeros(nlat*nlon)
    varp=zeros(nlat*nlon)
    
    if len(var.shape)== 3:        
        var=reshape(var,(nt,nlat*nlon)) 
        print('l_trend: assuming variable as 3D [time,lat,lon]')
        for i in tqdm(range(nlat*nlon)):
            v=var[:,i]  
            vart[i], intercept, r_value, varp[i], std_err=stats.linregress(time,v)
            
        vart=reshape(vart,(nlat,nlon))
        varp=reshape(varp,(nlat,nlon))
        #return (vart,varp)
        
    elif len(var.shape)== 2:
        print('l_trend: assuming variable as 2D [time,lat*lon]')
        for i in tqdm(range(nlat*nlon)):
            v=var[:,i]  
            #vart[i]=stats.linregress(time,v).slope
            vart[i], intercept, r_value, varp[i], std_err=stats.linregress(time,v)
    
        vart=reshape(vart,(nlat,nlon))
        varp=reshape(varp,(nlat,nlon))
        #return vart
        
    else:
        raise ValueError('Variable shape is not 2D or 3D. plese instert variable in this format var[time,lat,lon] or var[time,lon*lat]')
    
    if sig==False:
        return (vart, varp) 
    else:
        for i in range(nlat):
            for j in range (nlon):
                if varp[i,j]>sig:
                  vart[i,j]=nan
        return (vart, varp)
           

## mapping functing
	 #plot(var,lon,lat,title='',clf=[],cl=[], cmap='coolwarm',lon1=-180,lon2=180,lat1=-90,lat2=90,bar=1,p=1,m=1)

	#inputs: 	var= 2D variable that will be plotted
	#			lon=lon
	#			lat=lat
	#			title='title'
	#			clf= array of filled contoured levels
	#			cl= array of contoured levels (optional)
	#			cmap= string of name of colorbar (default coolwarm, for list: matplotlib colorbars)
	#			lon1= start of lon (default -180)
	#			lon2= end of lon (default 180)
	#			lat1= start of lat (default -90)
	#			lat2= end of lat (default 90)
	#			bar= 1 (default) to plot colorbar; or 0 doesn't plot colorbar in figure
	#			p= 1 (default) plots parallel line, or 0 doesn't plot 
	#			m= 1 (default) plots meridioinal line, or 0 doesn't plot 


def plot(var,lon,lat,title='',clf=[],cl=[], cmap='coolwarm',lon1=-180,lon2=180,lat1=-90,lat2=90,bar=1,p=1,m=1,lmask=0, alpha=.95):

    if lon1== -180:
        if lon2==180:
            if nanmax(lon)>181:
                lon1=0
                lon2=360

    if lon1== -180:
        if lon2==180:
            if lat1==-90:
                if lat2==90:
                    lon1=nanmin(lon)
                    lon2=nanmax(lon)
                    lat1=nanmin(lat)
                    lat2=nanmax(lat)


    map = Basemap(projection='cyl',llcrnrlat=lat1,urcrnrlat=lat2,\
                llcrnrlon=lon1,urcrnrlon=lon2,resolution='l')

    map.drawcoastlines(linewidth=.6,)
    parallels = arange(lat1,lat2+1, (lat2-lat1)//4) 
    meridians = arange(lon1,lon2,(lon2-lon1)//4)

    if lmask==1:
        map.fillcontinents(color='white', lake_color='white', ax=None, zorder=None, alpha=None)
    if m==1:
        map.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.05,fontsize=8,dashes=[1, 1000])
    if m==0:
        map.drawmeridians(meridians,linewidth=0.05,fontsize=8,dashes=[1, 1000])
    if p==1:
        map.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.05,fontsize=8,dashes=[1, 1000])
    if p==0:
        map.drawparallels(parallels,linewidth=0.05,fontsize=8,dashes=[1, 1000])

    # map.drawparallels(parallels,plabels,linewidth=0.02,fontsize=8)
    # map.drawmeridians(meridians,mlabels,linewidth=0.02,fontsize=8)
    lons,lats= meshgrid(lon,lat)
    x,y = map(lons,lats)

    if len(cl)==1:

        if len(clf)>1:
            csf = map.contourf(x,y,var,clf,extend='both',cmap=cmap, alpha=.95)
            if bar==1:
                cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="12%")
            cs = map.contour(x,y,var,cl,colors='k',linewidths=0.3, alpha=.95)
            #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
            plt.title(title,fontsize=9) 
        else:
            csf = map.contourf(x,y,var,extend='both',cmap=cmap, alpha=.95)
            #cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="9%")
            if bar==1:
                cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="12%")
            cs = map.contour(x,y,var,colors='k',linewidths=0.3, alpha=.95)
            #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
            plt.title(title,fontsize=9)
    else:

        if len(clf)>1:
            csf = map.contourf(x,y,var,clf,extend='both',cmap=cmap, alpha=.95)
            if bar==1:
                cb = map.colorbar(csf,"bottom", extend='both',size="5%", pad="15%")
            #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
            plt.title(title,fontsize=9) 
        else:
            csf = map.contourf(x,y,var,extend='both',cmap=cmap, alpha=.95)
            #cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="9%")
            if bar==1:
                cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="12%")
            #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
            plt.title(title,fontsize=9)
    return(x,y,map)




## 3D seasonal decompose from monthly time dinemsions to annual, DJF, MAM, JJA, SON

	 #ann, djf,mam,jja,son= season(data,lon,lat,time)

	#inputs:	data=3D data[time lat lon]
	#			lon=lon
	#			lat=lat
	#			time=time

	#outputs:	ann= annual mean
	#			djf= DJF mean
	#			mam= MAM mean
	#			jja= JJA mean
	#			son= SON mean

def season(data,lon,lat,time):
    nlon=len(lon)
    nlat=len(lat)
    nt=len(time)
    mo=12
    yr=nt//mo

    data=reshape(data,(yr,mo,nlat,nlon))

    ann=nanmean(data,1)

    d=data[:-1,11:12,:,:]
    j=data[1:,0:1,:,:]
    f=data[1:,1:2,:,:]
    #jf=data[:,0:2,:,:]
    #djf=concatenate((d,jf),axis=1)
    d=squeeze(nanmean(d,1))
    j=squeeze(nanmean(j,1))
    f=squeeze(nanmean(f,1))
    #jf=squeeze(nanmean(jf,1))

    djf=(d+j+f)/3

    mam=squeeze(nanmean(data[1:,2:5,:,:],1))
    jja=squeeze(nanmean(data[1:,5:8,:,:],1))
    son=squeeze(nanmean(data[1:,8:11,:,:],1))

    return (ann, djf,mam,jja,son)

## 1D seasonal decompose from monthly time dinemsions to annual, DJF, MAM, JJA, SON

	#ann, djf,mam,jja,son= season(data,time)

	#inputs:	data=1D data[time]
	#			time=time

	#outputs:	ann= annual mean
	#			djf= DJF mean
	#			mam= MAM mean
	#			jja= JJA mean
	#			son= SON mean

def season1d(data,time):
    nt=len(time)
    mo=12
    yr=nt//mo

    data=reshape(data,(yr,mo))

    ann=nanmean(data,1)
    d=data[:-1,11:12]
    j=data[1:,0:1]
    f=data[1:,1:2]
    #jf=data[:,0:2]
    #djf=concatenate((d,jf),axis=1)

    #djf=squeeze(nanmean(djf,1))

    d=squeeze(nanmean(d,1))
    j=squeeze(nanmean(j,1))
    f=squeeze(nanmean(f,1))
    #jf=squeeze(nanmean(jf,1))

    djf=(d+j+f)/3
    mam=squeeze(nanmean(data[1:,2:5],1))
    jja=squeeze(nanmean(data[1:,5:8],1))
    son=squeeze(nanmean(data[1:,8:11],1))

    return (ann, djf,mam,jja,son)


## interpolates data in desired grid
	
	#data_interp=interp(var, lon, lat, new_lons, new_lats,time=arange(1))

	#inputs:	var= input variable 2D or 3D (includes time dimension)
	#			lon= lon of the variable
	#			lat= lat of the variable
	#			new_lons= new lon that grid needs to be shifted to
	#			new_lats= new lat that grid needs to be shifted to
	#			time= (optional)

	#outputs:	data_interp= intrepolated data to new grids

def interp(var, lon, lat, new_lons, new_lats,time=arange(1)):
    nlon=len(new_lons)
    nlat=len(new_lats)

    new_lons, new_lats=meshgrid(new_lons, new_lats)

    if len(time)==1:
        data_interp=zeros((nlat,nlon))

        data=var[:,:]
        data_interp[:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)

    else:
        nt=len(time)
        data_interp=zeros((nt,nlat,nlon))
        for i in tqdm(range(nt)):
            data=squeeze(var[i,:,:])
            data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)
                
    return data_interp



## write variables in netcdf output. This function can write upto 2 variables in one file and required dimensions

     #wnc(x,y,data_out1,var1='data1',data_out2=array([1]),var2='data2',t=array([1]),e=array([1]),file='output')

    #inputs:    x=lon
    #           y=lat
    #           data_out1=first variable to write in file
    #           var1='data1' ; first varible name assigned in the file
    #           data_out2=second variable to write in file; (optional)
    #           var2='data2'; first varible name assigned in the file (if second variable is given to write)
    #           t= time dimension array
    #           e= ensemble dimension array (can  be used as vertical level)
    #           file= string of output file name (dont need to add .nc)


def wnc(x,y,data_out1,var1='data1',data_out2=array([1]),var2='data2',t=array([1]),e=array([1]),file='output'):

    nx = len(x); ny = len(y)
    if len(t)==1:
        nt=1
    else:
        nt=len(t)
    if len(e)==1:
        ne=1
    else:
        ne=len(e)

    out=file+'.nc'
    # open a new netCDF file for writing.
    ncfile = nc(out,'w') 

    ncfile.createDimension('lon',nx)
    ncfile.createDimension('lat',ny)

    if len(t)>1:
        ncfile.createDimension('time',nt)
    if len(e)>1:
        ncfile.createDimension('ens',ne)
    # create the variable (4 byte integer in this case)
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.

    # write data to variable.
    lon = ncfile.createVariable('lon',dtype('float').char,('lon'))
    lon[:] = x
    lon.units='degrees East'
    lon.long_name = 'Longitude'

    lat = ncfile.createVariable('lat',dtype('float').char,('lat'))
    # write data to variable.
    lat[:] = y
    lat.units='degrees North'
    lat.long_name = 'Latitude'

    if len(t)>1:
        time = ncfile.createVariable('time',dtype('float').char,('time'))
        # write data to variable.
        time[:] = t
        time.units='months since 1979-01-01 00:00'

    if len(e)>1:
        ens = ncfile.createVariable('ens',dtype('float').char,('ens'))
        ens[:] = e

    # write data 1

    if len(e)==1:
        if len(t)>1:
            data1 = ncfile.createVariable(var1,dtype('float').char,('time','lat','lon'))
            # write data to variable.
            data1[:] = data_out1

    if len(t)==1:
        if len(e)>1:
            data1 = ncfile.createVariable(var1,dtype('float').char,('ens','lat','lon'))
            # write data to variable.
            data1[:] = data_out1

    if len(t)>1:
        if len(e)>1:
            data1 = ncfile.createVariable(var1,dtype('float').char,('ens','time','lat','lon'))
            # write data to variable.
            data1[:] = data_out1

    if len(t)==1:
        if len(e)==1:
            data1 = ncfile.createVariable(var1,dtype('float').char,('lat','lon'))
            # write data to variable.
            data1[:] = data_out1

        # write data 2
        
    if len(data_out2)>1:
            if len(e)==1:
                if len(t)>1:
                    data2 = ncfile.createVariable(var1,dtype('float').char,('time','lat','lon'))
                    # write data to variable.
                    data2[:] = data_out2

            if len(t)==1:
                if len(e)>1:
                    data2 = ncfile.createVariable(var1,dtype('float').char,('ens','lat','lon'))
                    # write data to variable.
                    data2[:] = data_out2

            if len(t)>1:
                if len(e)>1:
                    data2 = ncfile.createVariable(var1,dtype('float').char,('ens','time','lat','lon'))
                    # write data to variable.
                    data2[:] = data_out2

            if len(t)==1:
                if len(e)==1:
                    data2 = ncfile.createVariable(var1,dtype('float').char,('lat','lon'))
                    # write data to variable.
                    data2[:] = data_out2

    ncfile.close()


## convert vertical pressure levels to geomatric height

     # H = p2h(T,slp,P)

        #equation from Hypsometric
        # H= z2-z1= R*T/g * ln(P0/P)
        # H= Height
        # g=9.81 m/s2
        # R=287.04 J K-1 kg-1

    #inputs:    T = air temperature one array (Kelvin)
    #           SLP = sea level pressure one array (hPa)
    #           P = Pressure level that needs to be converted in to height (hPa)

    #outputs:   H = Height (meters)

    #example:       #h=zeros(ta.shape)

                    # for i in range(len(lev)):
                    #     for j in range(len(lat)):
                    #         for k in range(len(lon)):
                    #             h[i,j,k]=p2h(T[i,j,k],slp[j,k],P[i])

def p2h(T,slp,P):


    g=9.81
    R=287.04

    H=(R*T/g)*(log(slp/P))

    return H


## static stability (buyoncy frequency N2)

     # N2, pn = N2(ta,slp,plev,lon,lat)

    #inputs:    ta = air temperature (Kelvin) (3D [time, lat, lon])
    #           slp= sea level pressure (hPa) (2D [lat, lon])
    #           plev= pressure level (hPa) (1D vertical array)

    #outputs:   N2= static stability (s^-2)
    #           pn= new pressure level (hPa)


def N2(ta,slp,plev,lon,lat):

    np=len(plev)

    nlat=len(lat)
    nlon=len(lon)


    tp=zeros(ta.shape)
    h=zeros(ta.shape)

    #convert to pt

    for i in tqdm(range(np)):
        for j in range(nlat):
            for k in range(nlon):
                tp[i,j,k]=ta[i,j,k]*((slp[j,k]/(plev[i]))**0.286)
                h[i,j,k]=p2h(ta[i,j,k],slp[j,k],plev[i])

    dtheta=tp[1:,:,:]-tp[:-1,:,:]
    theta=(tp[1:,:,:]+tp[:-1,:,:])/2
    dz=h[1:,:,:]-h[:-1,:,:]
    #hn=h[:-1,:,:]+dz/2
    pn=(plev[1:]+plev[:-1])/2

    g=9.81

    N2=(g/theta)*(dtheta/dz)

    return (N2, pn)

def cmap():
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    bottom = cm.get_cmap('YlOrRd', 128)
    top = cm.get_cmap('Blues_r', 128)

    newcolors = vstack((top(linspace(0, 1, 128)),
                       bottom(linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    return newcmp


## This function takes two time series (x 1D, y 3D) and output gives y removing the x signal

def deregress(x,y,lon=[],lat=[]):

    

    nlon=len(lon)
    nlat=len(lat)

    if nlat>1:

        y_dr=zeros(y.shape)
        y_dr[:]=nan

        for i in tqdm(range(nlat)):
            for j in range(nlon):

                y1=y[i,j,:]
                nx=isnan(x)
                ny=isnan(y1)
                ny[nx==True]=True
                ny=ny==False

                slope, intercept, r_value, p_value, std_err = stats.linregress(x[ny],y1[ny])
                reg=x*slope + intercept
                y_dr[i,j,:]=y1-reg

        return y_dr
    
    if nlat<1:
        y1=y
        nx=isnan(x)
        ny=isnan(y1)
        ny[nx==True]=True
        ny=ny==False
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[ny],y1[ny])
        reg=x*slope + intercept
        y_dr=y1-reg

        return y_dr
        
# Area functions

def spheric_dist(lat1,lat2,lon1,lon2):

    R=6367442.76
    # %  Determine proper longitudinal shift.

    l=absolute(lon2-lon1) 

    l[l>=180]=360-l[l>=180]

    #l(l>=180)=360-l(l>=180);
    # %                 
    # %  Convert Decimal degrees to radians.
    # %
    deg2rad=pi/180
    lat1=lat1*deg2rad
    lat2=lat2*deg2rad
    l=l*deg2rad

    # %
    # %  Compute the distances
    # %

    dist=R*arcsin(sqrt(((sin(l)*cos(lat2))**2)+(((sin(lat2)*cos(lat1))-(sin(lat1)*cos(lat2)*cos(l)))**2)))

    #done
    return dist

def get_grid_area(lon_rho,lat_rho):

    I, J=lon_rho.shape
    lon_u=zeros((I+1,J))
    lon_u[1:-1,:]=0.5*(lon_rho[0:-1,:]+lon_rho[1:,:])
    lon_u[0,:]=lon_rho[0,:]-0.5*(lon_rho[1,:]-lon_rho[0,:])
    lon_u[-1,:]=lon_rho[-1,:]+0.5*(lon_rho[-1,:]-lon_rho[-2,:]) 

    lat_u=zeros((I+1,J))
    lat_u[1:-1,:]=0.5*(lat_rho[0:-1,:]+lat_rho[1:,:]) 
    lat_u[0,:]=lat_rho[0,:]-0.5*(lat_rho[1,:]-lat_rho[0,:])
    lat_u[-1,:]=lat_rho[-1,:]+0.5*(lat_rho[-1,:]-lat_rho[-2,:]) 

    lon_v=zeros((I,J+1))
    lon_v[:,1:-1]=0.5*(lon_rho[:,0:-1]+lon_rho[:,1:]) 
    lon_v[:,0]=lon_rho[:,0]-0.5*(lon_rho[:,1]-lon_rho[:,0])
    lon_v[:,-1]=lon_rho[:,-1]+0.5*(lon_rho[:,-1]-lon_rho[:,-2]) 

    lat_v=zeros((I,J+1))
    lat_v[:,1:-1]=0.5*(lat_rho[:,0:-1]+lat_rho[:,1:])
    lat_v[:,0]=lat_rho[:,0]-0.5*(lat_rho[:,1]-lat_rho[:,0])
    lat_v[:,-1]=lat_rho[:,-1]+0.5*(lat_rho[:,-1]-lat_rho[:,-2]) 

    dx_rho=zeros((I,J))
    dx_rho=spheric_dist(lat_u[0:-1,:],lat_u[1:,:],lon_u[0:-1,:],lon_u[1:,:])
    dy_rho=zeros((I,J))
    dy_rho=spheric_dist(lat_v[:,0:-1],lat_v[:,1:], lon_v[:,0:-1],lon_v[:,1:])

    A=dy_rho*dx_rho
    return (A, dx_rho, dy_rho) 

def surface_integral(variable,dx_rho,dy_rho):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Calculate the integral of a variable over a surface
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    AREA=squeeze(dy_rho)*squeeze(dx_rho)
    #calculate for all rho cubes on the surface
    VAR=variable.T*AREA;
    # Intergrate the rho cubes over the surface
    Test=isnan(VAR);
    P=nonzero(Test==False);
    surface_int=sum(VAR[P]);
    return  surface_int 


def sah_area(psl,lon,lat):
    nlon= len(lon)
    nlat=len(lat)

    psl_mat=zeros(psl.shape)

    psl_mat[nonzero(psl>=1020)]=1
    psl_mat[nonzero(psl<1020)]=0

    # find SAH lonlat
    i=squeeze(nonzero((lat>=-62.3340) & (lat<=-0.7000)))
    k=squeeze(nonzero((lon>=-60) & (lon<=20)))

    psl_a=psl_mat[i[0]:i[-1]+1,k[0]:k[-1]+1]

    lat=lat[i]
    lon=lon[k]
  
    lat_rho, lon_rho=meshgrid(lat,lon)
    A, dx_rho, dy_rho=get_grid_area(lon_rho,lat_rho)

    area=surface_integral(psl_a,dx_rho,dy_rho)

    return area


def sah_maxslp(psl,lon,lat,time):
    nlon= len(lon)
    nlat=len(lat)
    nt=len(time)
    mo=12
    yr=nt//mo
    # find SAH lonlat
    i=squeeze(nonzero((lat>=-62.3340) & (lat<=-0.7000)))
    k=squeeze(nonzero((lon>=-60) & (lon<=20)))

    psl_a=psl[:,i[0]:i[-1]+1,k[0]:k[-1]+1]

    maxpsl=zeros(nt)
    for i in tqdm(range(nt)):
        maxpsl[i]=nanmax(psl_a[i,:,:])

    return maxpsl

def sah_int(psl,lon,lat,time):
    nlon= len(lon)
    nlat=len(lat)
    nt=len(time)
    mo=12
    yr=nt//mo
    # ssn decompose

    djf, mam, jja, son=ssn_decompose(psl,lon,lat,time)

    psl=reshape(psl,(yr,mo,nlat,nlon))
    ann=squeeze(nanmean(psl,1))

    # find SAH lonlat

    # %Annual: 35 W to 11 E, 38 S to 22 S
    # %DJF: 23 W to 8 E, 38 S to 27 S
    # %MAM: 27 W to 7 E, 36 S to 26 S
    # %JJA: 40 W 8E,  37S-15S 
    # %SON: 39 W to 14 E, 39 S to 19 S

    #DJF

    i=squeeze(nonzero((lat>=-38) & (lat<=-27)))
    k=squeeze(nonzero((lon>=-23) & (lon<=8)))

    djfint=djf[:,i[0]:i[-1]+1,k[0]:k[-1]+1]
    djfint=nanmean(nanmean(djfint,2),1)

    #MAM

    i=squeeze(nonzero((lat>=-36) & (lat<=-26)))
    k=squeeze(nonzero((lon>=-27) & (lon<=7)))

    mamint=mam[:,i[0]:i[-1]+1,k[0]:k[-1]+1]
    mamint=nanmean(nanmean(mamint,2),1)

    #JJA

    i=squeeze(nonzero((lat>=-37) & (lat<=-15)))
    k=squeeze(nonzero((lon>=-40) & (lon<=8)))

    jjaint=jja[:,i[0]:i[-1]+1,k[0]:k[-1]+1]
    jjaint=nanmean(nanmean(jjaint,2),1)

    #SON

    i=squeeze(nonzero((lat>=-39) & (lat<=-19)))
    k=squeeze(nonzero((lon>=-39) & (lon<=14)))

    sonint=djf[:,i[0]:i[-1]+1,k[0]:k[-1]+1]
    sonint=nanmean(nanmean(sonint,2),1)

    #Ann

    i=squeeze(nonzero((lat>=-38) & (lat<=-22)))
    k=squeeze(nonzero((lon>=-35) & (lon<=11)))

    annint=ann[:,i[0]:i[-1]+1,k[0]:k[-1]+1]
    annint=nanmean(nanmean(annint,2),1)

    return (djfint, mamint, jjaint, sonint, annint)



def fa_interp(var, lon, lat, new_lons, new_lats,time=1):
    nlon=len(new_lons)
    nlat=len(new_lats)

    shift=len(nonzero(lon>180)[0])

    #new_lons, new_lats=meshgrid(new_lons, new_lats)

    

    if time==1:
        data_interp=zeros((nlat,nlon))

        if shift>1:
            fltrn = new_lons >= 180
            new_lons = concatenate(((new_lons - 360)[fltrn], new_lons[~fltrn]))

            new_lons, new_lats=meshgrid(new_lons, new_lats)

            fltr = lon >= 180
            lon = concatenate(((lon - 360)[fltr], lon[~fltr]))

            data=var[:,:]
            data = concatenate((data[:, fltr], data[:, ~fltr]), axis=-1)
            data_interp[:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)

            data_interp = concatenate((data_interp[:, ~fltrn],data_interp[:, fltrn]), axis=-1)



        else:
            new_lons, new_lats=meshgrid(new_lons, new_lats)

            data=var[:,:]
            data_interp[:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)


    else:
        nt=len(time)
        data_interp=zeros((nt,nlat,nlon))
        if shift>1:

            fltrn = new_lons >= 180
            new_lons = concatenate(((new_lons - 360)[fltrn], new_lons[~fltrn]))
            new_lons, new_lats=meshgrid(new_lons, new_lats)

            fltr = lon >= 180
            lon = concatenate(((lon - 360)[fltr], lon[~fltr]))

            for i in tqdm(range(nt)):

                data=var[i,:,:]
                data = concatenate((data[:, fltr], data[:, ~fltr]), axis=-1)
                data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)
                data_interp = concatenate((data_interp[:, ~fltrn],data_interp[:, fltrn]), axis=-1)
        else:
            new_lons, new_lats=meshgrid(new_lons, new_lats)

            for i in tqdm(range(nt)):

                data=var[i,:,:]
                data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)


    return data_interp

def test():

    print('AOESpy v1 Test')
    print('author: Abdullah al Fahad (afahad@gmu.edu)')
    print('For Latest update: https://github.com/afahadabdullah/AOESpy')
    print('- - - - - - - - - - - - - - - - - - -')
    file='/homes/afahad/data/sst_erai_1979_2016.nc'
    sst= rnc('sst',file)
    lon=rnc('lon',file)
    lat=rnc('lat',file)
    sst=nanmean(sst,0)
    f()
    plot(sst,lon,lat, 'Test AOESpy annual mean SST plot from Era-interim data', cmap=cmap(), lat1=-89, lat2=89)
    d()

