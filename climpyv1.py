# -*- coding: utf-8 -*-
# @Author: afahad
# @Date:   2018-08-18 02:07:27
# @Last Modified by:   afahad
# @Last Modified time: 2018-11-05 08:55:49
#Created on Tue Jun 19 04:29:53 2018

## All of my functions ! ##

from numpy import *
from matplotlib.mlab import find
from scipy import stats
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset as nc
import  mpl_toolkits.basemap

d=plt.show
# Read netcdf

def rnc(var,file):
    f = nc(file)
    v = f.variables[var][:]
    f.close()
    return v

# Trends (time, nlon*nlat)

def l_trend(var,lon,lat,time,sig=False):
    nlon=len(lon)
    nlat=len(lat)
    nt=len(time)
    vart=zeros(nlat*nlon)
    varp=zeros(nlat*nlon)
    
    if len(var.shape)== 3:        
        var=reshape(var,(nt,nlat*nlon)) 
        print('l_trend: assuming variable as 3D [time,lat,lon]')
        for i in range(nlat*nlon):
            v=var[:,i]  
            vart[i], intercept, r_value, varp[i], std_err=stats.linregress(time,v)
            
        vart=reshape(vart,(nlat,nlon))
        varp=reshape(varp,(nlat,nlon))
        #return (vart,varp)
        
    elif len(var.shape)== 2:
        print('l_trend: assuming variable as 2D [time,lat*lon]')
        for i in range(nlat*nlon):
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
           

#mapping functing
def plot(var,lon,lat,title,clevs=[],clevsf=[],cbar='coolwarm',lon1=-180,lon2=180,lat1=-90,lat2=90):

    map = Basemap(projection='cyl',llcrnrlat=lat1,urcrnrlat=lat2,\
                llcrnrlon=lon1,urcrnrlon=lon2,resolution='l')

    map.drawcoastlines(linewidth=.6,color='gray')
    parallels = arange(lat1,lat2+1,30.) 
    meridians = arange(lon1,lon2,60.)
    map.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.05,fontsize=8)
    map.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.05,fontsize=8)
    lons,lats= meshgrid(lon,lat)
    x,y = map(lons,lats)
    if len(clevsf)>1:
        csf = map.contourf(x,y,var,clevsf,extend='both',cmap=cbar)
        cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="12%")
        #cs = map.contour(x,y,var,clevs,colors='k',linewidths=0.3)
        #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
        plt.title(title,fontsize=9) 
    else:
        csf = map.contourf(x,y,var,extend='both',cmap=cbar)
        #cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="9%")
        cb = map.colorbar(csf,"bottom", extend='both',size="3%", pad="12%")
        #cs = map.contour(x,y,var,colors='k',linewidths=0.3)
        #plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=6, colors='k')
        plt.title(title,fontsize=9)


# seasonal decompose

def ssn_decompose(data,lon,lat,time):
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

def ssn_decompose1d(data,time):
    nt=len(time)
    mo=12
    yr=nt//mo

    data=reshape(data,(yr,mo))

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

    return (djf,mam,jja,son)

# Area functions !

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
    for i in range(nt):
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

            for i in range(nt):

                data=var[i,:,:]
                data = concatenate((data[:, fltr], data[:, ~fltr]), axis=-1)
                data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)
                data_interp = concatenate((data_interp[:, ~fltrn],data_interp[:, fltrn]), axis=-1)
        else:
            new_lons, new_lats=meshgrid(new_lons, new_lats)

            for i in range(nt):

                data=var[i,:,:]
                data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)


    return data_interp


def fa_interp_v2(var, lon, lat, new_lons, new_lats,time=1):
    nlon=len(new_lons)
    nlat=len(new_lats)

    new_lons, new_lats=meshgrid(new_lons, new_lats)

    

    if time==1:
        data_interp=zeros((nlat,nlon))

        data=var[:,:]
        data_interp[:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)

    else:

            for i in range(nt):

                data=var[i,:,:]
                data_interp[i,:,:] = mpl_toolkits.basemap.interp(data, lon, lat, new_lons, new_lats,checkbounds=False, masked=False, order=1)
                

    return data_interp

def nph_area(psl,lon,lat):
    nlon= len(lon)
    nlat=len(lat)

    psl_mat=zeros(psl.shape)

    psl_mat[nonzero(psl>=1020)]=1
    psl_mat[nonzero(psl<1020)]=0

    # find SAH lonlat
    i=squeeze(nonzero((lat>=15) & (lat<=55)))
    k=squeeze(nonzero((lon>=-180) & (lon<=-122)))

    psl_a=psl_mat[i[0]:i[-1]+1,k[0]:k[-1]+1]

    lat=lat[i]
    lon=lon[k]
  
    lat_rho, lon_rho=meshgrid(lat,lon)
    A, dx_rho, dy_rho=get_grid_area(lon_rho,lat_rho)

    area=surface_integral(psl_a,dx_rho,dy_rho)

    return area


        