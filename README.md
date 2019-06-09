# AOESpy
AOESpy (v1)
Author@ Abdullah al Fahad 
(afahad@gmu.edu)
http://afahadabdullah.com

# Dependency Library:

python 3+
Numpy
Scipy
mpl_toolkits.basemap
Matplotlib
netCDF4

### Function:

## Read netcdf data: rnc

	data=rnc(var,file)
	
	inputs:	var= string variable name that needs to be read 
			file=string file path

	outputs:	data= data read from the file


## Linear Trend: ltrend
	 
vart, varp=ltrend(var,lon,lat,time,sig=False)

	inputs:	var= variable as 3D [time,lat,lon] or 2D [time,lat*lon]
			lon=lon array
			lat=lat array
			time=time array
			sig= alpha significance value (e.g. 0.05, 0.1), if given the output data
			will have nan values in insignificant points (default False)

	outputs:	#vart= linear trend of the variable along time dimension
				#varp= P value of the trend		           

## Mapping function: plot

plot(var,lon,lat,title='',clf=[],cl=[], cmap='coolwarm',lon1=-180,lon2=180,lat1=-90,lat2=90,bar=1,p=1,m=1)

	inputs: 	var= 2D variable that will be plotted
			lon=lon
lat=lat
title='title'
clf= array of filled contoured levels
cl= array of contoured levels (optional)
cmap= string of name of colorbar (default coolwarm, for list: matplotlib colorbars)
lon1= start of lon (default -180)
lon2= end of lon (default 180)
lat1= start of lat (default -90)
lat2= end of lat (default 90)
bar= 1 (default) to plot colorbar; or 0 doesn't plot colorbar in figure
p= 1 (default) plots parallel line, or 0 doesn't plot 
m= 1 (default) plots meridioinal line, or 0 doesn't plot 


## 3D seasonal decompose from monthly time dimensions to annual, DJF, MAM, JJA, SON: season

	 ann, djf,mam,jja,son= season(data,lon,lat,time)

	inputs:	data=3D data [time lat lon]
			lon=lon
			lat=lat
			time=time

	outputs:	ann= annual mean
			djf= DJF mean
			mam= MAM mean
			jja= JJA mean
			son= SON mean


## 1D seasonal decompose from monthly time dinemsions to annual, DJF, MAM, JJA, SON: season1d

	ann, djf,mam,jja,son= season1d(data,time)

	inputs:	data=1D data [time]
			time=time

	outputs:	ann= annual mean
			djf= DJF mean
			mam= MAM mean
			jja= JJA mean
			son= SON mean


## interpolates data in desired grid: interp
	
	data_interp=interp(var, lon, lat, new_lons, new_lats,time=arange(1))

	inputs:	var= input variable 2D or 3D (includes time dimension)
			lon= lon of the variable
			lat= lat of the variable
			new_lons= new lon that grid needs to be shifted to
			new_lats= new lat that grid needs to be shifted to
			time= (optional)

	outputs:	data_interp= intrepolated data to new grids


## write variables in netcdf output. This function can write upto 2 variables in one file and required dimensions: wnc

     wnc(x,y,data_out1,var1='data1',data_out2=array([1]),var2='data2',t=array([1]),e=array([1]),file='output')

    inputs:	x=lon
    	           	y=lat
         	data_out1=first variable to write in file
               	var1='data1' ; first varible name assigned in the file
               	data_out2=second variable to write in file; (optional)
var2='data2'; first varible name assigned in the file (if second variable is given to write)
              	 t= time dimension array
               	e= ensemble dimension array (can be used as vertical level)
               	file= string of output file name (dont need to add .nc)


## convert vertical pressure levels to geometric height: p2h

     H = p2h(T,slp,P)

        #equation from Hypsometric
        # H= z2-z1= R*T/g * ln(P0/P)
        # H= Height
        # g=9.81 m/s2
        # R=287.04 J K-1 kg-1

    inputs:	T = air temperature one array (Kelvin)
SLP = sea level pressure one array (hPa)
         	 P = Pressure level that needs to be converted in to height (hPa)

    outputs:   	H = Height (meters)

    #example:       #h=zeros(ta.shape)

                    # for i in range(len(lev)):
                    #     for j in range(len(lat)):
                    #         for k in range(len(lon)):
                    #             h[i,j,k]=p2h(T[i,j,k],slp[j,k],P[i])


## static stability (buyoncy frequency N2): N2

     N2, pn = N2(ta,slp,plev,lon,lat)

    inputs:   	 ta = air temperature (Kelvin) (3D [time, lat, lon])
    		slp= sea level pressure (hPa) (2D [lat, lon])
    		plev= pressure level (hPa) (1D vertical array)

    outputs:   	N2= static stability (s^-2)
pn= new pressure level (hPa)


## customized colormap
     cmap()


## This function takes two time series (x 1D, y 3D) and output gives y removing the x signal
    deregress(x,y,lon=[],lat=[])


### Other handly functions ###

# d() 
#displays plot based on matplotlib.pyplot

## f()
      	 #an output default figure size 


# shiftgrid
	dataout, newlon= shiftgrid(lon0, datain, lonsin, start=True, cyclic=360.0)

	inputs: 	lon0=starting longitude for shifted grid (ending longitude if start=False)
			datain= original data with longitude the right-most dimension.
			lonsin= original longitudes
			start= if True, lon0 represents the starting longitude of the new         
                   			grid. if False, lon0 is the ending longitude. Default True.
			cyclic=	width of periodic domain (default 360)

	outputs:	dataout= shifted input data
			newlon= new shifted longitude array

