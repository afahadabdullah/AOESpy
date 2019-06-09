# AOESpy (Still Updating)
AOESpy (v1)
Author@ Abdullah al Fahad 
(afahad@gmu.edu)
http://afahadabdullah.com

Dependency Library:

Numpy
Scipy
mpl_toolkits.basemap
Matplotlib
netCDF4 


Function:

### Other handly functions ###
# d(): displays plot based on matplotlib.pyplot

# shiftgrid:
	dataout, newlon= shiftgrid(lon0, datain, lonsin, start=True, cyclic=360.0)

	inputs: 	lon0=starting longitude for shifted grid (ending longitude if start=False)
			datain= original data with longitude the right-most dimension.
			lonsin= original longitudes
			start= if True, lon0 represents the starting longitude of the new         
                   			grid. if False, lon0 is the ending longitude. Default True.
			cyclic=	width of periodic domain (default 360)

	outputs:	dataout= shifted input data
			newlon= new shifted longitude array


