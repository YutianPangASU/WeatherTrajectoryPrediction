import numpy as np
# ascii_grid = np.loadtxt("NOAAGlobalTemp.gridded.v4.0.1.201711.asc", skiprows=1)

from netCDF4 import Dataset
data = Dataset('gistemp250_GHCNv4.nc')
#data = Dataset('Complete_TAVG_Daily_LatLong1_2010.nc')
from netCDF4 import date2index
from datetime import datetime

year = 2019
month = 2
day = 15
filename = 'tempanomly_{}_{}_{}'.format(year, month, day)
timeindex = date2index(datetime(year, month, day), data.variables['time'])

lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
lon, lat = np.meshgrid(lon, lat)
temp_anomaly = data.variables['tempanomaly'][timeindex]  # the base period 1951-1980

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(10, 8))
#m = Basemap(projection='lcc', resolution='c', width=8E6, height=8E6, lat_0=45, lon_0=-100,)  # north america
m = Basemap(width=2559500*2, height=1759500*2, resolution='l', projection='laea', lat_ts=50, lat_0=38, lon_0=-98)

m.shadedrelief(scale=0.5)
m.pcolormesh(lon, lat, temp_anomaly, latlon=True, cmap='RdBu_r')
m.drawcoastlines()
m.drawstates(linewidth=.25)
m.drawcountries(linewidth=1)

# draw parallels
m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
# draw meridians
m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])


#plt.clim(-8, 8)

plt.title('temperature anamoly {} {} {}'.format(year, month, day))
plt.colorbar(label='temperature anomaly (°C)')
#plt.show()
plt.savefig('{}.png'.format(filename))
# plt.close(fig)