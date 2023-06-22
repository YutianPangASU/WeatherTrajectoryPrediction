import numpy as np
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

date = 20190625
time = '060000'
filename = 'ciws.EchoTop.{}T{}Z.nc'.format(date, time)
path_to_file = '/media/ypang6/paralab/Research/data/{}ET/'.format(date)

nc = NetCDFFile(path_to_file + filename)
data = nc.variables['ECHO_TOP'][:]
loncorners = nc.variables['x0'][:]
latcorners = nc.variables['y0'][:]

# create new figure, axes instances.
fig = plt.figure()
#plt.title('{}'.format(filename))

# setup mercator map projection
m = Basemap(width=2559500*2, height=1759500*2, resolution='l', projection='laea', lat_ts=50, lat_0=38, lon_0=-98)

m.drawcoastlines()
m.drawstates(linewidth=.25)
m.drawcountries(linewidth=1)
# m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
# draw meridians
m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])


m.shadedrelief()
# m.drawcoastlines(color='gray')
# m.drawcountries(color='gray')
# m.drawstates(color='gray')


ny = data.shape[2]
nx = data.shape[3]
lons, lats = m.makegrid(nx, ny)

# get lat/lons of ny by nx evenly space grid
x, y = m(lons, lats)  # compute map proj coordinates

data = data[0, 0, :, :].clip(min=0)

# draw filled contours
cs = m.contour(x, y, data)


#plt.colorbar(label=r'$\log_{10}({\rm data})$')

#add color bar
cbar = m.colorbar(cs, location='bottom', pad="5%")
#cbar.set_label('VIL Unit')

#plt.show()
plt.savefig('{}.png'.format(filename))
plt.close(fig)