import numpy as np
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

date = 20190624
time = '120000'
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
#m = Basemap(llcrnrlon=-130.5, llcrnrlat=21.36, urcrnrlon=-58.5, urcrnrlat=50.90,
#           rsphere=(6378137.00, 6356752.3142), resolution='h', projection='merc', area_thresh=10000.)

m.drawcoastlines()
m.drawstates(linewidth=.25)
m.drawcountries(linewidth=1)
# m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
# draw meridians
m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])


m.shadedrelief(scale=0.5)
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
#cs = m.contourf(x, y, data, vmin=0, vmax=64000)
#cs = m.contour(x, y, data, cmap='YlGnBu', vmin=0, vmax=64000)
#cs = m.contourf(x, y, data, cmap='gist_earth', vmin=0, vmax=64000)
cs = m.contourf(x, y, data, cmap='YlGnBu', vmin=0, vmax=64000)

# # load the shapefile, use the name 'states'
# m.readshapefile('st99_d00', name='states', drawbounds=True)
#
# # collect the state names from the shapefile attributes so we can
# # look up the shape obect for a state by it's name
# state_names = []
# for shape_dict in m.states_info:
#     state_names.append(shape_dict['NAME'])
#
# ax = plt.gca() # get current axes instance
#
# # get Texas and draw the filled polygon
# seg = m.states[state_names.index('Texas')]
# poly = Polygon(seg, facecolor='red', edgecolor='red')
# ax.add_patch(poly)
#plt.colorbar(label=r'$x10^3$', orientation='horizontal')

# add color bar
bounds = np.asarray([0, 8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000])
cbar = m.colorbar(cs, location='bottom', pad="10%", size='5%', ticks=bounds)
cbar.set_label('EchoTop/Ft')

plt.tight_layout()
#plt.show()
plt.savefig('{}.png'.format(filename), dpi=300)
# plt.close(fig)
