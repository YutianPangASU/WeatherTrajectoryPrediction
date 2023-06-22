import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
os.environ['PROJ_LIB'] = '/home/ypang6/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

num_flight = 200

zid = pickle.load((open('FP_ZID_20190624.p', 'rb')))
ztl = pickle.load((open('FP_ZTL_20190624.p', 'rb')))
zdc = pickle.load((open('FP_ZDC_20190624.p', 'rb')))
zob = pickle.load((open('FP_ZOB_20190624.p', 'rb')))
zny = pickle.load((open('FP_ZNY_20190624.p', 'rb')))

# create new figure, axes instances.
#fig = plt.figure()
plt.figure(figsize=(10, 8))

# m = Basemap(llcrnrlon=-134.5, llcrnrlat=19.36, urcrnrlon=-61.5, urcrnrlat=48.90,
#             rsphere=(6378137.00, 6356752.3142), resolution='h', projection='merc', area_thresh=10000.)
#m = Basemap(llcrnrlon=-100, llcrnrlat=26, urcrnrlon=-70, urcrnrlat=47,
#            rsphere=(6378137.00, 6356752.3142), resolution='h', projection='merc', area_thresh=10000.)
            
m = Basemap(width=2559500*2, height=1759500*2, resolution='l', projection='laea', lat_ts=50, lat_0=38, lon_0=-98)


m.drawcoastlines()
m.drawstates(linewidth=.25)
m.drawcountries(linewidth=1)
# m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
# draw meridians
m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])
m.shadedrelief(scale=0.5)

for sector, c in zip([zdc, zid, ztl, zob, zny], ['black', 'blue', 'green', 'red', 'yellow']):
    k = 0
    for key, values in sector.items():
        # Convert latitude and longitude to coordinates X and Y
        x, y = m(values.values[:, 1], values.values[:, 0])
        if k == 0:
            m.plot(x, y, marker=None, color=c, linewidth=0.75)
        else:
            m.plot(x, y, marker=None, color=c, linewidth=0.75)

        k = k+1
        if k == num_flight:
            break

plt.legend(handles=[mpatches.Patch(color='black', label='ZDC'),
                    mpatches.Patch(color='blue', label='ZID'),
                    mpatches.Patch(color='red', label='ZOB'),
                    mpatches.Patch(color='green', label='ZTL'),
                    mpatches.Patch(color='yellow', label='ZNY')])
#plt.show()
plt.tight_layout()
plt.title('Visualization of Flight Plans ({} Each)'.format(num_flight))
plt.savefig('Sectors_{}_fp.png'.format(num_flight), dpi=300)
plt.close()
