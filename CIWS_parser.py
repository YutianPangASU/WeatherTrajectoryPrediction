from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import pyproj
from utils import *
import cv2 as cv

class load_ET(object):

    def __init__(self, date):

        a = 2559500
        b = 1759500

        self.x = np.arange(-a, a+1000, 1000)  # longitude
        self.y = np.arange(-b, b+1000, 1000)  # latitude
        self.lon = np.zeros_like(self.x, dtype='float64')  # allocate space
        self.lat = np.zeros_like(self.y, dtype='float64')
        self.date = date

    def save_labels(self):

        # # handle = str('20170406EchoTop/ciws.EchoTop.20170406T000000Z.nc')
        # # data = Dataset(handle)
        #
        # # data = Dataset("20170406EchoTop/ciws.EchoTop.20170406T000000Z.nc")
        #
        # print data.file_format
        #
        # print data.dimensions.keys()
        #
        # print data.dimensions['time']
        #
        # print data.variables.keys()
        #
        # print data.dimensions['x0']  # get dimensions of a variable
        #
        # print data.variables['ECHO_TOP']  # get variable information
        #
        # print data.Conventions
        #
        # print data.variables['ECHO_TOP'].units   # check unit of your specified variable
        #
        #
        # #x = np.asarray(data.variables['x0'])  # projection x coordinate ??
        # #y = np.asarray(data.variables['y0'])

        # convert to WGS84
        p = pyproj.Proj("+proj=laea +lat_0=38 +lat_ts=60 +lon_0=-98 +k=90 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs")

        # lon, lat = p(x, y, inverse=True)
        # p1 = pyproj.Proj(init='epsg:3857')
        # p2 = pyproj.Proj(init='epsg:4326')

        for i in range(len(self.x)):
            for j in range(len(self.y)):
                self.lon[i], self.lat[j] = p(self.x[i], self.y[j], inverse=True)

        # save lon and lat
        np.save('lon.npy', self.lon)
        np.save('lat.npy', self.lat)


    def load_labels(self):

        self.lon = np.load('lon.npy')
        self.lat = np.load('lat.npy')

    def save_pics(self):

        handle = sorted(os.listdir("data/" + str(self.date) + "ET"))

        print ('There is ' + repr(len(handle)) + ' data files')

        for i in range(len(handle)):
        #for i in range(10):
            data = Dataset("data/" + str(self.date) + "ET/" + handle[i])
            values = np.squeeze(data.variables['ECHO_TOP'])  # extract values

            # save EchoTop values and restore a 3d array
            #self.GY.append(values)

            plt.contourf(self.lon, self.lat, values)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(handle[i])
            # plt.axes([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()])
            plt.show()

            plt.savefig('EchoTopPic/' + str('{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())))
            print ('I\'m reading file ' + repr(i))

        #io.savemat('EchoTop_20170406_WholeDay.mat', {'EchoTop': np.asarray(self.GY)})  # save whole day values into mat file

    def plot_weather_contour(self, unix_time, call_sign):

        pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime('%Y%m%d %H%M%S')  # time handle to check CIWS database
        array = np.asarray([0, 230, 500, 730,
                            1000, 1230, 1500, 1730,
                            2000, 2230, 2500, 2730,
                            3000, 3230, 3500, 3730,
                            4000, 4230, 4500, 4730,
                            5000, 5230, 5500, 5730])

        nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))])))  # find the closest time for downloading data from CIWS
        nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730

        # find compared nc file
        data = Dataset("data/" + pin[:8] + "EchoTop/ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z.nc")
        values = np.squeeze(data.variables['ECHO_TOP'])  # extract values
        plt.contourf(self.lon, self.lat, values)

        plt.savefig('EchoTopPic/' + str(call_sign) + ' ' + pin)

        # plt.show()

    def crop_weather_contour_FET(self, num, unix_time, call_sign, lat_start_idx, lat_end_idx, lon_start_idx, lon_end_idx, y_train, lon_start_idx_ori, lon_end_idx_ori, lat_start_idx_ori, lat_end_idx_ori, hold=False):

        pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime('%Y%m%d %H%M%S')  # time handle to check CIWS database

        # for ET
        # array = np.asarray([0, 230, 500, 730, 1000, 1230, 1500, 1730, 2000, 2230, 2500, 2730, 3000, 3230, 3500, 3730, 4000, 4230, 4500, 4730, 5000, 5230, 5500, 5730])

        # for FET
        array = np.asarray([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])

        nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))]))[0])  # find the closest time for downloading data from CIWS
        nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730

        # find compared nc file
        data = Dataset("data/" + pin[:8] + "FET/ciws.EchoTopsForecast." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z.nc")
        values = np.squeeze(data.variables['ECHO_TOP'])[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]

        # delete negative values
        values[values < 0] = 0

        # resize the matrix to 100 by 100 using opencv function
        resized_values = cv.resize(values, (100, 100))

        # normalize the data
        # scaled_values = scale_linear_bycolumn(resized_values, high=1.0, low=0.0)

        # load self.lon and self.lat
        self.lon = np.load('lon.npy')
        self.lat = np.load('lat.npy')

        # resize long and lat to 100 for plots
        lon_new = np.linspace(self.lon[lon_start_idx], self.lon[lon_end_idx], num=100)
        lat_new = np.linspace(self.lat[lat_start_idx], self.lat[lat_end_idx], num=100)

        # plt.contourf(self.lon[lon_start_idx:lon_end_idx], self.lat[lat_start_idx:lat_end_idx], resized_values)
        plt.contourf(lon_new, lat_new, resized_values)

        #plt.contourf(lon_new, lat_new, scaled_values)

        if hold is True:
            plt.hold(True)
            xx = np.asarray([self.lon[lon_start_idx_ori], y_train[0], y_train[2], y_train[4], self.lon[lon_end_idx_ori]])
            yy = np.asarray([self.lat[lat_start_idx_ori], y_train[1], y_train[3], y_train[5], self.lat[lat_end_idx_ori]])
            plt.plot(xx, yy, "--ko", linewidth=2)
            plt.plot([xx[0], xx[-1]], [yy[0], yy[-1]], "-k*")
            plt.plot(y_train[0], y_train[1], 'r*', y_train[2], y_train[3], 'g*', y_train[4], y_train[5], 'b*')
        #plt.show()

        # save figure
        plt.savefig('x_train/' + str(call_sign) + ' ' + pin + ' ' + str(num))
        plt.hold(False)

        # return the x_train matrix
        return resized_values
        #return scaled_values

    def crop_weather_contour_ET(self, num, unix_time, call_sign, lat_start_idx, lat_end_idx, lon_start_idx,
                                 lon_end_idx, y_train, lon_start_idx_ori, lon_end_idx_ori, lat_start_idx_ori,
                                 lat_end_idx_ori, hold=False):

        pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime(
            '%Y%m%d %H%M%S')  # time handle to check CIWS database

        # for ET
        array = np.asarray([0, 230, 500, 730, 1000, 1230, 1500, 1730, 2000, 2230, 2500, 2730, 3000, 3230, 3500, 3730,
                            4000, 4230, 4500, 4730, 5000, 5230, 5500, 5730])

        # for FET
        # array = np.asarray([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])

        nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))]))[
                                0])  # find the closest time for downloading data from CIWS
        nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730

        # find compared nc file
        data = Dataset(
            "data/" + pin[:8] + "ET/ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z.nc")
        values = np.squeeze(data.variables['ECHO_TOP'])[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]

        # delete negative values
        values[values < 0] = 0

        # resize the matrix to 100 by 100 using opencv function
        resized_values = cv.resize(values.T, (100, 100))

        # normalize the data
        # scaled_values = scale_linear_bycolumn(resized_values, high=1.0, low=0.0)

        # load self.lon and self.lat
        self.lon = np.load('lon.npy')
        self.lat = np.load('lat.npy')

        # resize long and lat to 100 for plots
        lon_new = np.linspace(self.lon[lon_start_idx], self.lon[lon_end_idx], num=100)
        lat_new = np.linspace(self.lat[lat_start_idx], self.lat[lat_end_idx], num=100)

        # plt.contourf(self.lon[lon_start_idx:lon_end_idx], self.lat[lat_start_idx:lat_end_idx], resized_values)
        #plt.contourf(lon_new, lat_new, scaled_values)
        plt.contourf(lon_new, lat_new, resized_values)

        if hold is True:
            plt.hold(True)
            xx = np.asarray(
                [self.lon[lon_start_idx_ori], y_train[0], y_train[2], y_train[4], self.lon[lon_end_idx_ori]])
            yy = np.asarray(
                [self.lat[lat_start_idx_ori], y_train[1], y_train[3], y_train[5], self.lat[lat_end_idx_ori]])
            plt.plot(xx, yy, "--ko", linewidth=2)
            plt.plot([xx[0], xx[-1]], [yy[0], yy[-1]], "-k*")
            plt.plot(y_train[0], y_train[1], 'r*', y_train[2], y_train[3], 'g*', y_train[4], y_train[5], 'b*')
        # plt.show()

        # save figure
        plt.savefig('x_train/' + str(call_sign) + ' ' + pin + ' ' + str(num))
        plt.hold(False)

        # return the x_train matrix
        return resized_values
        #return scaled_values


if __name__ == '__main__':

    a = 2559500
    b = 1759500
    date = 20170406

    unix_time = 1491450567.000  # a correct time
    call_sign = 'AAL717'

    fun = load_ET(date)
    # fun.save_labels()  # only need to run this function once
    fun.load_labels()
    fun.save_pics()
    fun.plot_weather_contour(unix_time, call_sign)


