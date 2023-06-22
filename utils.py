#! /anaconda/bin/python3 python
# -*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-29

A list of useful python functions.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-29
"""

import math
import numpy as np
import datetime


def unixtime_to_datetime(unix_time):  # input can be an array
    time = []
    for i in range(len(unix_time)):
        time.append(datetime.datetime.utcfromtimestamp(int(float(unix_time[i]))).strftime('%Y-%m-%d %H:%M:%S'))
    return time


def find_nearest_value(array, num):
    nearest_val = array[abs(array - num) == abs(array - num).min()]
    return nearest_val


def find_nearest_index(array, num):
    nearest_idx = np.where(abs(array - num) == abs(array - num).min())[0]
    return nearest_idx


def eliminate_zeros(num):  # num should be a 4 digits number

    if num[0] == '0' and num[1] == '0' and num[2] == '0':
        return num[3]
    if num[0] == '0' and num[1] == '0' and num[2] != '0':
        return num[2:]
    if num[0] == '0' and num[1] != '0':
        return num[1:]
    if num[0] != '0':
        return num


def make_up_zeros(str):
    if len(str) == 4:
        return str
    if len(str) == 3:
        return "0" + str
    if len(str) == 2:
        return "00" + str
    if len(str) == 1:
        return "000" + str


def get_weather_file(unix_time):
    pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime(
        '%Y%m%d %H%M%S')  # time handle to check CIWS database
    array = np.asarray([0, 230, 500, 730,
                        1000, 1230, 1500, 1730,
                        2000, 2230, 2500, 2730,
                        3000, 3230, 3500, 3730,
                        4000, 4230, 4500, 4730,
                        5000, 5230, 5500, 5730])

    # find the closest time for downloading data from CIWS
    nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))])))
    nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730
    return pin, nearest_value


def check_convective_weather_files(weather_path, unix_time):
    pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime('%Y%m%d %H%M%S')  # time handle to check CIWS database
    array = np.asarray([0, 230, 500, 730,
                        1000, 1230, 1500, 1730,
                        2000, 2230, 2500, 2730,
                        3000, 3230, 3500, 3730,
                        4000, 4230, 4500, 4730,
                        5000, 5230, 5500, 5730])

    nearest_value = int(find_nearest_value(array, 0.001+np.asarray([int(eliminate_zeros(pin[-4:]))])))  # find the closest time for downloading data from CIWS
    nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730

    filename = pin[:8] + "ET/ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z.nc"

    return weather_path + filename



def flight_plan_parser(str):  # use local waypoint database

    str = str[:-5] # remove last 5 characters
    str_list = str.split('.') # break the string
    str_list = list(filter(None, str_list)) # remove empty strings
    print (str_list)

    # store coordinates
    coords = []

    import csv
    for i in range(len(str_list)):
        with open('myFPDB.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] == str_list[i]:
                    coords += [[row[1], row[2]]]
    return coords


def fetch_from_web(str):  # use online waypoint database source

    str = str[:-5] # remove last 5 characters
    str_list = str.split('.') # break the string
    str_list = list(filter(None, str_list)) # remove empty strings
    print ("FP:{}".format(str_list))

    # store coordinates
    coords = []

    import urllib.request
    # query departure airports
    websource = urllib.request.urlopen("https://opennav.com/airport/{}".format(str_list[0]))
    l = websource.readlines()[18].decode("utf-8")
    lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
    coords += [[lon, lat]]

    # query waypoints
    for n in range(1, len(str_list)-1):
        try:
            websource = urllib.request.urlopen("https://opennav.com/waypoint/US/{}".format(str_list[n]))  # US only
            l = websource.readlines()[18].decode("utf-8")
            lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
            coords += [[lon, lat]]
        except:
            try:
                websource = urllib.request.urlopen("https://opennav.com/navaid/US/{}".format(str_list[n]))
                l = websource.readlines()[18].decode("utf-8")
                lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
                coords += [[lon, lat]]
            except:
                print("Waypoint {} not found from {}.".format(str_list[n], "https://opennav.com"))
                pass

    # query arrival airports
    websource = urllib.request.urlopen("https://opennav.com/airport/{}".format(str_list[-1]))
    l = websource.readlines()[18].decode("utf-8")
    lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
    coords += [[lon, lat]]

    return np.asarray(coords).astype(float)  # return flight plan as np.array


def find_index_fp(x, y, resize_ratio):
    y_max, y_min, x_max, x_min = lat2y(53.8742945085336), lat2y(19.35598953632181), lot2x(-61.65138656927017), lot2x(
        -134.3486134307298)

    s_y = np.linspace(y_min, y_max, int(3520 / resize_ratio))
    s_x = np.linspace(x_min, x_max, int(5120 / resize_ratio))

    step_x = s_x[1] - s_x[0]
    step_y = s_y[1] - s_y[0]

    # save weather values at traj point
    x_p = np.int(round((lot2x(x) - x_min) / step_x))
    y_p = np.int(round((lat2y(y) - y_min) / step_y))

    return x_p, y_p


def download_from_web(date):

    url = 'https://nomads.ncdc.noaa.gov/data/namanl/{}/{}/namanl_218_{}_0000_001.grb'.format(date[:6], date, date)

    import urllib.request
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open("NOAA/{}".format(file_name), 'wb')
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)

    f.close()
    print("Finished downloading weather files.")


def lat2y(a):
    Radius = 6378137.0  # Radius of Earth
    return math.log(math.tan(math.pi / 4 + math.radians(a) / 2)) * Radius


def lot2x(a):
    Radius = 6378137.0  # Radius of Earth
    return math.radians(a) * Radius


def merc_index_to_wgs84(index, resize_ratio):

    import pyproj

    lat_max = 53.8742945085336
    lat_min = 19.35598953632181
    lon_max = -61.65138656927017
    lon_min = -134.3486134307298

    p1 = pyproj.Proj(init="epsg:4326")
    p2 = pyproj.Proj(init="epsg:3857")
    x_min, y_min = pyproj.transform(p1, p2, lon_min, lat_min)
    x_max, y_max = pyproj.transform(p1, p2, lon_max, lat_max)
    s_y = np.linspace(y_min, y_max, 3520)
    s_x = np.linspace(x_min, x_max, 5120)
    step_x = s_x[1] - s_x[0]
    step_y = s_y[1] - s_y[0]

    y_merc = s_y[index[1]*resize_ratio]
    x_merc = s_x[index[0]*resize_ratio]

    lon, lat = pyproj.transform(p2, p1, x_merc, y_merc)

    print(lat, lon)
    return [lat, lon]


def get_date_list():
    # get date_list
    list_1 = list(range(20181101, 20181131))
    list_2 = list(range(20181201, 20181232))
    list_3 = list(range(20190101, 20190132))
    list_4 = list(range(20190201, 20190206))
    list_5 = [20170405, 20170406, 20170407, 20180723, 20180724, 20170905, 20170906]
    date_list = list_1 + list_2 + list_3 + list_4 + list_5
    to_remove = [20181229, 20181230, 20181231, 20190125, 20190126, 20190127, 20190128, 20181102]
    [date_list.remove(day) for day in to_remove]
    return date_list


if __name__ == '__main__':

    fp = 'KJFK..COATE.Q436.RAAKK.Q438.RUBYY..DABJU..KG78M..DBQ.J100.JORDY..KP72G..OBH.J10.LBF..LEWOY..KD60U..JNC..HVE..PROMT.Q88.HAKMN.ANJLL1.KLAX/0539'

    #flight_plan_parser(fp)
    #wps = fetch_from_web(fp)
    date = '20170405'
    #download_from_web(date)
    index = [4266, 1965]
    merc_index_to_wgs84(index, resize_ratio=1)
