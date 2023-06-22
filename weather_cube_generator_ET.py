#!/home/anaconda3 python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 2, 2019
Last Modified on Feb 11, 2019

@author: Nan Xu
@modified: Yutian Pang

"""

import os
import time
import pandas as pd
from utils import *
import numpy as np
from netCDF4 import Dataset


class weather_cube_generator(object):

    def __init__(self, cfg):
        self.cube_size = cfg['cube_size']
        self.resize_ratio = cfg['resize_ratio']
        self.weather_path = cfg['weather_path']
        self.date = cfg['date']
        self.downsample_ratio = cfg['downsample_ratio']
        self.call_sign = cfg['call_sign']
        print("Processing flight {}_{}".format(self.date, self.call_sign))

        self.traj = pd.read_csv(cfg['trajectory_path'])
        # self.traj = self.traj.iloc[::self.downsample_ratio, :].reset_index()  # downsample trajectory

        self.departure_airport = cfg['departure_airport']
        self.arrival_airport = cfg['arrival_airport']

        try:
            os.makedirs('weather data/{}2{}_ET'.format(self.departure_airport, self.arrival_airport))
        except OSError:
            pass

        try:
            os.makedirs('weather data/{}2{}_ET_point'.format(self.departure_airport, self.arrival_airport))
        except OSError:
            pass

        # self.lats = np.load('lats.npy')
        # self.lons = np.load('lons.npy')

    def find_mean(self, x, y, values):
        # find mean
        x_p_index = self.resize_ratio * np.linspace(x - 1, x + 1, 2 * self.resize_ratio + 1)
        #y_p_index = values.shape[0] - self.resize_ratio * np.linspace(y - 1, y + 1, 2 * self.resize_ratio + 1)
        y_p_index = self.resize_ratio * np.linspace(y - 1, y + 1, 2 * self.resize_ratio + 1)

        x_p_index = x_p_index.astype('int')
        y_p_index = y_p_index.astype('int')

        point_t_values = values[y_p_index.min():y_p_index.max(), x_p_index.min():x_p_index.max()]
        point_t_values = np.sum(point_t_values) / (4 * self.resize_ratio ** 2)

        return point_t_values

    def get_cube(self):

        y_max, y_min, x_max, x_min = lat2y(53.8742945085336), lat2y(19.35598953632181), lot2x(-61.65138656927017), lot2x(-134.3486134307298)

        dim = np.int32(np.linspace(1, len(self.traj)-1, len(self.traj)-1))
        nn = np.int32(np.linspace(1, self.cube_size, self.cube_size))
        nn2 = np.int32(np.linspace(2, self.cube_size, self.cube_size-1))

        s_y = np.linspace(y_min, y_max, int(3520/self.resize_ratio))
        s_x = np.linspace(x_min, x_max, int(5120/self.resize_ratio))

        step_x = s_x[1] - s_x[0]
        step_y = s_y[1] - s_y[0]

        weather_tensor = []
        point_t = []

        # information need from the original data file
        x = self.traj['LONGITUDE']
        y = self.traj['LATITUDE']
        t = self.traj['UNIX TIME']

        start = time.time()

        #for i in range(1, 100):  # debug only
        for i in dim:

            # compute index
            if (i+1) % int((len(self.traj)/10)) == 0:
                print("Working on Point {}/{}".format(1+i, len(self.traj)))

            # check weather file exists at time i
            weather_file = check_convective_weather_files(self.weather_path, t[i])
            data = Dataset(weather_file)
            values = np.squeeze(data.variables['ECHO_TOP'])

            # search direction
            dx_ = x[i] - x[i-1] + 1e-8
            dire_x = dx_/np.abs(dx_)
            dy_ = y[i] - y[i-1] + 1e-8
            dire_y = dy_ / np.abs(dy_)

            # Line 1  Along the Traj
            slope_m = (lat2y(y[i]) - lat2y(y[i-1]) + 1e-8) / (lot2x(x[i]) - lot2x(x[i-1]) + 1e-8)
            angle_m = math.atan(slope_m)

            # Line 2 Bottom Boundary
            slope_b = -(lot2x(x[i]) - lot2x(x[i-1]) + 1e-8) / (lat2y(y[i]) - lat2y(y[i-1]) + 1e-8)
            angle_b = math.atan(slope_b)

            delta_Xb = np.abs(step_x * self.cube_size * math.cos(angle_b))
            Y_b = lambda s: slope_b * (s - lot2x(x[i])) + lat2y(y[i])
            Xb_2 = lot2x(x[i]) + 0.5 * delta_Xb  # x-coord right-bottom corner
            Yb_2 = Y_b(Xb_2)  # y-coord right-bottom corner

            # point count
            h = 0

            # store 20x20 values
            weather_v = np.zeros((self.cube_size**2, 1))

            # save weather values at traj point
            x_p = np.int(round((lot2x(x[i]) - x_min)/step_x))
            y_p = np.int(round((lat2y(y[i]) - y_min) / step_y))

            point_t_values = self.find_mean(x_p, y_p, values)
            point_t.append((x_p, y_p, point_t_values))

            # Loop to generate all points coordinates
            for i in nn:

                d_x0 = np.abs(step_y * math.cos(angle_m))
                d_y0 = np.abs(step_y * math.sin(angle_m))

                Xb_2i = np.int(round((Xb_2 - x_min) / step_x))
                Yb_2i = np.int(round((Yb_2 - y_min) / step_y))

                point = (Xb_2i, Yb_2i)

                weather_v[h] = self.find_mean(Xb_2i, Yb_2i, values)

                h = h + 1

                for j in nn2:

                    d_x = np.abs(step_x * math.cos(angle_b))

                    Y_b2 = lambda s: slope_b * (s - Xb_2) + Yb_2
                    x_ = Xb_2 - d_x * (j - 1)
                    y_ = Y_b2(x_)

                    x_i = np.int(round((x_ - x_min) / step_x))
                    y_i = np.int(round((y_ - y_min) / step_y))

                    point = (x_i, y_i)  # index of weather

                    weather_v[h] = self.find_mean(x_i, y_i, values)

                    h = h + 1

                Xb_2 = Xb_2 + dire_x * d_x0
                Yb_2 = Yb_2 + dire_y * d_y0

            weather_v = weather_v.reshape(self.cube_size, self.cube_size)
            weather_tensor.append(weather_v)

        print("Total time for one trajectory is: ", time.time() - start)

        # save data
        np.save('weather data/{}2{}_ET/{}_{}'.format(self.departure_airport, self.arrival_airport, self.date, self.call_sign), weather_tensor)
        np.save('weather data/{}2{}_ET_point/{}_{}'.format(self.departure_airport, self.arrival_airport, self.date, self.call_sign), point_t)


if __name__ == '__main__':

    # cfg ={'cube_size': 20,
    #       'resize_ratio': 1,
    #       'downsample_ratio': 5,
    #       'date': 20170405,
    #       'call_sign': 'AAL133',
    #       'departure_airport': 'JFK',
    #       'arrival_airport': 'LAX',
    #       'weather_path': '/mnt/data/Research/data/',
    #       }
    #
    # cfg['trajectory_path'] = 'raw_track/track_point_{}_{}2{}/{}_{}.csv'.\
    #     format(cfg['date'], cfg['departure_airport'], cfg['arrival_airport'], cfg['call_sign'], cfg['date'])
    #
    # fun = weather_cube_generator(cfg)
    # fun.get_cube()


    # run on trajectory point

    date_list = [20170405, 20170406, 20170407]  # folder name to loop through

    cfg = {'cube_size': 20,  # the size of cube to generate
           'resize_ratio': 1,  # ratio of resize performs to the original weather source
           'downsample_ratio': 5,  # downsample ratio to trajectory files
           'departure_airport': 'JFK',
           'arrival_airport': 'LAX',
           'output_dimension': 1000,  # output dimension for trajectory and flight plan
           'altitude_buffer': 0,  # altitude buffer unit: feet
           'weather_path': '/mnt/data/Research/data/',  # path to weather file
           }

    for date in date_list:
        call_sign_list = sorted([x.split('.')[0] for x in os.listdir("raw_track/track_point_{}_{}2{}/".
                                                                     format(date, cfg['departure_airport'], cfg['arrival_airport']))])
        for call_sign in call_sign_list:

            cfg['date'] = date
            cfg['call_sign'] = call_sign.split('_')[0]

            # modify departure and arrival airport
            # cfg['trajectory_path'] = 'raw_track/track_points_{}_{}2{}/{}_{}.csv'. \
            cfg['trajectory_path'] = 'raw_track/track_point_{}_{}2{}/{}_{}.csv'. \
                format(cfg['date'], cfg['departure_airport'], cfg['arrival_airport'], cfg['call_sign'], cfg['date'])
            print(cfg['trajectory_path'])

            try:
                fun = weather_cube_generator(cfg)
                fun.get_cube()
                del fun
                print("Finish weather data for {}.".format(call_sign))
            except:  # ignore file not found error
                print("Error in weather data for {}".format(call_sign))
                pass




