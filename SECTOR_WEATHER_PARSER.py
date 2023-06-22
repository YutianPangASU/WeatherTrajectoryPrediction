#! /home/ypang6/anaconda3/bin/python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-09-12

This Python script is used to process the ciws-echotop weather features. User is able to change the size of the weather
cube and the resize ratio for different resolution. The dimension of the processed weather data is determined by the
trajectory length N and cube size C. The dimension of the weather tensor is CxCx(N-1).
For instance, the trajectory feed has 50 points and the cube size is 25, the output dimension would be a dictionary with
size ?x25x25x49 where ? is the number of call signs.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-09-12
"""

import os
import time
import pickle
from utils import *
import numpy as np
from netCDF4 import Dataset


class weather_cube_generator(object):
    def __init__(self, cfg):
        self.sector_name = cfg['sector_name']
        self.date = cfg['date']
        self.resize_ratio = cfg['resize_ratio']
        self.cube_size = cfg['cube_size']
        self.weather_path = cfg['weather_path']
        self.fp = pickle.load(open('FP_{}_{}.p'.format(self.sector_name, self.date), 'rb'))
        self.traj_dict = pickle.load(open('TRACKS_{}_{}.p'.format(self.sector_name, self.date), 'rb'))

    def get_weather_cube(self):
        weather_tensor_dict = {}
        weather_point_dict = {}
        for self.call_sign, self.traj in self.traj_dict.items():
            print('Processing Flight {}'.format(self.call_sign))
            try:
                weather_tensor_dict[self.call_sign], weather_point_dict[self.call_sign] = self.get_cube()
                print("Finish weather data for {}.".format(self.call_sign))
            except:  # ignore file not found error
                print("Error in weather data for {}".format(self.call_sign))
                pass
        pickle.dump(weather_tensor_dict, open('WEATHER_CUBE_{}_{}.p'.format(self.sector_name, self.date), 'wb'))
        pickle.dump(weather_point_dict, open('WEATHER_POINT_{}_{}.p'.format(self.sector_name, self.date), 'wb'))

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
        x = np.asarray(self.traj[10])  # longitude
        y = np.asarray(self.traj[9])  # latitude
        t = np.asarray(self.traj.index.values)  # unix time

        start = time.time()

        for i in dim:

            # compute index
            if (i+1) % int((len(self.traj)/5)) == 0:
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

        return weather_tensor, point_t
        # save data
        # np.save('weather_data/{}_{}_ET/{}'.format(self.date, self.sector_name, self.call_sign), weather_tensor)
        # np.save('weather_data/{}_{}_ET_point/{}'.format(self.date, self.sector_name, self.call_sign), point_t)


if __name__ == '__main__':
    # data = pickle.load(open('WEATHER_CUBE_ZID_20190805.p', 'rb'))
    # data1 = pickle.load(open('TRACKS_ZID_20190805.p', 'rb'))
    cfg = {}
    cfg['date'] = '20190624'
    #cfg['sector_name'] = 'ZID'
    cfg['cube_size'] = 32
    cfg['sector_name'] = 'ZOB'
    cfg['resize_ratio'] = 10
    cfg['weather_path'] = '/media/ypang6/paralab/Research/data/'
    fun = weather_cube_generator(cfg)
    fun.get_weather_cube()
