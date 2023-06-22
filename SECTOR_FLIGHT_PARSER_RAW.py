#! /home/ypang6/anaconda3/bin/python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-09-10

This Python script is used to process the raw IFF csv sector data.
The inputs are data path and sector name.
The outputs are two dictionaries, the raw flight track dictionary for one day and the string flight plan for one day.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-09-10
"""

import pandas as pd
import numpy as np
import os


class FAA_Sector_Parser(object):

    def __init__(self, cfg):
        self.date = cfg['file_date']
        self.sector_name = cfg['sector_name']

    def get_flight_data(self):

        df = pd.read_csv('{}/{}/IFF_{}_{}.csv'.format(cfg['path_to_data'], self.sector_name, self.sector_name, str(self.date)), names=range(0, 18), low_memory=False)
        print("File Loaded.")

        df_clean = df.loc[df.index[df[0] == 3]]  # take trajectory
        df_fp = df.loc[df.index[df[0] == 4], [2, 17]]
        df_clean = df_clean.loc[:, [1, 2, 7, 9, 10, 11]]

        call_sign_list = df_clean[7].unique().tolist()
        flight_id_list = df_clean[2].unique().tolist()
        print('Total number of flight passing through {} on {} is {}'.format(self.sector_name, self.date, len(flight_id_list)))

        # dict_tracks = {}
        # dict_fps = {}
        # for callsign in call_sign_list:
        #     tracks = df_clean[df_clean[7] == callsign]
        #     fps = df_fp[df_fp[7] == callsign].iloc[0][17]
        #
        #     dict_tracks[callsign] = tracks
        #     dict_fps[callsign] = fps

        dict_tracks = {}
        dict_fps = {}
        for flight_id in flight_id_list:
            tracks = df_clean[df_clean[2] == flight_id]
            fps = df_fp[df_fp[2] == flight_id].iloc[0][17]

            dict_tracks[flight_id] = tracks
            dict_fps[flight_id] = fps

        print('SAVING.......................')
        np.save('{}/FP_{}_{}.npy'.format(self.sector_name, self.sector_name, self.date), dict_fps)
        np.save('{}/TRACKS_{}_{}.npy'.format(self.sector_name, self.sector_name, self.date), dict_tracks)
        print('DONE')


if __name__ == '__main__':
    cfg = {}
    cfg['path_to_data'] = '/media/ypang6/paralab/Research/data/'
    cfg['sector_name'] = 'ZOB' #'zny zdc zob

    date_list = [20190624]
    #date_list = sorted([x.split('_')[2].split('.')[0] for x in os.listdir(cfg['path_to_data'] + cfg['sector_name'])])

    for date in date_list:
        cfg['file_date'] = date
        try:
            fun = FAA_Sector_Parser(cfg).get_flight_data()
            del fun
            print("Finish flight data for {}.".format(date))
        except:
            print("Error in flight data on {}.".format(date))
            pass
