import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import time


class FAA_Parser(object):

    def __init__(self, call_sign, time):

        # t0 = time.time()
        # n = sum(1 for line in open('data/IFF_USA_' + time + '_050000_86396.csv'))
        # print "loaded " + str(n) + " rows of data"
        # print('Elapsed time : ', time.time() - t0)

        self.df = pd.read_csv('data/IFF_USA_' + time + '.csv', skiprows=0, nrows=5000000, names=range(0, 18))

        # specific row numbers to keep
        self.rows = []
        self.rows.extend(self.df.index[self.df[7] == call_sign])

        # specific colomn numbers to keep
        # cols = [0, 1, 7, 17]  # flightID, time, flight number, flight plan
        self.cols = [0, 1, 7, 9, 10, 11, 17]  # include lat and lon

        self.track_point = []

    def get_flight_plan(self):

        # restore the data from dataframe
        data = self.df.ix[self.rows][self.cols]

        # clear ? and nan values in the dataframe
        data = data.replace({'?': np.nan}).dropna()

        # divide data into flight plan and track point
        self.track_point = np.asarray(data[data[0] == 3])
        flight_plan = data[data[0] == 4]

        # flight_plan = list(set(flight_plan[17]))  # remove duplicate in flight plan
        fp, fp_indices = np.unique(flight_plan[17], return_index=True)
        fp_indices = np.sort(fp_indices)

        flight_plan = flight_plan.values[fp_indices]
        flight_plan_change_time = flight_plan[:, 1]
        flight_plan_change = flight_plan[:, -1]

        self.track_point = np.delete(self.track_point, [0, 2, 6], axis=1)  # col1:unix time, col2:lon, col3:lat
        self.track_point[:, [1, 2]] = self.track_point[:, [2, 1]]  # swap last two colomns

        return flight_plan_change_time, flight_plan_change, self.track_point

    def plot_real_trajectory(self):
        #fig, ax = plt.subplots()
        plt.plot(self.track_point[:, 1], self.track_point[:, 2])

        #plot_url py.plot_mpl(fig, filename="mpl-scatter")
        #plt.plot(self.track_point[:, 3], self.track_point[:, 4], 'go-')
        plt.show()


if __name__ == '__main__':  # main function only for testing purpose

    fun = FAA_Parser(call_sign='AAL1446', time='20170406')
    flight_plan__sequence_change_time, flight_plan_change_sequence, trajectory = fun.get_flight_plan()
    # fun.plot_real_trajectory()
