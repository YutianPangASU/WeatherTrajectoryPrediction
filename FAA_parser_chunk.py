import pandas as pd
import numpy as np
import time


class FAA_Parser(object):

    def __init__(self, call_sign, time, chunk_size):

        self.time = time
        self.call_sign = call_sign
        self.chunk_size = chunk_size

        # specific row numbers to keep
        self.rows = []

        # specific colomn numbers to keep
        # cols = [0, 1, 7, 17]  # flightID, time, flight number, flight plan
        self.cols = [0, 1, 7, 9, 10, 11, 12, 16, 17, 18]  # include lat and lon

        # track point is the array to store trajectories
        # self.track_point = []

    def count_rows(self):

        t0 = time.time()
        n = sum(1 for line in open('data/IFF_USA_' + self.time + '.csv'))
        print "loaded " + str(n) + " rows of data"
        print('Elapsed time : ', time.time() - t0)

    def get_flight_plan(self):

        # chunk number index
        i = 0

        df = pd.read_csv('data/IFF_USA_' + self.time + '.csv', chunksize=self.chunk_size, iterator=True,
                         names=range(0, 19), low_memory=False)

        flight_plan_change_time = []
        flight_plan_change = []
        track_point = np.empty((0, 7))

        for chunk in df:

            i = i + 1
            print "reading chunk number " + str(i)

            # self.rows.extend(chunk.index[chunk[7] == self.call_sign])
            # if self.rows.__len__() != 0:
            #     self.rows = np.asarray(self.rows) - self.chunk_size * (i - 1)
            # self.rows = list(self.rows)
            self.rows = []
            self.rows.extend(chunk.index[chunk[7] == self.call_sign])

            # restore the data from dataframe
            data = chunk.ix[self.rows][self.cols]

            # clear ? and nan values in the dataframe
            # data = data.replace({'?': np.nan}).dropna()

            # divide data into flight plan and track point
            track_point_chunk = np.asarray(data[data[0] == 3])
            flight_plan_chunk = data[data[0] == 4]

            # flight_plan = list(set(flight_plan[17]))  # remove duplicate in flight plan
            fp, fp_indices = np.unique(flight_plan_chunk[17], return_index=True)
            fp_indices = np.sort(fp_indices)

            flight_plan_chunk = flight_plan_chunk.values[fp_indices]
            flight_plan_change_time_chunk = flight_plan_chunk[:, 1]
            flight_plan_change_chunk = flight_plan_chunk[:, -1]

            track_point_chunk = np.delete(track_point_chunk, [0, 2, 8], axis=1)  # col1:unix time, col2:lon, col3:lat
            track_point_chunk[:, [1, 2]] = track_point_chunk[:, [2, 1]]  # swap lat and lon

            if flight_plan_change_chunk.size != 0:
                flight_plan_change.append(flight_plan_change_chunk)
                flight_plan_change_time.append(flight_plan_change_time_chunk)
            if track_point_chunk.size != 0:
                track_point = np.concatenate((track_point, track_point_chunk), axis=0)

        return flight_plan_change_time, flight_plan_change, track_point


class save_files(object):

    def __init__(self, list, filename, time):

        self.list = list
        self.filename = filename
        self.time = time

    def save_trx(self):

        for j in range(len(self.list)):

            f = open('./cache/' + time + "_" + self.filename + "_" + str(j) + '.trx', 'wb')
            f.write("TRACK_TIME 1121238067\n\n")

            fm = open('./cache/' + time + "_" + self.filename + "_" + str(j) + '_mfl.trx', 'wb')

            for i in range(len(self.list[j])):
                f.write("TRACK A" + str(i) + " ALOR1 370500N 1030900W 470 360 0 ZAB ZAB71\n")
                f.write("FP_ROUTE " + str(self.list[j][i]) + "\n\n")
                fm.write("A" + str(i) + " 400\n")

            f.close()
            fm.close()

    def save_csv(self):

        my_df = pd.DataFrame(self.list)
        my_df.to_csv("./traj_csv/" + time + "_" + self.filename + '.csv', index=False,
                     header=False)


if __name__ == '__main__':

    # call_sign = 'AAL717'
    # time = '20170406'

    call_sign = raw_input("Please input the flight call sign: ")
    call_sign = str(call_sign)
    print "The flight call sign is " + call_sign

    time = raw_input("Please input the date for data file: ")
    time = str(time)
    print "Reading FAA_USA_" + time + ".csv"

    chunk_size = 1e6
    print "Chunk size is " + str(int(chunk_size))

    fun = FAA_Parser(call_sign, time, chunk_size)
    flight_plan_sequence_change_time, flight_plan_change_sequence, trajectory = fun.get_flight_plan()
    save_files(trajectory, call_sign, time).save_csv()
    save_files(flight_plan_change_sequence, call_sign, time).save_trx()
