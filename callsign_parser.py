import pandas as pd
import numpy as np


class call_sign_parser(object):

    def __init__(self, time, start_row_num, end_row_num):

        self.df = pd.read_csv('data/IFF_USA_' + str(time) + '.csv', skiprows=start_row_num,
                              nrows=end_row_num, names=range(0, 8), index_col=False)
        self.cols = [0, 7]

    def count_rows(self):

        n = sum(1 for line in open('data/IFF_USA_' + str(time) + '.csv'))
        print "loaded " + str(n) + " rows of data"

    def parser(self):

        data = self.df.ix[:, self.cols].replace({'?': np.nan}).dropna()
        call_signs = np.unique(np.asarray(data[data[0] == 3])[:, -1])
        # clean nonsense numbers
        call_signs_cleaned = np.asarray([x for x in call_signs if not x.isdigit()])
        return call_signs_cleaned


if __name__ == '__main__':

    start_row_num = 0
    end_row_num = 1e7

    time = raw_input("Please input the date to fetch data: ")
    print "Creating call_sign_" + str(time) + ".csv"

    fun = call_sign_parser(time, start_row_num, end_row_num)
    #fun.count_rows()
    call_sign = fun.parser()
    with open('call_sign_' + str(time) + '.csv', 'w') as file:
        file.write("\n".join(call_sign))
    print "Done"
