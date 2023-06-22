import os
import numpy as np
import tensorflow as tf


class test_weather_lstm(object):

    def __init__(self, cfg):
        self.input_dimension = cfg['input_dimension']
        self.cube_size = cfg['cube_size']
        self.save_dir = cfg['save_dir']

    def load_data(self):
        # get file list
        self.file_list = sorted(os.listdir('training data/{}/weather data/JFK2LAX_ET'.format(self.input_dimension)))
        data_size = len(self.file_list)

        # create array to store files
        x_fp = np.empty((data_size, self.input_dimension, 3), dtype=float)
        x_weather = np.empty((data_size, self.input_dimension-1, self.cube_size, self.cube_size), dtype=float)
        y_traj = np.empty((data_size, self.input_dimension, 3), dtype=float)

        # load files and store into one array
        for i in range(data_size):
            x_fp[i, :, :] = np.load('training data/{}/flightplan data/{}'.format(self.input_dimension, self.file_list[i]))
            x_weather[i, :, :, :] = np.load('training data/{}/weather data/JFK2LAX_ET/{}'.format(self.input_dimension, self.file_list[i]))
            y_traj[i, :, :] = np.load('training data/{}/trajectory data/{}'.format(self.input_dimension, self.file_list[i]))

        # data normalization
        lat_max = 53.8742945085336
        lat_min = 19.35598953632181
        lon_min = -134.3486134307298
        lon_max = -61.65138656927017

        x_fp[:, :, 0] = (x_fp[:, :, 0] - lat_min) / (lat_max - lat_min)  # normalize lat
        x_fp[:, :, 1] = (x_fp[:, :, 1] - lon_min) / (lon_max - lon_min)  # normalize lon

        y_traj[:, :, 0] = (y_traj[:, :, 0] - lat_min) / (lat_max - lat_min)  # normalize lat
        y_traj[:, :, 1] = (y_traj[:, :, 1] - lon_min) / (lon_max - lon_min)  # normalize lon

        # normalize weather cubes, clip to 0
        x_weather[x_weather < 0] = 0
        x_weather = x_weather/np.amax(x_weather)
        x_weather = np.expand_dims(x_weather, axis=4)

        # only consider 2d case now
        self.valid_x_fp = x_fp[:, :, 0:2]
        self.valid_y_traj = y_traj[:, :, 0:2]
        self.valid_weather = x_weather[:, :, :, :, :]

        print("Done loading the validation data.")

    def inverse_normalization(self, tensor):

        lat_max = 53.8742945085336
        lat_min = 19.35598953632181
        lon_min = -134.3486134307298
        lon_max = -61.65138656927017

        delta_lat = lat_max - lat_min
        delta_lon = lon_max - lon_min

        tensor[:, :, 0] = tensor[:, :, 0] * delta_lat + lat_min
        tensor[:, :, 1] = tensor[:, :, 1] * delta_lon + lon_min

        return tensor

    def conv_lstm_graph(self, x, x_conv, y_true, batch_size):

        # load data
        self.load_data()

        # set dimensions
        _, time_steps, y_dim = x.get_shape().as_list()
        dim_out = x.get_shape().as_list()[-1]
        #dim_hid = dim_out + x.get_shape().as_list()[2] + 4  # 4 is the number of unit in last dense layer of convnet
        dim_hid = dim_out + x.get_shape().as_list()[2]  # 4 is the number of unit in last dense layer of convnet

        # build w and b tensors
        w_f = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w1')
        b_f = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b1')
        w_i = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w2')
        b_i = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b2')
        w_c = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w3')
        b_c = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b3')
        w_o = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w4')
        b_o = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b4')

        # initial conditions
        # c_t_0 = tf.zeros([batch_size, dim_out], name='c_t_0')
        # h_t_0 = tf.zeros([batch_size, dim_out], name='h_t_0')
        c_t_0 = y_true[:, 0, :]
        h_t_0 = y_true[:, 0, :]

        x_t = x[:, 0, :]
        self.y_pred = tf.expand_dims(x[:, 0, :], axis=1)

        for t in range(time_steps-1):

            # convnet layers
            x_conv1 = tf.layers.conv2d(x_conv[:, t, :, :],
                                       filters=2,
                                       strides=2,
                                       kernel_size=6,
                                       padding='valid',
                                       activation=tf.nn.relu,
                                       name='conv1',
                                       reuse=tf.AUTO_REUSE)

            x_conv2 = tf.layers.conv2d(x_conv1,
                                       filters=4,
                                       strides=2,
                                       kernel_size=3,
                                       padding='valid',
                                       activation=tf.nn.relu,
                                       name='conv2',
                                       reuse=tf.AUTO_REUSE)

            x_flat1 = tf.reshape(x_conv2, [-1, 3*3*4])
            x_fc1 = tf.layers.dense(x_flat1, 16, activation=tf.nn.relu, name='fc1', reuse=tf.AUTO_REUSE)
            x_fc2 = tf.layers.dense(x_fc1, 4, activation=tf.nn.relu, name='fc2', reuse=tf.AUTO_REUSE)

            # x_t_2 = x_t * x_fc2
            # h_x = tf.concat([h_t_0, x_t_2], 1)

            # concatenate hidden tensor and x
            #h_x = tf.concat([h_t_0, x_t, x_fc2], 1)
            h_x = tf.concat([h_t_0, x_t], 1)

            # compute three gates
            f_t = tf.sigmoid(tf.nn.xw_plus_b(h_x, w_f, b_f))
            i_t = tf.sigmoid(tf.nn.xw_plus_b(h_x, w_i, b_i))
            o_t = tf.sigmoid(tf.nn.xw_plus_b(h_x, w_o, b_o))

            # compute cell tensor
            c_t_hat = tf.nn.tanh(tf.nn.xw_plus_b(h_x, w_c, b_c))
            c_t = f_t * c_t_0 + i_t * c_t_hat

            # hidden tensor
            h_t = o_t * tf.nn.tanh(c_t)

            # update parameters
            h_t_0 = h_t
            c_t_0 = c_t

            x_t = x[:, t+1, :]

            self.y_pred = tf.concat([self.y_pred, tf.expand_dims(h_t, axis=1)], axis=1)

        self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.y_pred[:, :, 0] - y_true[:, :, 0]) +
                                           tf.square(self.y_pred[:, :, 1] - y_true[:, :, 1])), axis=None)

    def valid_model(self):

        self.x_weather = tf.placeholder(tf.float32, [None, self.input_dimension-1, self.cube_size, self.cube_size, 1])
        self.x_fp = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.y_traj = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
	#self.x_weather = tf.placeholder(tf.float32, [None, 9, self.cube_size, self.cube_size, 1])
        #self.x_fp = tf.placeholder(tf.float32, [None, 10, 2])
        #self.y_traj = tf.placeholder(tf.float32, [None, 10, 2])
        valid_size = tf.placeholder(dtype=tf.int32, name='batch_size')

        # build graph
        self.conv_lstm_graph(self.x_fp, self.x_weather, self.y_traj, valid_size)

        print("Start validation.")

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # session initialization
            sess.run(tf.global_variables_initializer())

            # Restore latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint('./{}/'.format(self.save_dir)))

            feed_value_validation = {self.x_weather: self.valid_weather,
                                     self.x_fp: self.valid_x_fp,
                                     self.y_traj: self.valid_y_traj,
                                     valid_size: self.valid_weather.shape[0], }

            self.y_pred = sess.run(self.y_pred, feed_dict=feed_value_validation)

        self.y_pred = self.inverse_normalization(self.y_pred)
        self.y_true = self.inverse_normalization(self.valid_y_traj)

        sess.close()

        print("Finish validation.")

    def plot_results(self):

        import matplotlib.pyplot as plt
        from sklearn.metrics.pairwise import euclidean_distances

        start = [40.65046, -73.79619]
        destination = [33.94004, -118.40546]
        training_fp = self.inverse_normalization(self.valid_x_fp)

        print('Plotting..................................')
        for i in range(self.y_pred.shape[0]):

            plt.figure(i)

            # fix plot limit
            #plt.xlim(33, 46)
            #plt.ylim(-120, -70)

            pred_traj = self.y_pred[i, :, :]
            true_traj = self.y_true[i, :, :]

            #pred_traj = pred_traj[10:, :]

            # pred_traj = np.insert(pred_traj, [0], [start], axis=0)
            # pred_traj = np.insert(pred_traj, [-1], [destination], axis=0)
            #
            # true_traj = np.insert(true_traj, [0], [start], axis=0)
            # true_traj = np.insert(true_traj, [-1], [destination], axis=0)

            # from sklearn.metrics.pairwise import euclidean_distances
            # distance_matrix = euclidean_distances(pred_traj, pred_traj)
            # import seaborn as sns
            # ax = plt.axes()
            # sns.heatmap(distance_matrix, cmap="YlGnBu", ax=ax)
            # ax.set_title('Distance between each point')
            # ax.set_label("Points")
            # plt.show()

            plt.plot(training_fp[i, :, 0], training_fp[i, :, 1])

            plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r*')
            #plt.plot(pred_traj[:, 0], pred_traj[:, 1])
            plt.plot(true_traj[:, 0], true_traj[:, 1])

            plt.legend(['train_fp', 'predicted', 'true'])
            plt.title('Test on Validation Data')
            plt.savefig('./Epoch_{}_Dimension_{}/{}.png'.format(cfg['epoch'], cfg['input_dimension'], self.file_list[i]))
            plt.close(i)  # close the current figure

        print("Done.")


if __name__ == '__main__':

    cfg = {'input_dimension': 1000,  # number of trajectory points in the data
           'cube_size': 20,  # weather cube size
           'epoch': 20,
           }

    cfg['save_dir'] = './Epoch_{}_Dimension_{}'.format(cfg['epoch'], cfg['input_dimension'])

    fun = test_weather_lstm(cfg)
    fun.valid_model()
    fun.plot_results()
