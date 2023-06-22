import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class train_weather_lstm(object):

    def __init__(self, cfg):
        self.lr = cfg['lr']
        self.epoch = cfg['epoch']
        self.input_dimension = cfg['input_dimension']
        self.cube_size = cfg['cube_size']
        self.split_ratio = cfg['split_ratio']
        self.batch_size = cfg['batch_size']
        self.save_dir = cfg['save_dir']

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            for the_file in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

    def load_data(self):

	print("Loading Data................................")
        # get file list
        file_list = sorted(os.listdir('training data/{}/weather data/JFK2LAX_ET'.format(self.input_dimension)))
        data_size = len(file_list)

        # create array to store files
        x_fp = np.empty((data_size, self.input_dimension, 3), dtype=float)
        x_weather = np.empty((data_size, self.input_dimension-1, self.cube_size, self.cube_size), dtype=float)
        y_traj = np.empty((data_size, self.input_dimension, 3), dtype=float)

        # load files and store into one array
        for i in range(data_size):
            x_fp[i, :, :] = np.load('training data/{}/flightplan data/{}'.format(self.input_dimension, file_list[i]))
            x_weather[i, :, :, :] = np.load('training data/{}/weather data/JFK2LAX_ET/{}'.format(self.input_dimension, file_list[i]))
            y_traj[i, :, :] = np.load('training data/{}/trajectory data/{}'.format(self.input_dimension, file_list[i]))

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

        # do train test split use sklearn function
        self.train_x_fp, self.test_x_fp, self.train_x_weather, self.test_x_weather, self.train_y_traj, self.test_y_traj \
            = train_test_split(x_fp, x_weather, y_traj, test_size=self.split_ratio, shuffle=False, random_state=None)

        # only consider 2d case now (longitude and latitude)
	self.train_x_weather = self.train_x_weather[:, :, :, :, :]
	self.test_x_weather = self.test_x_weather[:, :, :, :, :]
        
	self.train_x_fp = self.train_x_fp[:, :, 0:2]
        self.test_x_fp = self.test_x_fp[:, :, 0:2]
        self.train_y_traj = self.train_y_traj[:, :, 0:2]
        self.test_y_traj = self.test_y_traj[:, :, 0:2]

        print("Done loading the data!")

    def conv_lstm_graph(self, x, x_conv, y_true, batch_size):

        # load data
        self.load_data()

        # set dimensions
        _, time_steps, y_dim = x.get_shape().as_list()
        dim_out = x.get_shape().as_list()[-1]
        #dim_hid = dim_out + x.get_shape().as_list()[2] + 4  # 4 is the number of unit in last dense layer of convnet
        dim_hid = dim_out + x.get_shape().as_list()[2]  # no conv layers

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
        #c_t_0 = tf.zeros([batch_size, dim_out], name='c_t_0')
        #h_t_0 = tf.zeros([batch_size, dim_out], name='h_t_0')
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
            h_x = tf.concat([h_t_0, x_t], 1)  # no conv layers

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

    def train_model(self):

        self.x_weather = tf.placeholder(tf.float32, [None, self.input_dimension-1, self.cube_size, self.cube_size, 1])
        self.x_fp = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.y_traj = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        #self.x_weather = tf.placeholder(tf.float32, [None, 9, self.cube_size, self.cube_size, 1])
        #self.x_fp = tf.placeholder(tf.float32, [None, 10, 2])
        #self.y_traj = tf.placeholder(tf.float32, [None, 10, 2])
        batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')

        # build graph
        self.conv_lstm_graph(self.x_fp, self.x_weather, self.y_traj, batch_size)

        # store loss values
        self.train_loss = []
        self.test_loss = []

        # SGD
        train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # batch number
        batch_num = int(self.train_x_fp.shape[0] / self.batch_size)

        print("Start training.")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for j in range(self.epoch):
                for i in range(batch_num):

                    train_x_fp_batch = self.train_x_fp[self.batch_size * i:self.batch_size * (i+1), :, :]
                    train_x_weather_batch = self.train_x_weather[self.batch_size * i:self.batch_size * (i+1), :, :, :, :]
                    train_y_traj_batch = self.train_y_traj[self.batch_size * i:self.batch_size * (i+1), :, :]

                    feed_value_train = {self.x_weather: train_x_weather_batch,
                                        self.x_fp: train_x_fp_batch,
                                        self.y_traj: train_y_traj_batch,
                                        batch_size: self.batch_size, }

                    feed_value_test = {self.x_weather: self.test_x_weather,
                                       self.x_fp: self.test_x_fp,
                                       self.y_traj: self.test_y_traj,
                                       batch_size: self.test_x_fp.shape[0]}

                    [_, loss_train] = sess.run([train_step, self.loss], feed_dict=feed_value_train)

                    loss_test = sess.run(self.loss, feed_dict=feed_value_test)

                    print("Epoch: {} Batch: {} Train Loss: {} Test Loss: {}".format(j+1, i+1, loss_train, loss_test))

                self.train_loss = np.append(self.train_loss, loss_train)
                self.test_loss = np.append(self.test_loss, loss_test)

            save_path = tf.train.Saver().save(sess, '{}/model.ckpt'.format(self.save_dir))
            print("Model saved in path: {}".format(save_path))

        sess.close()
        print("Finish training.")

    def draw_loss(self):

        import matplotlib.pyplot as plt

        xx = np.array(range(len(self.train_loss))) + 1

        plt.plot(xx, self.train_loss, 'k-', linewidth=1.0)
        plt.plot(xx, self.test_loss, linewidth=1.0)

        plt.legend(["Train Loss", "Test Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Loss vs. Epochs')

        plt.savefig('{}/{}.png'.format(self.save_dir, self.save_dir))


if __name__ == '__main__':

    cfg = {'lr': 0.05,
           'epoch': 20,
           'batch_size': 64,
           'input_dimension': 1000,  # number of trajectory points in the data
           'cube_size': 20,  # weather cube size
           'split_ratio': 0.1,  # train test split ratio
           }

    cfg['save_dir'] = './Epoch_{}_Dimension_{}'.format(cfg['epoch'], cfg['input_dimension'])

    fun = train_weather_lstm(cfg)
    fun.train_model()
    fun.draw_loss()
