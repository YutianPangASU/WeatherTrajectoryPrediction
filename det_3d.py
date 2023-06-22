import os, pickle, argparse
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
import poly_inverse_time_decay as td
#from matplotlib.patches import Ellipse
import edward2 as ed
from sklearn.model_selection import train_test_split
sns.set()
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.disable_v2_behavior()


class BBB_Traj(object):
    def __init__(self, cfg):
        self.date = cfg['date']
        self.sector = cfg['sector']
        self.input_dimension = cfg['input_dimension']
        self.batch_size = cfg['batch_size']
        self.lstm_hidden_dim = cfg['lstm_hidden_dim']
        self.epoch = cfg['epoch']
        self.test_ratio = cfg['test_ratio']
        self.train_keep_prob = cfg['train_keep_prob']
        self.test_keep_prob = cfg['test_keep_prob']
        self.save_dir = cfg['save_path']
        self.figure_number = cfg['figure_number']
        self.maximum_separation_distance = cfg['maximum_separation_distance']
        self.test_number = cfg['test_number']

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # else:
        #     for the_file in os.listdir(self.save_dir):
        #         file_path = os.path.join(self.save_dir, the_file)
        #         try:
        #             if os.path.isfile(file_path):
        #                 os.unlink(file_path)
        #         except Exception as e:
        #             print(e)
        seed = 1
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

    def load_data(self):
        print('loading data.........')
        fp = pickle.load(open('TP-data/{0}_{1}/FP_{0}_{1}.p'.format(self.sector, self.date), 'rb'))
        traj = pickle.load(open('TP-data/{0}_{1}/TRACKS_{0}_{1}.p'.format(self.sector, self.date), 'rb'))
        cube = pickle.load(open('TP-data/{0}_{1}/WEATHER_CUBE_{0}_{1}.p'.format(self.sector, self.date), 'rb'))

        cube_arr = []
        for key, arr in cube.items():
            cube_arr.append(np.asarray(arr))
        cube = np.expand_dims(np.asarray(cube_arr), axis=4)

        fp_arr = []
        for key, arr in fp.items():
            fp_arr.append(arr.reset_index().values)
        fp = np.asarray(fp_arr)

        traj_arr = []
        for key, arr in traj.items():
            traj_arr.append(arr.reset_index().values)
        traj = np.asarray(traj_arr)

        # separate the data based on deviations
        dev_max = abs(np.amax(fp[:, :, 1:3] - traj[:, :, 1:3], axis=1))
        row_num = np.unique(np.where(dev_max > self.maximum_separation_distance)[0])
        cube, fp, traj = cube[row_num, :, :, :, :], fp[row_num, :, :], traj[row_num, :, :]
        print('Separate {} data based on threshold value {}'.format(fp.shape[0], self.maximum_separation_distance))

        # filter data relevant to weather with threshold of ET values
        echo_mean = np.mean(cube, axis=(1, 2, 3, 4))
        echo_max = np.max(cube, axis=(1, 2, 3, 4))
        list1 = np.array(np.where(echo_max > 25000)).tolist()[0]
        list2 = np.array(np.where(echo_mean > 1000)).tolist()[0]
        idx = list(set(list1) & set(list2))
        fp, traj, cube = fp[idx], traj[idx], cube[idx]
        print('Separated {} data based on the weather threshold'.format(fp.shape[0]))

        # get sector range in data
        self.lat_max = np.max([traj[:, :, 1], fp[:, :, 1]])
        self.lat_min = np.min([traj[:, :, 1], fp[:, :, 1]])
        self.lon_min = np.min([fp[:, :, 2], traj[:, :, 2]])
        self.lon_max = np.max([fp[:, :, 2], traj[:, :, 2]])
        self.alt_min = np.min([fp[:, :, 3], traj[:, :, 3]])
        self.alt_max = np.max([fp[:, :, 3], traj[:, :, 3]])
        weather_max = np.max(cube)

        # normalization
        cube = cube/weather_max

        fp[:, :, 1], traj[:, :, 1] = (fp[:, :, 1] - self.lat_min) / (self.lat_max - self.lat_min), \
                                     (traj[:, :, 1] - self.lat_min) / (self.lat_max - self.lat_min)# normalize latitude

        fp[:, :, 2], traj[:, :, 2] = (fp[:, :, 2] - self.lon_min) / (self.lon_max - self.lon_min), \
                                     (traj[:, :, 2] - self.lon_min) / (self.lon_max - self.lon_min) # normalize longitude

        fp[:, :, 3], traj[:, :, 3] = (fp[:, :, 3] - self.alt_min) / (self.alt_max - self.alt_min), \
                                     (traj[:, :, 3] - self.alt_min) / (self.alt_max - self.alt_min)  # normalize altitude

        # special note for the last dimension: time, lat, lon, alt
        fp = fp[:, :, 1:4]
        traj = traj[:, :, 1:4]

        # train test split
        self.fp_train, self.fp_test, self.traj_train, self.traj_test, self.cube_train, self.cube_test \
            = train_test_split(fp, traj, cube, test_size=self.test_ratio, shuffle=False, random_state=None)

        print('finish loading data')

    def inverse_normalization(self, tensor):

        delta_lat = self.lat_max - self.lat_min
        delta_lon = self.lon_max - self.lon_min
        delta_alt = self.alt_max - self.alt_min

        tensor[:, :, 0] = tensor[:, :, 0] * delta_lat + self.lat_min
        tensor[:, :, 1] = tensor[:, :, 1] * delta_lon + self.lon_min
        tensor[:, :, 2] = tensor[:, :, 2] * delta_alt + self.alt_min

        return tensor

    def build_graph(self, keep_prob):

        with tf.variable_scope("RCNN", reuse=tf.AUTO_REUSE):
            #
            # kernel = _create_weights([1, 5, 5, 1, 3])  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
            # conv = _create_conv2d(self.cube, kernel)  # create the conv layer
            # bias = _create_bias([3])  # out_channels
            # preactivation = tf.nn.bias_add(conv, bias)  # add bias
            # conv1 = tf.nn.relu(preactivation, name='conv1')

            conv1 = tf.layers.conv3d(self.cube, filters=3, kernel_size=[1, 5, 5], strides=[1, 3, 3], padding='valid',
                                     activation=tf.nn.tanh, name='conv1')

            conv2 = tf.layers.conv3d(conv1, filters=1, kernel_size=[1, 3, 3], strides=[1, 3, 3], padding='same',
                                     activation=tf.nn.tanh, name='conv2')

            flat1 = tf.reshape(conv2, [-1, self.input_dimension - 1, 4 * 4 * 1], name='flat1')

            dense1 = tf.layers.dense(flat1, 4, activation=tf.nn.tanh, name='dense1')

            dense2 = tf.layers.dense(dense1, 1, activation=tf.nn.tanh, name='dense2')

            lstm_in = tf.concat([self.fp[:, 1:, :], dense2], -1)

            # # multi layers lstm
            # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [2, 8, 64, 8, 2]]
            # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            # multi_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(multi_rnn_cell,
            #                                                input_keep_prob=keep_prob,
            #                                                output_keep_prob=keep_prob,
            #                                                state_keep_prob=keep_prob,
            #                                                variational_recurrent=True,
            #                                                dtype=tf.float32,
            #                                                input_size=lstm_in.get_shape()[-1],)
            # self.outputs, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=lstm_in, dtype=tf.float32)

            # single layer lstm
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim, activation=tf.nn.tanh)
            init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            lstm_out, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, initial_state=init_state)

            dense3 = tf.layers.dense(lstm_out, self.lstm_hidden_dim/2, activation=tf.nn.tanh, name='dense3')

            dense4 = tf.layers.dense(dense3, self.lstm_hidden_dim/4, activation=tf.nn.tanh, name='dense4')

            dense5 = tf.layers.dense(dense4, 3, activation=tf.nn.tanh, name='dense5')
            self.outputs = dense5

            #self.outputs = tf.nn.dropout(dense5, keep_prob)

            # zero initialization doesn't work
            # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            # lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, dtype=tf.float32)

            #concatenate the 1st point in outputs [Nx50x2]
            #self.outputs = tf.concat([tf.expand_dims(self.fp[:, 0, :], 1), self.outputs], 1)
            self.loss = tf.sqrt(tf.reduce_mean((self.traj[:, 1:, :] - self.outputs)**2))

            # self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.outputs[:, :, 0] - self.traj[:, 1:, 0]) +
            #                                    tf.square(self.outputs[:, :, 1] - self.traj[:, 1:, 1]) +
            #                                    tf.square(self.outputs[:, :, 2] - self.traj[:, 1:, 2])), axis=None)

    def train_model(self):

        # define variables
        self.fp = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='fp')
        self.cube = tf.placeholder(tf.float32, [None, self.input_dimension-1, 32, 32, 1], name='cube')
        self.traj = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='traj')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        # load data
        self.load_data()

        # build computational graph
        print('build graph..........')
        self.build_graph(self.keep_prob)
        #vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DROPOUT")
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RCNN")

        # lr decay
        global_step = tf.train.get_or_create_global_step()
        learning_rate = td.poly_inverse_time_decay(0.001, global_step, decay_steps=1, decay_rate=0.0001, power=0.75)

        # optimizer choice
        #train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(self.loss, var_list=vars)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=vars)
        #train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False).minimize(self.loss, var_list=vars)

        # batch number
        batch_num = int(self.fp_train.shape[0] / self.batch_size)

        # training session
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

        f = open('{}/loss_logs.csv'.format(self.save_dir), 'w')
        f.write('Epoch, Loss Train, Loss Test\n')

        print('start training...........')

        for j in range(self.epoch):
            for i in range(batch_num):
                train_fp_batch = self.fp_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_traj_batch = self.traj_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_cube_batch = self.cube_train[self.batch_size * i:self.batch_size * (i + 1), :, :, :, :]

                _, loss_train = self.sess.run([train_op, self.loss], feed_dict={self.fp: train_fp_batch,
                                                                                self.traj: train_traj_batch,
                                                                                self.cube: train_cube_batch,
                                                                                self.keep_prob: self.train_keep_prob})

                loss_test = self.sess.run(self.loss, feed_dict={self.fp: self.fp_test[:self.batch_size, :, :],
                                                                  self.traj: self.traj_test[:self.batch_size, :, :],
                                                                  self.cube: self.cube_test[:self.batch_size, :, :, :, :],
                                                                  self.keep_prob: self.train_keep_prob})

                print("Epoch: %d\t Batch: %d\t TrainLoss: %.4f\t TestLoss: %.4f\t" % (j+1, i+1, loss_train, loss_test))

            # record loss values
            f.write("%d, %f, %f\n" % (j+1, loss_train, loss_test))
        f.close()

        save_path = tf.train.Saver().save(self.sess, '{}/model.ckpt'.format(self.save_dir))
        print("Model saved in path: {}".format(save_path))

        #sess.close()
        print("Finish training.")

    def test_model(self, draw_figure=False, pre_trained=False):
        if pre_trained is True:
            # define variables
            self.fp = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='fp')
            self.cube = tf.placeholder(tf.float32, [None, self.input_dimension - 1, 32, 32, 1], name='cube')
            self.traj = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='traj')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

            # load data
            self.load_data()

            # build computational graph
            print('build graph..........')
            self.build_graph(self.keep_prob)

        # batch number
        self.batch_num = int(self.fp_test.shape[0] / self.batch_size)

        # testing number loop
        for test_num in range(self.test_number):

            if pre_trained is True:
                self.sess = tf.Session()

                # Restore latest checkpoint
                tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint('./{}/'.format(self.save_dir)))

            print("#########################################################")
            print('start testing #{}...........'.format(test_num+1))

            for i in range(self.batch_num):
                test_fp_batch = self.fp_test[self.batch_size * i:self.batch_size * (i + 1), :, :]
                test_traj_batch = self.traj_test[self.batch_size * i:self.batch_size * (i + 1), :, :]
                test_cube_batch = self.cube_test[self.batch_size * i:self.batch_size * (i + 1), :, :, :, :]

                pred_traj_batch, test_loss = self.sess.run([self.outputs, self.loss], feed_dict={self.fp: test_fp_batch,
                                                                                            self.traj: test_traj_batch,
                                                                                            self.cube: test_cube_batch,
                                                                                            self.keep_prob: self.test_keep_prob})
                if i==0:
                    pred_traj = pred_traj_batch
                else:
                    pred_traj = tf.concat([pred_traj, pred_traj_batch], axis=0, name='stack_test')

                print('Batch: {} Testing Loss: {}'.format(i, test_loss))

            # do inverse normalization to the prediction
            test_pred_traj = self.inverse_normalization(pred_traj.eval(session=self.sess))

            #self.sess.close()

            # concatenate the testing result
            if test_num == 0:
                self.pred_traj = np.expand_dims(test_pred_traj, axis=0)
            else:
                self.pred_traj = np.concatenate([self.pred_traj, np.expand_dims(test_pred_traj, axis=0)], axis=0)

        # inverse normalization for traj and fp
        self.traj_test, self.fp_test = self.inverse_normalization(self.traj_test), self.inverse_normalization(self.fp_test)

        # calculate mean and variance of predicted trajectory
        self.pred_traj_mean = np.mean(self.pred_traj, axis=0)
        self.pred_traj_std_lat = np.std(self.pred_traj[:, :, :, 0], axis=0)  # y
        self.pred_traj_std_lon = np.std(self.pred_traj[:, :, :, 1], axis=0)  # x
        self.pred_traj_std_alt = np.std(self.pred_traj[:, :, :, 2], axis=0)  # z

        # calculate variance
        self.dev_ori = self.traj_test[:self.batch_size*self.batch_num, 1:, :] - self.fp_test[:self.batch_size*self.batch_num, 1:, :]
        self.dev_new = self.traj_test[:self.batch_size*self.batch_num, 1:, :] - self.pred_traj_mean
        np.save('{}/traj_test.npy'.format(self.save_dir), self.traj_test)
        np.save('{}/fp_test.npy'.format(self.save_dir), self.fp_test)
        np.save('{}/pred_traj_mean.npy'.format(self.save_dir), self.pred_traj_mean)

        self.l2_ori = np.sum(np.sum(self.dev_ori[:, :, :2] ** 2, axis=2), axis=1) / 49
        self.l2_new = np.sum(np.sum(self.dev_new[:, :, :2] ** 2, axis=2), axis=1) / 49
        np.save('det_{}_l2.npy'.format(self.sector), self.l2_new)
        np.save('testset_{}_l2.npy'.format(self.sector), self.l2_ori)

        percent_reduced = len(np.where(self.l2_new / self.l2_ori < 1)[0]) / float(len(self.l2_ori))
        print("l2-norm: {} percent of tracks is reduced.".format(percent_reduced))

        dev_reduced = np.var(self.l2_new[np.where(self.l2_new / self.l2_ori < 1)]) / np.var(self.l2_ori[np.where(self.l2_new / self.l2_ori < 1)])
        print("Variance is reduced by {}".format(1 - dev_reduced))

        std_reduced = (np.std(self.l2_new) - np.std(self.l2_ori)) / np.std(self.l2_ori)
        print("Total std (without altitude) is reduced by {}".format(std_reduced))

        # save results
        f = open('{}/statistical_result'.format(self.save_dir), 'w')
        f.write("l2-norm: {} percent of tracks is reduced.".format(percent_reduced))
        f.write("Variance is reduced by {}".format(1 - dev_reduced))
        f.write("Total std (without altitude) is reduced by {}".format(std_reduced))
        f.close()

        if draw_figure is True:
            print('drawing......................')
            self.draw_figure()
            print('drawing finished')
        print("#########################################################")

    def draw_figure(self):

        # draw combined dist plot
        sns.distplot(self.l2_ori, color='blue', kde=False, label='original', norm_hist=True)
        sns.distplot(self.l2_new, color='green', kde=False, label='predicted', norm_hist=True)
        plt.title('l-2 norm')
        plt.xlim([0, 40])
        plt.legend()
        plt.xlabel("l-2 norm of combined deviation")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig("{}/combine.png".format(self.save_dir))
        plt.close()

        # draw separate plot
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 3, 1)
        sns.distplot(self.dev_ori[:, :, 0].ravel(), color='green', kde=False, label='original', norm_hist=True)
        sns.distplot(self.dev_new[:, :, 0].ravel(), color='blue', kde=False, label='predicted', norm_hist=True)
        plt.title('Longitude Deviation')
        plt.legend()
        plt.xlim([-1, 1])
        plt.xlabel("Deviation in degree")
        plt.ylabel("Density")

        plt.subplot(1, 3, 2)
        sns.distplot(self.dev_new[:, :, 1].ravel(), color='green', kde=False, label='predicted', norm_hist=True)
        sns.distplot(self.dev_ori[:, :, 1].ravel(), color='blue', kde=False, label='original', norm_hist=True)
        plt.title('Latitude Deviation')
        plt.legend()
        plt.xlim([-1, 1])
        plt.xlabel("Deviation in degree")
        plt.ylabel("Density")

        plt.subplot(1, 3, 3)
        sns.distplot(self.dev_new[:, :, 2].ravel(), color='green', kde=False, label='predicted', norm_hist=True)
        sns.distplot(self.dev_ori[:, :, 2].ravel(), color='blue', kde=False, label='original', norm_hist=True)
        plt.title('Altitude Deviation')
        plt.legend()
        plt.xlim([-1, 1])
        plt.xlabel("Deviation in 100 feet")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.savefig("{}/separate.png".format(self.save_dir))
        plt.close()

        # draw plot for each data point
        for idx in range(self.figure_number):
            x = np.linspace(start=1, stop=49, num=49)
            plt.figure(figsize=(20, 5), facecolor='white')
            plt.title('{} {} Testing result: {}'.format(self.date, self.sector, idx))

            plt.subplot(1, 3, 1)
            plt.ylabel('Latitude')
            plt.xlabel('Time Step')
            plt.ylim([self.lat_min, self.lat_max])
            plt.plot(x, self.pred_traj_mean[idx, :, 0], '*', label='mean prediction')
            plt.plot(x, self.fp_test[idx, 1:, 0], '*', label='flight plan')
            plt.plot(x, self.traj_test[idx, 1:, 0], '*', label='ground truth')
            plt.fill_between(x, self.pred_traj_mean[idx, :, 0] + 2*self.pred_traj_std_lat[idx, :],
                             self.pred_traj_mean[idx, :, 0] - 2*self.pred_traj_std_lat[idx, :], alpha=0.2)
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.ylabel('Longitude')
            plt.xlabel('Time Step')
            plt.ylim([self.lon_min, self.lon_max])
            plt.plot(x, self.pred_traj_mean[idx, :, 1], '*', label='mean prediction')
            plt.plot(x, self.fp_test[idx, 1:, 1], '*', label='flight plan')
            plt.plot(x, self.traj_test[idx, 1:, 1], '*', label='ground truth')
            plt.fill_between(x, self.pred_traj_mean[idx, :, 1] + 2*self.pred_traj_std_lon[idx, :],
                             self.pred_traj_mean[idx, :, 1] - 2*self.pred_traj_std_lon[idx, :], alpha=0.2)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.ylabel('Altitude')
            plt.xlabel('Time Step')
            #plt.ylim([self.alt_min, self.alt_max])
            plt.plot(x, self.pred_traj_mean[idx, :, 2], '*', label='mean prediction')
            plt.plot(x, self.fp_test[idx, 1:, 2], '*', label='flight plan')
            plt.plot(x, self.traj_test[idx, 1:, 2], '*', label='ground truth')
            plt.fill_between(x, self.pred_traj_mean[idx, :, 2] + 2*self.pred_traj_std_alt[idx, :],
                             self.pred_traj_mean[idx, :, 2] - 2*self.pred_traj_std_alt[idx, :], alpha=0.2)
            plt.legend()

            plt.savefig("{}/{}.png".format(self.save_dir, idx))
            plt.close()

    def plot_loss(self):
        f = np.loadtxt(open('{}/loss_logs.csv'.format(self.save_dir), "rb"), delimiter=",", skiprows=1)
        x = range(len(f))
        plt.figure()
        plt.title('Training Process')
        plt.plot(x, f[:, 1], 'blue', label='Loss Train')
        plt.plot(x, f[:, 2], 'red', label='Loss Test')
        plt.legend()
        #plt.show()
        plt.savefig("{}/training_loss.png".format(self.save_dir))


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sector", required=True,
                    help="dataset to choose: ZID/ZTL/ZOB/ZNY/ZDC")
    ap.add_argument("-e", "--epochs", required=True,
                    help="number of training epochs")
    args = vars(ap.parse_args())

    cfg = {'date': 20190624,
           'sector': args['sector'],
           'input_dimension': 50,
           'batch_size': 256,
           'lstm_hidden_dim': 64,
           'epoch': int(args['epochs']),
           'test_ratio': 0.2,
           'train_keep_prob': 0.5,
           'test_keep_prob': 0.5,
           'figure_number': 100,
           'maximum_separation_distance': 0,
           'test_number': 100, }

    cfg['save_path'] = './Det_{}_{}_epoch_{}'.format(cfg['date'], cfg['sector'], cfg['epoch'])

    fun = BBB_Traj(cfg)
    fun.train_model()
    fun.test_model(draw_figure=True, pre_trained=False)
    fun.plot_loss()
