import os, pickle, argparse
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
import poly_inverse_time_decay as td
from matplotlib.patches import Ellipse
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
        dev_max = abs(np.amax(fp[:, :, 1:4] - traj[:, :, 1:4], axis=1))
        row_num = np.where(dev_max > self.maximum_separation_distance)[0]
        cube, fp, traj = cube[row_num, :, :, :, :], fp[row_num, :, :], traj[row_num, :, :]
        print('Separate {} data based on threshold value {}'.format(fp.shape[0], self.maximum_separation_distance))

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

    def build_graph(self, keep_prob=None):

        with tf.variable_scope("RCBBB", reuse=tf.AUTO_REUSE):

            self.cube = tf.keras.Input(shape=(49, 32, 32, 1))
            self.fp = tf.keras.Input(shape=(50, 3))

            conv1 = tfp.layers.Convolution3DReparameterization(filters=3,
                                                   kernel_size=[1, 5, 5],
                                                   strides=[1, 3, 3],
                                                   padding='valid',
                                                   activation=tf.nn.tanh,
                                                   name='conv1')(self.cube)

            conv2 = tfp.layers.Convolution3DReparameterization(filters=1,
                                                   kernel_size=[1, 3, 3],
                                                   strides=[1, 3, 3],
                                                   padding='same',
                                                   activation=tf.nn.tanh,
                                                   name='conv2')(conv1)

            flat1 = tf.keras.layers.Reshape([self.input_dimension - 1, conv2.shape[-3]*conv2.shape[-2]*conv2.shape[-1]],
                                            name='flat1')(conv2)

            dense1 = tfp.layers.DenseReparameterization(4, activation=tf.nn.tanh, name='dense1')(flat1)
            dense2 = tfp.layers.DenseReparameterization(1, activation=tf.nn.tanh, name='dense2')(dense1)

            # recurrent cell need to be changed
            lstm_in = tf.keras.layers.concatenate([self.fp[:, 1:, :], dense2], axis=-1)

            # single layer lstm
            #lstm_out = tf.keras.layers.LSTM(self.lstm_hidden_dim)(lstm_in)

            lstm_cell = ed.layers.LSTMCellReparameterization(self.lstm_hidden_dim, activation='tanh')
            #lstm_cell = ed.layers.LSTMCellFlipout(self.lstm_hidden_dim, activation='tanh')
            #lstm_cell = tf.keras.layers.LSTMCell(self.lstm_hidden_dim, activation=tf.nn.tanh)
            lstm_out = tf.keras.layers.RNN(cell=lstm_cell, return_sequences=True)(lstm_in)

            dense3 = tfp.layers.DenseReparameterization(int(self.lstm_hidden_dim/2),
                                            activation=tf.nn.tanh,
                                            name='dense3')(lstm_out)

            dense4 = tfp.layers.DenseReparameterization(int(self.lstm_hidden_dim/4),
                                            activation=tf.nn.tanh,
                                            name='dense4')(dense3)

            dense5 = tfp.layers.DenseReparameterization(3, activation=tf.nn.tanh, name='dense5')(dense4)

            self.outputs = dense5

            model = tf.keras.Model(inputs=[self.cube, self.fp], outputs=dense5)

            print(model.summary())

            self.nll = tf.sqrt(tf.reduce_mean((self.traj[:, 1:, :] - self.outputs)**2))

            self.kl = sum(model.losses)/self.fp_train.shape[0]

            self.loss = self.kl + self.nll

    def train_model(self):

        # define variables
        self.fp = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='fp')
        self.cube = tf.placeholder(tf.float32, [None, self.input_dimension-1, 32, 32, 1], name='cube')
        self.traj = tf.placeholder(tf.float32, [None, self.input_dimension, 3], name='traj')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        # load data
        self.load_data()

        # build computational graph
        print('build graph..........')
        self.build_graph()
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RCBBB")

        # lr decay
        global_step = tf.train.get_or_create_global_step()
        learning_rate = td.poly_inverse_time_decay(0.001, global_step, decay_steps=1, decay_rate=0.0001, power=0.75)

        # optimizer choice
        #train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(self.loss, var_list=vars)
        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=vars)
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False).minimize(self.loss, var_list=vars)

        # batch number
        batch_num = int(self.fp_train.shape[0] / self.batch_size)

        # training session
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

        f = open('{}/loss_logs.csv'.format(self.save_dir), 'w')
        f.write('Epoch, ELBO Train, ELBO Test, KL Train, KL Test, NLL Train, NLL Test\n')

        print('start training...........')

        for j in range(self.epoch):
            for i in range(batch_num):
                train_fp_batch = self.fp_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_traj_batch = self.traj_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_cube_batch = self.cube_train[self.batch_size * i:self.batch_size * (i + 1), :, :, :, :]

                _, loss, kl_train, nll_train = self.sess.run([train_op, self.loss, self.kl, self.nll], feed_dict={self.fp: train_fp_batch,
                                                                     self.traj: train_traj_batch,
                                                                     self.cube: train_cube_batch})

                loss_test, kl_test, nll_test = self.sess.run([self.loss, self.kl, self.nll], feed_dict={self.fp: self.fp_test[:self.batch_size, :, :],
                                                                self.traj: self.traj_test[:self.batch_size, :, :],
                                                                self.cube: self.cube_test[:self.batch_size, :, :, :, :]})

                print("Epoch: %d\t Batch: %d\t TrainELBO: %.4f\t KLTrain: %.4f\t NLLTrain: %.4f\t TestELBO: %.4f\t KLTest: %.4f\t NLLTest: %.4f\t" %
                      (j+1, i+1, loss, kl_train, nll_train, loss_test, kl_test, nll_test))

            # record loss values
            f.write("%d, %f, %f, %f, %f, %f, %f\n" % (j+1, loss, loss_test, kl_train, kl_test, nll_train, nll_test))
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
                                                                                            self.cube: test_cube_batch})
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
        np.save('{}/pred_traj_std_lat.npy'.format(self.save_dir), self.pred_traj_std_lat)
        np.save('{}/pred_traj_std_lon.npy'.format(self.save_dir), self.pred_traj_std_lon)
        np.save('{}/pred_traj_std_alt.npy'.format(self.save_dir), self.pred_traj_std_alt)

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
            plt.figure(figsize=(20, 5))
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
        plt.plot(x, f[:, 1], 'blue', label='-ELBO Train')
        plt.plot(x, f[:, 2], 'red', label='-ELBO Test')
        plt.plot(x, f[:, 3], label='KL Train')
        plt.plot(x, f[:, 4], label='KL Test')
        plt.plot(x, f[:, 5], label='NLL Train')
        plt.plot(x, f[:, 6], label='NLL Test')
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
           'test_number': 100,}

    cfg['save_path'] = './Repara_{}_{}_epoch_{}'.format(cfg['date'], cfg['sector'], cfg['epoch'])

    fun = BBB_Traj(cfg)
    fun.train_model()
    fun.test_model(draw_figure=True, pre_trained=False)
    fun.plot_loss()
