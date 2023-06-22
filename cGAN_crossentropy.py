import os
#import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.stats as stats
from zodbpickle import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.model_selection import train_test_split
sns.set()

'''
What I cannot create, I do not understand. -- Richard Feynman

'''


class cGAN_Traj(object):
    def __init__(self, cfg):
        self.date = cfg['date']
        self.sector = cfg['sector']
        self.input_dimension = cfg['input_dimension']
        self.batch_size = cfg['batch_size']
        self.lstm_hidden_dim = cfg['lstm_hidden_dim']
        self.epoch = cfg['epoch']
        self.test_ratio = cfg['test_ratio']
        self.generator_steps = cfg['generator_steps']
        self.discriminator_steps = cfg['discriminator_steps']
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

    def load_data(self):
        print('loading data.........')
        fp = pickle.load(open('FP_{}_{}.p'.format(self.sector, self.date), 'rb'))
        traj = pickle.load(open('TRACKS_{}_{}.p'.format(self.sector, self.date), 'rb'))
        gc = pickle.load(open('GC_{}_{}.p'.format(self.sector, self.date), 'rb'))
        cube = pickle.load(open('WEATHER_CUBE_{}_{}.p'.format(self.sector, self.date), 'rb'))

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

        gc_arr = []
        for key, arr in gc.items():
            gc_arr.append(arr)
        gc = np.asarray(gc_arr)

        # separate the data based on deviations
        dev_max = abs(np.amax(fp[:, :, 1:3] - traj[:, :, 1:3], axis=1))
        row_num = np.where(dev_max > self.maximum_separation_distance)[0]
        cube, fp, traj, gc = cube[row_num, :, :, :, :], fp[row_num, :, :], traj[row_num, :, :], gc[row_num, :, :]
        print('Separate {} data based on threshold value {}'.format(fp.shape[0], self.maximum_separation_distance))

        # generate gaussian noise for generator  need truncated normal between [0, 1]
        # self.z = np.random.normal(loc=0.0, scale=1.0, size=[fp.shape[0], self.input_dimension, 2])
        lower, upper = 0, 1
        mu, sigma = 0.5, 1.0
        z = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=[fp.shape[0], self.input_dimension, 2])

        # get sector range in data
        self.lat_max = np.max([traj[:, :, 1], fp[:, :, 1], gc[:, :, 1]])
        self.lat_min = np.min([traj[:, :, 1], fp[:, :, 1], gc[:, :, 1]])
        self.lon_min = np.min([fp[:, :, 2], traj[:, :, 2], gc[:, :, 0]])
        self.lon_max = np.max([fp[:, :, 2], traj[:, :, 2], gc[:, :, 0]])
        weather_max = np.max(cube)

        # normalization
        cube = cube/weather_max

        fp[:, :, 1], traj[:, :, 1], gc[:, :, 1] = (fp[:, :, 1] - self.lat_min) / (self.lat_max - self.lat_min), \
                                                  (traj[:, :, 1] - self.lat_min) / (self.lat_max - self.lat_min), \
                                                  (gc[:, :, 1] - self.lat_min) / (self.lat_max - self.lat_min)  # normalize latitude

        fp[:, :, 2], traj[:, :, 2], gc[:, :, 0] = (fp[:, :, 2] - self.lon_min) / (self.lon_max - self.lon_min), \
                                                  (traj[:, :, 2] - self.lon_min) / (self.lon_max - self.lon_min), \
                                                  (gc[:, :, 0] - self.lon_min) / (self.lon_max - self.lon_min)  # normalize longitude

        # only consider horizontal coordinates
        fp = fp[:, :, 1:3]
        traj = traj[:, :, 1:3]
        gc[:, :, [0, 1]] = gc[:, :, [1, 0]]  # first column latitude

        # substitude gc into random vector z
        gc = z

        # train test split
        self.fp_train, self.fp_test, self.traj_train, self.traj_test, self.gc_train, self.gc_test, self.cube_train, self.cube_test \
            = train_test_split(fp, traj, gc, cube, test_size=self.test_ratio, shuffle=False, random_state=None)

        print('finish loading data')

    def inverse_normalization(self, tensor):

        delta_lat = self.lat_max - self.lat_min
        delta_lon = self.lon_max - self.lon_min

        tensor[:, :, 0] = tensor[:, :, 0] * delta_lat + self.lat_min
        tensor[:, :, 1] = tensor[:, :, 1] * delta_lon + self.lon_min

        return tensor

    def build_graph(self):
        def generator(cube, fp, gc, reuse=False):
            with tf.variable_scope("cGAN/Generator", reuse=tf.AUTO_REUSE):
                # extract weather features
                conv1 = tf.layers.conv3d(cube, filters=3, kernel_size=[1, 5, 5], strides=[1, 3, 3], padding='valid',
                                         activation=None, name='conv1')
                drop1 = tf.nn.dropout(conv1, 0.2)
                conv2 = tf.layers.conv3d(drop1, filters=1, kernel_size=[1, 3, 3], strides=[1, 3, 3], padding='same',
                                         activation=None, name='conv2')
                drop2 = tf.nn.dropout(conv2, 0.2)
                flat1 = tf.reshape(drop2, [-1, self.input_dimension-1, 4*4*1], name='flat1')
                dense1 = tf.layers.dense(flat1, 4, activation=None, name='dense1')
                drop3 = tf.nn.dropout(dense1, 0.5)
                features = tf.layers.dense(drop3, 1, activation=None, name='dense2')

                # concatenate gc(noise) [Nx50x2], fp(condition) [Nx50x2] and weather features extracted [Nx49x1]
                # The final input for lstm cell should be dimension [Nx49x5]
                lstm_in = tf.concat([gc[:, 1:, :], fp[:, 1:, :], features], -1)

                # Seq2Seq learning with lstm
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim)

                # define initial states of lstm
                init_state = tf.contrib.rnn.LSTMStateTuple(fp[:, 0, :], fp[:, 0, :])
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, initial_state=init_state)
                # zero initialization doesn't work
                # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                #lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, dtype=tf.float32)

                # concatenate the 1st point in gc and outputs [Nx50x2]
                outputs = tf.nn.sigmoid(lstm_outputs)
                #outputs = tf.concat([tf.expand_dims(fp[:, 0, :], 1), lstm_outputs], 1)
            return outputs

        def discriminator(x, fp, cube, reuse=False):
            with tf.variable_scope("cGAN/Discriminator", reuse=tf.AUTO_REUSE):
                # extract weather features
                conv3 = tf.layers.conv3d(cube, filters=1, kernel_size=[1, 7, 7], strides=[1, 3, 3], padding='valid',
                                         activation=None, name='conv3', reuse=tf.AUTO_REUSE)
                flat2 = tf.reshape(conv3, [-1, self.input_dimension - 1, 9 * 9 * 1], name='flat2')
                dense3 = tf.layers.dense(flat2, 16, activation=None, name='dense3')
                features = tf.layers.dense(dense3, 1, activation=None, name='dense4')

                # concatenate gc(noise) [Nx50x2], fp(condition) [Nx50x2] and weather features extracted [Nx49x1]
                # The final input for lstm cell should be dimension [Nx49x5]
                #lstm_in = tf.concat([x[:, 1:, :], fp[:, 1:, :], features], -1)
                lstm_in = tf.concat([x, fp[:, 1:, :], features], -1)

                # Seq2Seq learning with lstm
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim)
                init_state = tf.contrib.rnn.LSTMStateTuple(x[:, 0, :], x[:, 0, :])
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, initial_state=init_state)
                #lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_in, dtype=tf.float32)  # zero initialization

                # concatenate the 1st point in gc and outputs [Nx50x2]
                outputs = lstm_outputs
                #outputs = tf.concat([tf.expand_dims(x[:, 0, :], 1), lstm_outputs], 1)

                # logits
                logits = tf.layers.dense(lstm_outputs, 1, activation=tf.nn.sigmoid, name='dense5')

            return outputs, logits

        self.G_generate = generator(self.cube, self.fp, self.gc)  # generate data from output using noise gc(physics based noise)
        self.D_generate, G_logits = discriminator(self.G_generate, self.fp, self.cube, reuse=True)  # fake from generator then feed into discriminator

        # binary cross entropy loss
        self.G_loss  = tf.reduce_sum(tf.log(1 - G_logits), axis=[0, 1]) / self.batch_size
        # self.fake_loss = tf.reduce_mean(tf.sqrt(tf.square(self.D_generate[:, :, 0] - self.traj[:, 1:, 0]) +
        #                                 tf.square(self.D_generate[:, :, 1] - self.traj[:, 1:, 1])), axis=None)

        # real data feed into discriminator
        D_real, D_logits = discriminator(self.traj[:, 1:, :], self.fp, self.cube)

        # binary cross entropy loss
        self.D_loss = tf.reduce_sum(- tf.log(D_logits) - tf.log(1 - G_logits), axis=[0, 1]) / self.batch_size
        # self.real_loss = tf.reduce_mean(tf.sqrt(tf.square(D_real[:, :, 0] - self.traj[:, 1:, 0]) +
        #                                 tf.square(D_real[:, :, 1] - self.traj[:, 1:, 1])), axis=None)

    def train_model(self):
        # define variables
        self.fp = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.gc = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.cube = tf.placeholder(tf.float32, [None, self.input_dimension-1, 32, 32, 1])
        self.traj = tf.placeholder(tf.float32, [None, self.input_dimension, 2])

        # load data
        self.load_data()

        # build cGAN computational graph
        print('build graph..........')
        self.build_graph()

        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cGAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cGAN/Discriminator")

        # optimizer
        # G_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.fake_loss, var_list=gen_vars)  # G Train step
        # D_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.real_loss, var_list=disc_vars)  # D Train step
        # G_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.fake_loss, var_list=gen_vars)
        # D_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.real_loss, var_list=disc_vars)
        G_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.G_loss, var_list=gen_vars)
        D_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.D_loss, var_list=disc_vars)

        # batch number
        batch_num = int(self.fp_train.shape[0] / self.batch_size)

        # training session
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        f = open('{}/loss_logs.csv'.format(self.save_dir), 'w')
        f.write('Epoch,Batch,Discriminator Loss,Generator Loss\n')

        print('start training...........')

        for j in range(self.epoch):
            for i in range(batch_num):
                train_fp_batch = self.fp_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_gc_batch = self.gc_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_traj_batch = self.traj_train[self.batch_size * i:self.batch_size * (i + 1), :, :]
                train_cube_batch = self.cube_train[self.batch_size * i:self.batch_size * (i + 1), :, :, :, :]

                # minimize G loss
                for _ in range(self.discriminator_steps):
                    _, gloss = sess.run([G_step, self.G_loss], feed_dict={self.fp: train_fp_batch,
                                                                          self.gc: train_gc_batch,
                                                                          self.traj: train_traj_batch,
                                                                          self.cube: train_cube_batch})
                # minimize D loss
                for _ in range(self.generator_steps):
                    _, dloss = sess.run([D_step, self.D_loss], feed_dict={self.fp: train_fp_batch,
                                                                          self.gc: train_gc_batch,
                                                                          self.traj: train_traj_batch,
                                                                          self.cube: train_cube_batch})

                print("Epoch: %d\t Batch: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (j+1, i+1, dloss, gloss))

                # record loss values
                f.write("%d, %d, %f, %f\n" % (j+1, i+1, dloss, gloss))
        f.close()

        save_path = tf.train.Saver().save(sess, '{}/model.ckpt'.format(self.save_dir))
        print("Model saved in path: {}".format(save_path))

        sess.close()
        print("Finish training.")

    def test_model(self, draw_figures=False):
        # define variables
        self.fp = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.gc = tf.placeholder(tf.float32, [None, self.input_dimension, 2])
        self.cube = tf.placeholder(tf.float32, [None, self.input_dimension - 1, 32, 32, 1])
        self.traj = tf.placeholder(tf.float32, [None, self.input_dimension, 2])

        # testing number loop
        for test_num in range(self.test_number):

            # load data
            self.load_data()

            # build cGAN computational graph
            print('build graph..........')
            self.build_graph()

            # batch number
            self.batch_num = int(self.fp_test.shape[0] / self.batch_size)

            saver = tf.train.Saver()
            sess = tf.Session()

            # Restore latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint('./{}/'.format(self.save_dir)))

            print("#########################################################")
            print('start testing #{}...........'.format(test_num+1))

            for i in range(self.batch_num):
                test_fp_batch = self.fp_test[self.batch_size * i:self.batch_size * (i + 1), :, :]
                test_gc_batch = self.gc_test[self.batch_size * i:self.batch_size * (i + 1), :, :]
                test_traj_batch = self.traj_test[self.batch_size * i:self.batch_size * (i + 1), :, :]
                test_cube_batch = self.cube_test[self.batch_size * i:self.batch_size * (i + 1), :, :, :, :]

                pred_traj_batch, test_loss = sess.run([self.D_generate, self.G_loss], feed_dict={self.fp: test_fp_batch,
                                                                                                 self.gc: test_gc_batch,
                                                                                                 self.traj: test_traj_batch,
                                                                                                 self.cube: test_cube_batch})
                if i == 0:
                    pred_traj = pred_traj_batch
                else:
                    pred_traj = tf.concat([pred_traj, pred_traj_batch], axis=0, name='stack_test')

                print('Batch: {} Testing Loss: {}'.format(i, test_loss))

            # do inverse normalization to the prediction
            test_pred_traj = self.inverse_normalization(pred_traj.eval(session=sess))
            # test_pred_traj, traj_test, fp_test = self.inverse_normalization(pred_traj.eval(session=sess)), \
            #                                                self.inverse_normalization(self.traj_test), \
            #                                                self.inverse_normalization(self.fp_test)
            sess.close()

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

        # calculate variance
        self.dev_ori = self.traj_test[:self.batch_size*self.batch_num, 1:, :] - self.fp_test[:self.batch_size*self.batch_num, 1:, :]
        self.dev_new = self.traj_test[:self.batch_size*self.batch_num, 1:, :] - self.pred_traj_mean

        self.l2_ori = np.sum(np.sum(self.dev_ori ** 2, axis=2), axis=1)
        self.l2_new = np.sum(np.sum(self.dev_new ** 2, axis=2), axis=1)

        percent_reduced = len(np.where(self.l2_new / self.l2_ori < 1)[0]) / float(len(self.l2_ori))
        print("l2-norm: {} percent of tracks is reduced.".format(percent_reduced))

        dev_reduced = np.var(self.l2_new[np.where(self.l2_new / self.l2_ori < 1)]) / np.var(self.l2_ori[np.where(self.l2_new / self.l2_ori < 1)])
        print("Variance is reduced by {}".format(1 - dev_reduced))

        if draw_figures is True:
            print('drawing......................')
            self.draw_figures()
            print('drawing finished')
        print("#########################################################")

    def draw_figures(self):

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

        plt.subplot(1, 2, 1)
        sns.distplot(self.dev_ori[:, :, 0].ravel(), color='green', kde=False, label='original', norm_hist=True)
        sns.distplot(self.dev_new[:, :, 0].ravel(), color='blue', kde=False, label='predicted', norm_hist=True)
        plt.title('longitude')
        plt.legend()
        plt.xlim([-1, 1])
        plt.xlabel("Deviation in degree")
        plt.ylabel("Density")

        plt.subplot(1, 2, 2)
        sns.distplot(self.dev_new[:, :, 1].ravel(), color='green', kde=False, label='predicted', norm_hist=True)
        sns.distplot(self.dev_ori[:, :, 1].ravel(), color='blue', kde=False, label='original', norm_hist=True)
        plt.title('latitude')
        plt.legend()
        plt.xlim([-1, 1])
        plt.xlabel("Deviation in degree")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.savefig("{}/separate.png".format(self.save_dir))
        plt.close()

        # draw plot for each data point
        for idx in range(self.figure_number):
            fig, ax = plt.subplots()
            plt.title('{} {} Testing result: {}'.format(self.date, self.sector, idx))
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.plot(self.pred_traj_mean[idx, :, 1], self.pred_traj_mean[idx, :, 0], '*r', markersize=2, label='predicted_track')

            # draw confidence ellipse
            ells = [Ellipse(xy=[self.pred_traj_mean[idx, i, 1], self.pred_traj_mean[idx, i, 0]],
                            width=100 * self.pred_traj_std_lon[idx, i],
                            height=100 * self.pred_traj_std_lat[idx, i],
                            angle=90-np.rad2deg(
                                np.arctan((self.pred_traj_mean[idx, i, 0] - self.pred_traj_mean[idx, i-1, 0]) /
                                          (self.pred_traj_mean[idx, i, 1] - self.pred_traj_mean[idx, i-1, 1]))))
                    for i in range(self.input_dimension-1)]

            for e in ells:
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.5)
                e.set_facecolor('r')

            plt.plot(self.traj_test[idx, :, 1], self.traj_test[idx, :, 0], '*g', markersize=2, label='ground_truth')
            plt.plot(self.fp_test[idx, :, 1], self.fp_test[idx, :, 0], '*b', markersize=2, label='flight_plan')
            plt.legend()
            plt.savefig("{}/{}.png".format(self.save_dir, idx))
            plt.close(fig)

    def plot_training_loss(self):
        f = np.loadtxt(open('{}/loss_logs.csv'.format(self.save_dir), "rb"), delimiter=",", skiprows=1)
        x = range(len(f))
        plt.figure()
        plt.title('Training Loss')
        plt.plot(x, f[:, 2], 'blue', label='Discriminator Loss')
        plt.plot(x, f[:, 3], 'black', label='Generator Loss')
        plt.legend()
        #plt.show()
        plt.savefig("{}/training_loss.png".format(self.save_dir))


if __name__ == '__main__':
    cfg = {'date': 20190624,
           'sector': 'ZID',
           'input_dimension': 50,
           'batch_size': 64,
           'lstm_hidden_dim': 2,
           'epoch': 300,
           'test_ratio': 0.2,
           'generator_steps': 10,
           'discriminator_steps': 25,
           'figure_number': 50,
           'maximum_separation_distance': 0.1,
           'test_number': 5,}

    cfg['save_path'] = './{}_{}_epoch_{}'.format(cfg['date'], cfg['sector'], cfg['epoch'])

    fun = cGAN_Traj(cfg)
    fun.train_model()
    fun.test_model(draw_figures=True)
    fun.plot_training_loss()

