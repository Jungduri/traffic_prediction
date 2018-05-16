"""
Traffic flow prediction using LSTM networks.
From Seoul metropolitan area to Gangwon-do in South Korea. 
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    """
    define a class containing pre-processed data.
    the input should be path of the pre-processed data.
    """

    def __init__(self, file_path):

        # load file path
        self.file_path = file_path

        # initializing class variable
        # traffic variable
        self.df = None
        self.min_flow = None
        self.max_flow = None

        self._start()

    def _start(self):

        self.load_data()

    def code2onehot(self, code):

        if str(code) == '0':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        elif str(code) == '1':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

        elif str(code) == '10':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        elif str(code) == '11':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        elif str(code) == '12':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

        elif str(code) == '100':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

        elif str(code) == '101':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        elif str(code) == '110':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == '111':
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == '01S':
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A':
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A+1':
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A+2':
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A0':
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A-1':
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'A-2':
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif str(code) == 'S10':
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        else:
            return 'fail'

    def load_data(self):

        df = pd.read_csv(self.file_path)

        # change column order (weather variables + traffic)
        cols = df.columns.tolist()
        cols = [cols[0]] + cols[2:-1] + [cols[1]] + [cols[-1]]
        df = df[cols]

        # normalize data
        df_nor = df.iloc[:, 1:-1].as_matrix()

        min_column = np.min(df_nor, axis=0, keepdims=True)
        max_column = np.max(df_nor, axis=0, keepdims=True)

        numerator = df_nor - min_column
        denominator = max_column - min_column

        df_nor = np.multiply(numerator, 1 / (denominator + 1e-7))  # 0 to 1
        df_nor = (df_nor - 0.5) * 2  # - 1 to 1

        # code transfer
        codes = df[["PCODE"]].as_matrix()
        code_onehot = [self.code2onehot(code[0]) for code in codes]

        # concatenate data and pcode
        data = np.concatenate((code_onehot, df_nor), axis=1)

        # set class variables
        self.df = data
        self.min_flow = min_column[0, -1]
        self.max_flow = max_column[0, -1]


class Networks:

    def __init__(self, file_path):

        # networks train hyper parameters
        self.seq_length = 4  # num cells
        self.data_dim = 22  # input dimension: code 17 + weather 4 + traffic: 1
        self.hidden_dim = 30
        self.output_dim = 1
        self.learning_rate = 0.01
        self.iterations = 400

        # get data
        self.data = Data(file_path=file_path)

        # networks placeholders
        self.x = None
        self.y = None

        # networks data variable
        self.size_train = None
        self.x_train = None
        self.y_train = None
        self.size_validation = None
        self.x_validation = None
        self.y_validation = None
        self.size_test = None
        self.x_test = None
        self.y_test = None

        # networks output values/layer
        self.predictions = None
        self.loss = None
        self.optimizer = None

        # networks train history/result values
        self.acc_loss_train = []
        self.acc_loss_validation = []
        self.rmse_test = None
        self.prediction_test = None

        # options
        self.saveFig = 1

        self._start()

    def _start(self):
        self.conf_networks()
        self.build_net()
        self.run_net()
        self.plot_net()

    def conf_networks(self, ratio_train=0.7, ratio_validation=0.15, ratio_test=0.15):

        # split data into x,y
        x = self.data.df
        y = x[:, [-1]]  # Close as label

        x_data = []
        y_data = []

        for i in range(0, len(y) - self.seq_length):
            _x = x[i:i + self.seq_length]
            _y = y[i + self.seq_length]  # Next close price
            # print(_x, "->", _y)
            x_data.append(_x)
            y_data.append(_y)

        # get data set sizes
        size_train = int(len(y_data) * ratio_train)
        size_validation = int(len(y_data) * ratio_validation)
        size_test = int(len(y_data) * ratio_test)

        # split data sets
        x_train, y_train = np.array(x_data[0:size_train]), \
                           np.array(y_data[0:size_train])
        x_validation, y_validation = np.array(x_data[size_train:size_train+size_validation]), \
                                     np.array(y_data[size_train:size_train+size_validation])
        x_test, y_test = np.array(x_data[size_train+size_validation:len(x_data)]), \
                         np.array(y_data[size_train+size_validation:len(x_data)])

        # set class variable
        self.size_train = size_train
        self.x_train = x_train
        self.y_train = y_train
        self.size_validation = size_validation
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.size_test = size_test
        self.x_test = x_test
        self.y_test = y_test

    def build_net(self):

        # input, output placeholder
        self.x = x = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
        self.y = y = tf.placeholder(tf.float32, [None, 1])

        # build a LSTM network
        cell = tf.contrib.rnn.LSTMCell(
            num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        y_prediction = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

        # loss
        variables = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables
                            if 'bias' not in v.name]) * 0.1

        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_prediction, y)))) + l2_loss  # RMSE

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.predictions = y_prediction
        self.loss = loss
        self.optimizer = optimizer

    def run_net(self):

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # iteration
            for i in range(self.iterations):

                # train step
                _, step_loss_train = sess.run([self.optimizer, self.loss], feed_dict={
                    self.x: self.x_train, self.y: self.y_train})

                # validation step
                step_loss_validation, predict_validation = \
                    sess.run([self.loss, self.predictions],
                             feed_dict={self.x: self.x_validation, self.y: self.y_validation})
                rmse_validation = np.sqrt(np.mean(np.square(predict_validation - self.y_validation)))

                # append all losses
                self.acc_loss_train.append(step_loss_train)
                self.acc_loss_validation.append(step_loss_validation)

                # print
                if np.mod(i,20) == 0:
                    print("step: {}\n".format(i) +
                          "train loss: {}\n".format(step_loss_train) +
                          "validation loss: {}\n".format(step_loss_validation) +
                          "validation RMSE: {}\n".format(rmse_validation))

            # test
            predict_test = sess.run(self.predictions, feed_dict={self.x: self.x_test})
            self.prediction_test = predict_test
            self.rmse_test = np.sqrt(np.mean(np.square(predict_test - self.y_test)))

    def plot_net(self):

        true_traffic = (self.y_test/2 + 0.5)*(self.data.max_flow-self.data.min_flow)+self.data.min_flow
        prediction_traffic = (self.prediction_test/2 + 0.5)*(self.data.max_flow-self.data.min_flow)+self.data.min_flow

        plt.figure(0, figsize=(35, 10))
        plt1 = plt.subplot(211)
        true_plt = plt1.plot(true_traffic, 'r-.', linewidth=1.2, label='True')
        prediction_plt = plt1.plot(prediction_traffic, 'k', linewidth=1, label='Predict')
        plt1.set_xlabel("Time Period")
        plt1.set_ylabel("Traffic flow")
        plt1.legend()

        plt2 = plt.subplot(212)
        train_loss_plt = plt2.plot(self.acc_loss_train, 'r-.', linewidth=1.2, label='train loss')
        validation_loss_plt = plt2.plot(self.acc_loss_validation, 'k', linewidth=1, label='validation loss')
        plt2.text(self.iterations - self.iterations / 3, max(self.acc_loss_train) - max(self.acc_loss_train) / 2,
                  'test RMSE: {0:.4f}\n'.format(self.rmse_test) +
                  "iterations: {}\n".format(self.iterations) +
                  "sequence length: {}".format(self.seq_length), fontsize=20)
        plt2.set_xlabel("Iteration")
        plt2.set_ylabel("error")
        plt.tight_layout()
        plt.savefig('../result/권역_일단위예측/seoul_to_gangwon/result{}.png'.format(self.saveFig))
        plt.show()


if __name__ == '__main__':
    cls = Networks("../data/intertoll/total_seoul_to_gangwon_refine.csv")