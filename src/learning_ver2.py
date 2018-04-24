# 2시간까지 미리 예보

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src import data_cropping_panda as dcp

tf.set_random_seed(777)  # reproducibility

# train Parameters
seq_length = 12 # =cell 개수
data_dim = 19 # = 입력의 개수: pcode 17개 + 시간 1개 + 교통량 1개
hidden_dim = 40
output_dim = 6 # 예측 길이
learning_rate = 0.1
iterations = 500

## get data
[minFlow, maxFlow, x] = dcp.getData()
y = x[:,[ -1]]  # Close as label

dataX = []
dataY = []

for i in range(0, len(y) - seq_length - output_dim - 1): ## y- output이 2개이므로 1개 하락
    _x = x[i:i + seq_length]
    _y = y[i + seq_length:i + seq_length+output_dim]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

trainY = np.reshape(trainY,(train_size,output_dim))
testY = np.reshape(testY,(test_size,output_dim))

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim]) # 2시간 예측

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, output_dim])
predictions = tf.placeholder(tf.float32, [None, output_dim])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, np.sqrt(step_loss/len(dataY))))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    errs = np.zeros(output_dim)
    peaks = np.zeros(output_dim)

    for i in range(0, output_dim):
        # Plot predictions
        plt.figure(i)
        first_plt_test = plt.plot(testY[:,i]*maxFlow+minFlow,'r-.',linewidth=1.2,label='True')
        first_plt_predict = plt.plot(test_predict[:,i]*maxFlow+minFlow,'k',linewidth=1,label='Predict')
        plt.xlabel("Time Period")
        plt.ylabel("Traffic flow")
        plt.legend()
        plt.show()

        # calculate error
        errs[i] = np.sqrt(mean_squared_error(testY[:,i]*maxFlow,test_predict[:,i]*maxFlow))
        peaks[i] = np.max(np.abs(testY[:,i]*maxFlow,test_predict[:,i]*maxFlow))

        print("{} hour estimation reslut = RMSE {}, Peak {}".format(i+1,errs[i], peaks[i]))


