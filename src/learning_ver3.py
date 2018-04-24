## pcode기반 일 예측

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error

import os

import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

###############################################################################################################
##parshing 부분

## p-code dictionary 만들기
def makeDic(loc):

    dic = {}
    with open(loc) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dic[row['SDATE']] = row['P_CODE']

    return dic

def pcodeToOnehot(pcode):
    if str(pcode) == '0':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    elif str(pcode) == '1':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    elif str(pcode) == '10':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    elif str(pcode) == '11':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    elif str(pcode) == '12':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    elif str(pcode) == '100':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    elif str(pcode) == '101':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == '110':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == '111':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == '01S':
        return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A':
        return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A+1':
        return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A+2':
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A0':
        return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A-1':
        return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'A-2':
        return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    elif str(pcode) == 'S10':
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    else:
        return 'fail'


## read
df = pd.read_table("D:/Dropbox/01.Documents/03.KAIST-Ph.D/01.Research/01.Traffic_with_learning/02.2018S_ITS_P-Code/traffic_prediction/data/pcode/traffic_master_all.txt",sep=',')

ic_Code = 101 # 서울 Tollgate
# 필드 가져오기
df = df.loc[lambda df: df.IC_CODE==ic_Code, :][["SDATE","TOTAL_CNT","P_CODE","AVG_TEMP","RAIN_AMOUNT","SNOW_AMOUNT","RAIN_YN","SNOW_YN"]]

dics = makeDic('../PCODE.csv')

# pcode regen
for i in range(0,df.__len__()):
    newCode = dics[df.iloc[i,0]]
    df.iloc[i,2] = newCode

##### pcode 변환부분
# 데이터에서 key 받아오기
keys = df['SDATE'].as_matrix()
# 날짜별로 상응하는 코드 얻어옴
cor_codes = [dics.get(key) for key in keys]
# 날짜별로 상응하는 코드를 원핫으로 변경
code_onehot = [pcodeToOnehot(code) for code in cor_codes]

##### data + pcode
data = np.concatenate((code_onehot, df[["AVG_TEMP","RAIN_AMOUNT","SNOW_AMOUNT","RAIN_YN","SNOW_YN",'TOTAL_CNT']].as_matrix()), axis=1)

## normalization
maxN = np.max(data[:,-6:],axis=0,keepdims=True)
minN = np.min(data[:,-6:],axis=0,keepdims=True)

numerator = data[:, -6:] - minN
denominator = maxN - minN

data[:,-6:] = np.multiply(numerator,1/(denominator + 1e-7))


########################################################################################################################
saveFig = 8
## learning
# train Parameters
seq_length = 4 # =cell 개수
data_dim = 23 # = 입력의 개수: pcode 17개 + 기상 5개 + 교통량 1개
hidden_dim = 30
output_dim = 1
learning_rate = 0.01
iterations = 800

## get data
[minFlow, maxFlow, x] = [minN[0,-1], maxN[0,-1], data]
y = x[:,[ -1]]  # Close as label

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# data X -> 1569,6,23
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_mean(tf.divide(tf.abs(Y_pred - Y),Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# # cost/loss
# loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate)
# train = optimizer.minimize(loss)

# MAPE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
mape = tf.reduce_mean(tf.divide(tf.abs(targets - predictions),targets))
lossAll = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, np.sqrt(step_loss*maxFlow)))
        lossAll.append(step_loss*maxFlow)

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("MAPE: {}".format(mape_val))


# Plot predictions
plt.figure(0,figsize=(35,10))
plt1 = plt.subplot(211)
first_plt_test = plt1.plot(testY*maxFlow+minFlow,'r-.',linewidth=1.2,label='True')
first_plt_predict = plt1.plot(test_predict*maxFlow+minFlow,'k',linewidth=1,label='Predict')
plt1.set_xlabel("Time Period")
plt1.set_ylabel("Traffic flow")
plt1.legend()

plt2 = plt.subplot(212)
plt2.plot(lossAll,'r-.',linewidth=1.2)
plt2.set_xlabel("Iteration")
plt2.set_ylabel("error")
plt2.text(iterations-iterations/3, max(lossAll)-max(lossAll)/2,
          'test error(MAPE): {0:.4f}'.format(mape_val)+\
          '\ntest error(veh): {0:.1f}'.format(np.sum(abs(test_predict-testY))/len(testY) * maxFlow)+\
          "\niterations: {}".format(iterations)+\
          "\nsequence length: {}".format(seq_length),fontsize = 20)
plt.tight_layout()
plt.savefig('result{}.png'.format(saveFig))
plt.show()


