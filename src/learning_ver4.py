## 서울 -> 강원 간
## RMSE
## 아웃라이어 빼면 4%대 에러


import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error

import os

import matplotlib
matplotlib.use('TKAgg')
# matplotlib.rcParams['backend.qt4']='PySide'
import matplotlib.pyplot as plt

from sklearn import preprocessing

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

tf.set_random_seed(777)  # reproducibility

## read
## SEOUL TO GANGWON ##
df = pd.read_csv("data/intertoll/total_seoul_to_gangwon_refine.csv")
## GANGWON TO SEOUL ##
# df = pd.read_csv("data/intertoll/total_gangwon_to_seoul_refine.csv")

## column 순서 변경 ( 날씨 + 교통량 )
cols = df.columns.tolist()
cols = [cols[0]] + cols[2:-1] + [cols[1]] + [cols[-1]]
df = df[cols]

## normalize 교통량 ~ 기상정보 5행
df_nor = df.iloc[:,1:-1].as_matrix()

minFlow = df_nor[:,-1].min()
maxFlow = df_nor[:,-1].max()

minN = np.min(df_nor,axis=0,keepdims=True)
maxN = np.max(df_nor,axis=0,keepdims=True)

numerator = df_nor - minN
denominator = maxN - minN

df_nor = np.multiply(numerator,1/(denominator + 1e-7)) # 0 to 1
df_nor = ( df_nor - 0.5 ) *2 ## - 1 to 1


##### (data[:,-1]/2 + 0.5 )*(maxFlow-minFlow) + minFlow -> 다시 데이터 복귀

## pcode 변환부분 17
codes = df[["PCODE"]].as_matrix()
code_onehot = [pcodeToOnehot(code[0]) for code in codes]

##### data + pcode
data = np.concatenate((code_onehot, df_nor), axis=1)

########################################################################################################################
saveFig = 0
## learning
# train Parameters
seq_length = 4 # =cell 개수
data_dim = 22 # = 입력의 개수: pcode 17개 + 기상 4개 + 교통량 1개
hidden_dim = 30
output_dim = 1
learning_rate = 0.01
iterations = 290

## get data
x = data
y = x[:,[ -1]]  # Close as label

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# data X -> 1511,4,22
# data Y -> 1511,1
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), \
                np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), \
                np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))
# loss = tf.reduce_mean(tf.divide(tf.abs(Y_pred - Y),Y+1e7))  # MAPE
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
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
# mape = tf.reduce_mean(tf.divide(tf.abs(targets - predictions),targets+1e7))
lossAll = []
loss_validation = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        validation_loss = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        # print("MAPE: {}".format(mape_val))

        lossAll.append(np.sqrt(step_loss/train_size))
        loss_validation.append(validation_loss)

        print("[step: {}] loss: {}".format(i, np.sqrt(step_loss/train_size)))
        print("test RMSE: {}".format(validation_loss))

        # if i >= 30:
        #     if np.abs(validation_loss - np.sqrt(step_loss/train_size)) <= 0.01:
        #         break


##### (data[:,-1]/2 + 0.5 )*(maxFlow-minFlow) + minFlow -> 다시 데이터 복귀

# Plot predictions
plt.figure(0,figsize=(35,10))
plt1 = plt.subplot(211)
first_plt_test = plt1.plot((testY/2 + 0.5)*(maxFlow-minFlow)+minFlow,'r-.',linewidth=1.2,label='True')
first_plt_predict = plt1.plot((test_predict/2 + 0.5)*(maxFlow-minFlow)+minFlow,'k',linewidth=1,label='Predict')
plt1.set_xlabel("Time Period")
plt1.set_ylabel("Traffic flow")
plt1.legend()
# plt1.set_xticks(np.arange(0, len(test_predict)),[df["DATE"].as_matrix()[train_size:len(dataX)]],rotation=20)

plt2 = plt.subplot(212)
plt2.plot(lossAll,'r-.',linewidth=1.2)
plt2.set_xlabel("Iteration")
plt2.set_ylabel("error")
plt2.text(iterations-iterations/3, max(lossAll)-max(lossAll)/2,
          'test error(RMSE): {0:.4f}'.format(validation_loss)+
          # '\ntest error(veh): {0:.1f}'.format(rmse_val)+
          "\niterations: {}".format(iterations)+
          "\nsequence length: {}".format(seq_length),fontsize = 20)
plt.tight_layout()
plt.savefig('result/권역_일단위예측/seoul_to_gangwon/result{}.png'.format(saveFig))
plt.show()

a = 1
y = (testY/2 + 0.5)*(maxFlow-minFlow)+minFlow
y_hat = (test_predict/2 + 0.5)*(maxFlow-minFlow)+minFlow

### 특정 구간 확대해서 PLOT
startDate = 330
endDate = 360
date = df["DATE"].as_matrix()[-test_size:][startDate:endDate]
fig, ax1 = plt.subplots()
lns1 = ax1.plot(y[startDate:endDate],'r-.',linewidth=2.8,label='True')
lns2 = ax1.plot(y_hat[startDate:endDate],'k',linewidth=1.7,label='Predict')
# ax1.legend(prop={'size': 18},loc=1)
# plt.xticks(np.arange(0, len(test_predict)),df["DATE"].as_matrix()[-test_size:],rotation=45,size=2)
# plt.tick_params(labelsize=18)
ax1.set_xlabel("Time",fontsize=16)
ax1.set_ylabel("Traffic flow",fontsize=16)
ax1.set_xticks(np.arange(0, len(date))[::7])
ax1.set_xticklabels(date[::7],rotation=45)
ax1.set_ylim([20000,70000])

rain = df["AVG_RAIN"].as_matrix()[-test_size:][startDate:endDate]
ax2 = ax1.twinx()
lns3 = ax2.plot(rain,'b',linewidth=1.7,label='AVG_rain')
ax2.set_ylim([-20, 50])
ax2.set_ylabel('rain[mm]',color='b',fontsize=16)
ax2.tick_params('y', colors='b')
# ax2.legend(prop={'size': 18},loc=0)
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

fig.tight_layout()


# plt.tight_layout()



## p code 별 box plot
pcode_key = {0:'0',1:'1',2:'10',3:'11',4:'12',5:'100',6:'101',7:'110',8:'111',9:'01S',10:'A',11:'A+1',12:'A+2',13:'A0',14:'A-1',15:'A-2',16:'S10'}
date_key = {0:'CODE1',1:'CODE2',2:'CODE3',3:'CODE4',4:'CODE5',5:'CODE6',6:'CODE7',7:'CODE8',8:'CODE9',
            9:'CODE10',10:'CODE11',11:'CODE12',12:'CODE13',13:'CODE14',14:'CODE15',15:'CODE16',16:'CODE17'}

df_pcode = df[["DES_CAR_TOTAL","PCODE"]]


pcode = []
top = 120000
for i in range(0,len(pcode_key)):
    pcode.append(df_pcode.loc[lambda df_pcode: df_pcode.PCODE == pcode_key[i], :].as_matrix())

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(pcode_key)):
    box = ax.boxplot(list(pcode[i][:,0]),positions=[i],widths=0.8)

    ## 위에 텍스트 출력
    ax.text(i, top - (top * 0.05), int(np.average(list(pcode[i][:,0]))),
             horizontalalignment='center', weight='semibold',
             color='royalblue',fontsize=8)
    ax.text(i, top - (top * 0.07), int(np.std(list(pcode[i][:,0]))),
             horizontalalignment='center', weight='semibold',
             color='darkred',fontsize=8)
    ## 색깔 입히기
    plt.figtext(0.05, 0.88, 'mean',
                backgroundcolor="royalblue", color='white', weight='roman',
                size='x-small')
    plt.figtext(0.05, 0.835, 'std. ',
                backgroundcolor="darkred", color='white', weight='roman',
                size='x-small')

ax.set_xlim(0-0.5, len(pcode_key)-0.5)
ax.set_ylim(15000,top)
fig.canvas.set_window_title('Box Plot Group By CODE')
ax.set_axisbelow(True)
ax.set_title('Box Plot Group By Date Code',size=20)
ax.set_xlabel('Date codes',size=18,weight = 'bold')
ax.set_ylabel('Total traffic',size=18,weight = 'bold')
plt.xticks(np.arange(0, len(pcode_key)), list(date_key.values()),rotation=45,size=14)
plt.tight_layout()



## error boxplot

y = (testY/2 + 0.5)*(maxFlow-minFlow)+minFlow
y_hat = (test_predict/2 + 0.5)*(maxFlow-minFlow)+minFlow

loss_RMSE = np.mean(loss_validation)
loss_MAPE = np.mean(np.divide(np.abs(y-y_hat),y))
loss_MAD = np.mean(np.abs(y-y_hat))

fig, ax = plt.subplots(figsize=(10, 6))


box = ax.boxplot([loss_RMSE, loss_MAPE, loss_MAD],widths=0.8)
top = 30
## 위에 텍스트 출력
ax.text(i, top - (top * 0.05), int(np.average(loss_validation)),
         horizontalalignment='center', weight='semibold',
         color='royalblue',fontsize=8)
ax.text(i, top - (top * 0.07), int(np.std(loss_validation)),
         horizontalalignment='center', weight='semibold',
         color='darkred',fontsize=8)
## 색깔 입히기
plt.figtext(0.05, 0.88, 'mean',
            backgroundcolor="royalblue", color='white', weight='roman',
            size='x-small')
plt.figtext(0.05, 0.835, 'std. ',
            backgroundcolor="darkred", color='white', weight='roman',
            size='x-small')
ax.set_axisbelow(True)
ax.set_ylabel('RMSE',size=18,weight = 'bold')
plt.tight_layout()

