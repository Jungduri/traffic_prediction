## P-CODE 분석
## 1. P-CODE별 교통량 분석
## 2. PLOT 위주
## 3. 목적: P-CODE가 이전 P-CODE와 인과성이 있는가?

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def makeDic(loc):

    dic = {}
    with open(loc) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dic[row['SDATE']] = row['P_CODE']

    return dic



def getData():

    df = pd.read_table("D:/Dropbox/01.Documents/03.KAIST-Ph.D/01.Research/01.Traffic_with_learning/02.2018S_ITS_P-Code/traffic_prediction/data/pcode/traffic_master_all.txt",sep=',')

    ic_Code = 101 # 서울 Tollgate

    df = df.loc[lambda df: df.IC_CODE==ic_Code, :][["SDATE","TOTAL_CNT","P_CODE","AVG_TEMP","RAIN_AMOUNT","SNOW_AMOUNT","RAIN_YN","SNOW_YN"]]

    dics = makeDic('../PCODE.csv')

    # pcode regen
    for i in range(0,df.__len__()):
        newCode = dics[df.iloc[i,0]]
        df.iloc[i,2] = newCode

    return df




df = getData()

pcode_key = {0:'0',1:'1',2:'10',3:'11',4:'12',5:'100',6:'101',7:'110',8:'111',9:'01S',10:'A',11:'A+1',12:'A+2',13:'A0',14:'A-1',15:'A-2',16:'S10'}

df = df[["TOTAL_CNT","SDATE","P_CODE"]]
pcode = []
for i in range(0,len(pcode_key)):
    pcode.append(df.loc[lambda df: df.P_CODE == pcode_key[i], :].as_matrix())


## bar plot 부분 (Pcode가 특이하면 가시화하면 의미가 있음)
# plt.bar(range(len(pcode[14][:,0])),pcode[14][:,0])
for i in range(len(pcode_key)):
    plt.bar(range(len(pcode[i][:, 0])), pcode[i][:, 0],color='k')
    plt.title('{} case result'.format(pcode_key[i]))
    plt.savefig('{} result.png'.format(pcode_key[i]))
    plt.show(block = False)
    plt.close()

## Box plot 부분
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(pcode_key)):
    ax.boxplot(list(pcode[i][:,0]),positions=[i],widths=0.8)
    ax.set_xlim(0-0.5, len(pcode_key)-0.5)
    ax.set_ylim(60000,140000)

plt.xticks(np.arange(0, len(pcode_key)), list(pcode_key.values()))
plt.show()

df.groupby(df['P_CODE']).describe()
