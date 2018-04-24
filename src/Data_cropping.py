import csv
import numpy as np
import os

# pcode -> onehot
def makeDic(loc):

    dic = {}
    with open(loc) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dic[row['SDATE']] = row['P_CODE']

    return dic

def convertPCode(pcode):
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


def cropping(location, fileds,id):

    # 초기 정의
    rowData = []
    dataX = np.zeros([1, fileds.__len__() + 16 - 1]) # one hot 개수 - 교통량 개수
    dataY = []

    # 디버깅용 변수
    i=0

    # csv 파일 열기
    with open(location) as csvfile:
        reader = csv.DictReader(csvfile)

        dics = makeDic('../PCODE.csv')

        # 한 행 처리
        for row in reader:

            # 첫번째 행의 date 정보를 받아옴 -> 그에 맞는 P코드를 찾아줌
            date = row['SDATE']
            newCode = dics[date]

            # id와 matching
            if row["IC_CODE"] == str(id):
                #선택된 필드만 작업
                for filed in fileds:

                    # data 재정리
                    if filed == 'TOTAL_CNT':
                        dataY = np.append(dataY,row[filed])

                    elif filed == 'P_CODE':
                        rowData = np.append(rowData,convertPCode(newCode))

                    elif filed == 'AVG_TEMP':
                        rowData = np.append(rowData,row[filed])

                    elif filed == 'RAIN_AMOUNT':
                        rowData = np.append(rowData,row[filed])

                    elif filed == 'SNOW_AMOUNT':
                        rowData = np.append(rowData,row[filed])


                # 디버깅용
                if np.mod(i,1000) == 0:
                    print(i)

                # data stacking
                dataX = np.vstack((dataX,rowData))
                rowData = []
                i=i+1

        return [np.transpose(dataX), np.transpose(dataY)]


# makeDic('../PCODE.csv')

# 마지막 숫자는 코드
[X,Y] = cropping('../traffic_master_all.csv',['TOTAL_CNT','P_CODE','AVG_TEMP','RAIN_AMOUNT','SNOW_AMOUNT'],181)

print(1)