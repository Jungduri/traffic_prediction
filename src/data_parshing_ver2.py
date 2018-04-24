## n시간 단위 예측


import numpy as np
import pandas as pd

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


def getData():
    ## pcode data read
    pcode=pd.read_csv('data/PCODE.csv')
    # dataFrame -> dict
    pcode=pcode.set_index('DATE')['PCODE'].to_dict()

    # 1월부터 12월까지
    for i  in range(1,13):
        ##### df 읽는 부분
        # toll data read
        df = pd.read_table('data/2016/2016' + "{0:02d}".format(i)+'.txt', sep='|',header=None ,names=['DATE','TIME','CODE','INOUT_CODE','HIPASS_CODE','OPERATION_CODE','STATUS_CODE','CAR1','CAR2','CAR3','CAR4','CAR5','CAR6','TOTAL_CARS','NAN']).iloc[:,0:-1]
        # 관심 대상 추출
        df = df[['DATE','TIME','CODE','INOUT_CODE','TOTAL_CARS']]
        # 서울 tollgate 자료 추출
        df = df[df["CODE"]==101]
        # 들어오는 것 기준
        df = df[df["INOUT_CODE"] == 0]
        # DATE와 TIME기준으로 sort
        df = df.sort_values(by=["DATE","TIME"])
        # group by로 시간별 하나의 데이터만 취득(하이패스 유무에 고려X 모두 합한 것을 교통량으로 생각)
        df = df.groupby([df["DATE"], df["TIME"]], as_index=False).sum()[["DATE","TIME","TOTAL_CARS"]]

        ##### pcode 변환부분
        # 데이터에서 key 받아오기
        keys = df['DATE'].as_matrix()
        # 날짜별로 상응하는 코드 얻어옴
        cor_codes = [pcode.get(key) for key in keys]
        # 날짜별로 상응하는 코드를 원핫으로 변경
        code_onehot = [pcodeToOnehot(code) for code in cor_codes]

        ##### data + pcode
        # pocde와 전체 차 개수만 뽑아냄
        if i == 1: # 첫번째 시행일때 변수 선언
            data = np.concatenate((code_onehot,df[["TIME",'TOTAL_CARS']].as_matrix()),axis=1)
        else:
            data = np.concatenate((data,np.concatenate((code_onehot,df[["TIME",'TOTAL_CARS']].as_matrix()),axis=1)),axis=0)

        print("Loading and parshing is lenaing on "+"{}".format(i) + " month data\n")

    ## normalization
    # 시간
    data = data.astype('f')
    data[:,-2] = data[:,-2]/24

    # flow
    minFlow = min(data[:,-1])
    maxFlow = max(data[:,-1])

    numerator = data[:,-1] - minFlow
    denominator = maxFlow - minFlow

    data[:,-1] = numerator/(denominator + 1e-7)

    return int(minFlow), int(maxFlow), data

