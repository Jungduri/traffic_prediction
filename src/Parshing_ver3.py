#### learning ver4를 위한 작업
## IC_CODE와 local group matching
## 기상 + 이동 데이터 합침

import numpy as np
import pandas as pd
import operator


######### 나중에 보안
def load_list():

    ## TOLL 정보 !!
    toll_info = pd.read_table('data/intertoll/tollname/tollname.txt',sep='\t')
    toll_info = toll_info[['ic','권역']].as_matrix()
    toll_info = dict((x[0],x[1]) for x in toll_info[0:])

    ## 날씨정보 seoul
    weather_info_seoul = pd.read_csv('data/intertoll/weather_seoul.csv')
    weather_info_seoul = weather_info_seoul.as_matrix()
    weather_info_temp_seoul = dict((x[0],x[1]) for x in weather_info_seoul[0:])
    weather_info_rain_seoul = dict((x[0],x[2]) for x in weather_info_seoul[0:])
    weather_info_new_snow_seoul = dict((x[0],x[3]) for x in weather_info_seoul[0:])
    weather_info_total_snow_seoul = dict((x[0],x[4]) for x in weather_info_seoul[0:])

    ## 날씨정보 gangwon
    weather_info_gangwon = pd.read_csv('data/intertoll/weather_daegwanryung.csv')
    weather_info_gangwon = weather_info_gangwon.as_matrix()
    weather_info_temp_gangwon = dict((x[0],x[1]) for x in weather_info_gangwon[0:])
    weather_info_rain_gangwon = dict((x[0],x[2]) for x in weather_info_gangwon[0:])
    weather_info_new_snow_gangwon = dict((x[0],x[3]) for x in weather_info_gangwon[0:])
    weather_info_total_snow_gangwon = dict((x[0],x[4]) for x in weather_info_gangwon[0:])

    skipdict = [['연계공통'], ['연계할인'], ['최장*3'], ['최장'], ['최장*10'],['테스트개발']]

    table = {-2: 201301, -1: 201302, 0:201303 ,1:201304 ,2:201305 ,3:201306 ,4:201307 ,5:201308 ,6:201309 ,7:201310 ,8:201311 ,9:201312,
                 10:201401 ,11:201402 ,12:201403 ,13:201404 ,14:201405 ,15:201406 ,16:201407 ,17:201408 ,18:201409 ,19:201410 , 20:201411 ,21:201412,
                 22:201501, 23:201502, 24:201503, 25:201504 , 26:201505 , 27:201506 , 28:201507 , 29:201508 , 30:201509 , 31:201510 , 32:201511 ,33:201512 ,
                 34:201601 , 35:201602 , 36:201603, 37:201604, 38:201605, 39:201606, 40:201607, 41:201608, 42:201609, 43:201610, 44:201611, 45:201612,
                 46:201701, 47:201702, 48:201703, 49:201704}

    total_seoul_to_gangwon = pd.DataFrame()
    total_gangwon_to_seoul = pd.DataFrame()

    for j in range(-2,len(table)-2):
        df = pd.read_table('data/intertoll/'+'{}'.format(table[j])+'.txt', sep='|', header=None,
                           names=['DATE', 'ORI_TOLL_CODE', 'DES_TOLL_CODE', 'ORI_TOLL_NAME', 'DES_TOLL_NAME',
                                  'DES_CAR1', 'DES_CAR2','DES_CAR3','DES_CAR4','DES_CAR5','DES_CAR6','DES_CAR_TOTAL',
                                  'ORI_CAR1', 'ORI_CAR2','ORI_CAR3','ORI_CAR4','ORI_CAR5','ORI_CAR6','ORI_CAR_TOTAL',
                                  'NAN']).iloc[:, 0:-1]

        df = df[['DATE', 'ORI_TOLL_CODE', 'DES_TOLL_CODE', 'ORI_TOLL_NAME', 'DES_TOLL_NAME',
                 'DES_CAR_TOTAL','ORI_CAR_TOTAL']]

        ## Toll gate 정보 추가 ( 과거 없는거 반영 )
        test = df[[ 'ORI_TOLL_NAME','ORI_TOLL_CODE']].as_matrix()
        dic =  dict((x[1],x[0] )for x in test[0:])
        list_values = [ v for v in dic.values() ]

        ## 앞으로 넘길 것들
        ## ['연계공통', '연계할인', '최장*3', '최장', '최장*10']
        ## 권역 이름 NA로 표시

        for i in range(0,list_values.__len__()):
            if list_values[i] not in toll_info and [list_values[i]] not in skipdict:
                ## toll정보에 없고, skiplist에도 없으면 관찰함
                ## skip
                print(list_values[i], i)

                detailOpt = int(input("디테일을 보고 싶으면 1 아니면 0 "))
                if detailOpt == 1:
                    df.loc[df['ORI_TOLL_NAME'] == list_values[i]][
                        ['ORI_TOLL_NAME', 'DES_TOLL_NAME', "DES_CAR_TOTAL", 'ORI_CAR_TOTAL']]
                    detailOpt = 0

                addTollList = int(input("Toll list에 추가하려면 1 아니면 0 "))
                if addTollList == 1:
                    localName = input("권역 이름: ")
                    toll_info.update({list_values[i]:localName})
                    addTollList = 0

        ## toll_info로부터 ORI_LOCAL_NAME 만듦
        df['ORI_LOCAL_NAME'] = df['ORI_TOLL_NAME'].map(toll_info)
        df['DES_LOCAL_NAME'] = df['DES_TOLL_NAME'].map(toll_info)

        ## 지역간 이동 data frame 정의
        seoul_to_gangwon = df.loc[(df['ORI_LOCAL_NAME'] == '서울경기') & (df['DES_LOCAL_NAME'] == '강원')]
        gangwon_to_seoul = df.loc[(df['ORI_LOCAL_NAME'] == '강원') & (df['DES_LOCAL_NAME'] == '서울경기')]

        ## 원하는 열만 가져오기
        seoul_to_gangwon = seoul_to_gangwon[["DATE","ORI_LOCAL_NAME","DES_LOCAL_NAME","DES_CAR_TOTAL"]]
        gangwon_to_seoul = gangwon_to_seoul[["DATE","ORI_LOCAL_NAME","DES_LOCAL_NAME","DES_CAR_TOTAL"]]

        ## groupby로 summation하고 index 재정비
        seoul_to_gangwon = seoul_to_gangwon.groupby(seoul_to_gangwon.DATE).sum().reset_index()
        gangwon_to_seoul = gangwon_to_seoul.groupby(gangwon_to_seoul.DATE).sum().reset_index()

        ## 기상정보에서 얻어옮.
        seoul_to_gangwon["MEAN_TEMP"] = seoul_to_gangwon["DATE"].map(weather_info_temp_gangwon)
        gangwon_to_seoul["MEAN_TEMP"] = gangwon_to_seoul["DATE"].map(weather_info_temp_seoul)

        seoul_to_gangwon["AVG_RAIN"] = seoul_to_gangwon["DATE"].map(weather_info_rain_gangwon)
        gangwon_to_seoul["AVG_RAIN"] = gangwon_to_seoul["DATE"].map(weather_info_rain_seoul)

        seoul_to_gangwon["AVG_SNOW"] = seoul_to_gangwon["DATE"].map(weather_info_new_snow_gangwon)
        gangwon_to_seoul["AVG_SNOW"] = gangwon_to_seoul["DATE"].map(weather_info_new_snow_seoul)

        seoul_to_gangwon["TOTAL_SNOW"] = seoul_to_gangwon["DATE"].map(weather_info_total_snow_gangwon)
        gangwon_to_seoul["TOTAL_SNOW"] = gangwon_to_seoul["DATE"].map(weather_info_total_snow_seoul)

        total_seoul_to_gangwon = total_seoul_to_gangwon.append(seoul_to_gangwon, ignore_index=True)
        total_gangwon_to_seoul = total_gangwon_to_seoul.append(gangwon_to_seoul, ignore_index=True)

        print(table[j])

    total_seoul_to_gangwon.to_csv('data/intertoll/total_seoul_to_gangwon.csv',encoding='utf-8')
    total_gangwon_to_seoul.to_csv('data/intertoll/total_gangwon_to_seoul.csv',encoding='utf-8')






def run():
    load_list()

run()