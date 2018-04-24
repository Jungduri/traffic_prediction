import tensorflow as tf
from src import Data_cropping


class pCodedDNN():

    # data 위치
    loc = '../traffic_master_all.csv'

    def __init__(self):
        tf.set_random_seed(777)

    def start(self):
        [X,Y] = Data_cropping.cropping(self.loc, ['TOTAL_CNT', 'P_CODE', 'AVG_TEMP', 'RAIN_AMOUNT', 'SNOW_AMOUNT'])

        return 0


if __name__ == "__main__":
    pCodedDNN().start()