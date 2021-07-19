import os 
import pandas as pd
import numpy as np


def convert(val,index_array):
    Hi = 0
    Lo = 0
    for i in range(index_array.shape[0]):
        if index_array[i][1] >= val:
            Hi = i
            Lo = i-1
            break
        else:
            continue
    BPHi = index_array[Hi][1]
    BPLo = index_array[Lo][1]

    IAQIHi = index_array[Hi][0]
    IAQILo = index_array[Lo][0]

    IAQIp = (IAQIHi-IAQILo)*(val - BPLo) / (BPHi-BPLo) + IAQILo

    return IAQIp


csv_path = './dataset/air/pre_train/保定2016年.csv'
save_path = './dataset/air/pre_train/保定2016年_IAQI.csv'
df = pd.read_csv(csv_path)

# csv_path = './dataset/air/pre_test/石家庄20160701-20170701.csv'
# save_path = './dataset/air/pre_test/石家庄20160701-20170701_IAQI.csv'
# df = pd.read_csv(csv_path)

AQI = pd.read_csv('./AQI.csv')


col_list = ["PM2.5", "PM10", "SO2", "CO", "NO2", "O3_8h"]
for col in col_list:
    index_array = np.asarray(AQI[['IAQI',col]])
    df[f'{col}_IAQI'] = df[col].apply(lambda x : convert(x,index_array))

df.to_csv(save_path,index=False,encoding="utf-8")