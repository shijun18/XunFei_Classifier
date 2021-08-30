import os 
import pandas as pd
import numpy as np
from pandas.tseries.offsets import QuarterBegin


def run_air():
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


# def run_family():

#     def convert(val,max_len):
#         val = eval(val)
#         assert isinstance(val,list)
#         new_val = [0]*(max_len*2)
#         for i,item in enumerate(val):
#             new_val[i] = item['stream_id']
#             new_val[i+max_len] = item['datapoints'][0]['value']
#         return new_val

#     csv_path = './dataset/family/test.csv'
#     save_path = './dataset/family/pre_test.csv'

#     df = pd.read_csv(csv_path)
#     c_len = df['c'].apply(lambda x :len(eval(x)))
#     max_len = max(c_len)
#     df['c'] = df['c'].apply(lambda x: convert(x,max_len))

#     stream = {}
#     for item in df['c']:
#         for i in range(max_len):
#             if f'stream_{str(i)}' not in stream.keys():
#                 stream[f'stream_{str(i)}'] = []
#             stream[f'stream_{str(i)}'].append(item[i])
#             if f'action_{str(i)}' not in stream.keys():
#                 stream[f'action_{str(i)}'] = []
#             stream[f'action_{str(i)}'].append(item[i+max_len])
#     for key in stream.keys():
#         df[key] = stream[key]
#     # col_name = ['_id','f_i','loginname','_d','tag'] + list(stream.keys())
#     col_name = ['_id','f_i','loginname','_d'] + list(stream.keys())
#     print(col_name)
#     df[col_name].to_csv(save_path,index=False)


def run_family():
    # MAP = {
    #         '_module_version': {
    #             'FB56+ZSW1IKJ1.7':0, 
    #             'end':1
    #         }, 
    #         'power1': {
    #             '2':2, 
    #             '1':3
    #         }, 
    #         'power2': {
    #             '2':4, 
    #             '1':5
    #         }, 
    #         'power3': {
    #             '2':6, 
    #             '1':7
    #         }
    # }

    MAP = {
            '_module_version': {
                'FB56+ZSW1HKJ1.7':0, 
                'FB56+ZSW1IKJ1.7':1, 
                'end':2, 
                'FB56-ZTP07RS1.3':3, 
                'FB56+ZSW1GKJ1.7':4
            }, 
            'power1': {
                '2':5, 
                '1':6
            }, 
            'power2': {
                '2':7, 
                '1':8
            }, 
            'power3': {
                '2':9, 
                '1':10
            },
            'work1':{
                '3':11,
                '2':12, 
                '1':13
            }
    }

    def convert(val,max_len):
        stream_list = list(MAP.keys())
        val = eval(val)
        assert isinstance(val,list)
        new_val = [0]*max_len
        for item in val:
            if item['stream_id'] in stream_list:
                try:
                    index_dict = MAP[item['stream_id']]
                    new_val[index_dict[item['datapoints'][0]['value']]] = 1
                except:
                    continue
        return new_val
    
    def get_stream(val):
        stream = []
        val = eval(val)
        assert isinstance(val,list)
        for item in val:
            stream.append([item['stream_id'],item['datapoints'][0]['value']])
        return stream

    csv_path = './dataset/family/test.csv'
    save_path = './dataset/family/pre_test_v2.csv'

    
    df = pd.read_csv(csv_path)
    c_len = df['c'].apply(lambda x :len(eval(x)))
    max_len = max(c_len)
    print(max_len)

    stream_list = df['c'].apply(lambda x :get_stream(x)).tolist()
    # print(stream_list)

    stream_map = {}
    for item in stream_list:
        for sub_item in item:
            if sub_item[0] not in stream_map.keys():
                stream_map[sub_item[0]] = []
            stream_map[sub_item[0]].append(sub_item[1])
    for item in stream_map.keys():
        stream_map[item] = list(set(stream_map[item]))

    print(stream_map)

    df['c'] = df['c'].apply(lambda x: convert(x,14))

    stream = {}
    for item in df['c']:
        for i in range(14):
            if f'stream_{str(i)}' not in stream.keys():
                stream[f'stream_{str(i)}'] = []
            stream[f'stream_{str(i)}'].append(item[i])
    for key in stream.keys():
        df[key] = stream[key]
    # col_name = ['_id','f_i','loginname','_d','tag'] + list(stream.keys())
    col_name = ['_id','f_i','loginname','_d'] + list(stream.keys())
    print(col_name)
    df[col_name].to_csv(save_path,index=False)


if __name__ == '__main__':
    run_family()