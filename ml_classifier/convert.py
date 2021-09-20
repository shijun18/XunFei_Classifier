import os 
import pandas as pd
import numpy as np
import cv2


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



def run_plastic():
    def compute_coord(img_path,scale=False):
        print(img_path)
        img = cv2.imread(img_path,0)
        _,binary = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        mask = np.zeros_like(img,dtype=np.uint8)
        contours,_ = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = [cv2.contourArea(contours[i]) for i in range(len(contours))]
        max_idx = np.argmax(area)
        cv2.drawContours(mask,contours,max_idx,1,cv2.FILLED)

        l_w = 650
        r_w = 1300
        l_h1 = min(np.nonzero(mask[:,l_w])[0])
        l_h2 = max(np.nonzero(mask[:,l_w])[0])
        r_h1 = min(np.nonzero(mask[:,r_w])[0])
        r_h2 = max(np.nonzero(mask[:,r_w])[0])
        coord_list = [l_w,l_h1,l_h2,r_w,r_h1,r_h2]
        if scale:
            coord_list = [l_w/1624,l_h1/1240,l_h2/1240,r_w/1624,r_h1/1240,r_h2/1240]
        print(coord_list)
        return coord_list

    #train data
    train_path = '../converter/csv_file/plastic_drum.csv'
    df = pd.read_csv(train_path)
    index_array = np.asarray(df[[f'index_{case}' for case in range(1,7)]])
    select_index = np.nonzero(np.array(np.sum(index_array,axis=1)!=-6))[0]
    print(len(select_index))
    df = df.iloc[list(select_index)]

    coord = []
    for item in df.iterrows():
        coord.append(compute_coord(item[1][-1],scale=True))
    
    coord_array = np.asarray(coord)
    print(coord_array.shape)
    df['l_w'] = list(coord_array[:,0])
    df['l_h1'] = list(coord_array[:,1])
    df['l_h2'] = list(coord_array[:,2])
    df['r_w'] = list(coord_array[:,3])
    df['r_h1'] = list(coord_array[:,4])
    df['r_h2'] = list(coord_array[:,5])
    df['l_sub'] = list(coord_array[:,2] - coord_array[:,1])
    df['r_sub'] = list(coord_array[:,5] - coord_array[:,4])

    for i in range(1,7):
        if i < 5:
            df[f'scale_index_{i}'] = df[f'index_{i}'].apply(lambda x:x/1240)
        else:
            df[f'scale_index_{i}'] = df[f'index_{i}'].apply(lambda x:x/1624)
    del df['id']
    del df['label']
    print(df)
    save_path = './dataset/plastic_drum/train_xy_new.csv'
    df.to_csv(save_path,index=False)

    # test data
    test_csv = '../analysis/result/Plastic_Drum/v3.0-pretrained/submission_ave.csv'
    df = pd.read_csv(test_csv)
    index_array = np.asarray(df['category_id'])
    select_index = np.nonzero(index_array!=0)[0]
    print(len(select_index))
    df = df.iloc[list(select_index)]

    coord = []
    for item in df.iterrows():
        item_path = os.path.join('../dataset/Plastic_Drum/test',str(item[1][0]).zfill(2) + f'/{item[1][1]}')
        coord.append(compute_coord(item_path,scale=True))
    
    coord_array = np.asarray(coord)
    print(coord_array.shape)
    df['l_w'] = list(coord_array[:,0])
    df['l_h1'] = list(coord_array[:,1])
    df['l_h2'] = list(coord_array[:,2])
    df['r_w'] = list(coord_array[:,3])
    df['r_h1'] = list(coord_array[:,4])
    df['r_h2'] = list(coord_array[:,5])
    df['l_sub'] = list(coord_array[:,2] - coord_array[:,1])
    df['r_sub'] = list(coord_array[:,5] - coord_array[:,4])

    del df['category_id']
    del df['prob_1']
    del df['prob_2']
    print(df)
    save_path = './dataset/plastic_drum/test_xy_new.csv'
    df.to_csv(save_path,index=False)

if __name__ == '__main__':
    # run_family()
    run_plastic()