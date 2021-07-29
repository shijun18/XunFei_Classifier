import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

def sample_count(input_path,save_path):

    info = []
    for item in os.scandir(input_path):
        info_item = [item.name,len(os.listdir(item.path))]
        info.append(info_item)
    info.sort(key=lambda x:eval(x[0]))
    col = ['class','sample_num']
    csv_file = pd.DataFrame(columns=col,data=info)
    csv_file.to_csv(save_path,index=False)

def size_count(input_path,save_path):
    info = []
    for subdir in os.scandir(input_path):
        info_item = [subdir.name]
        size_list = []
        for item in os.scandir(subdir.path):
            img = Image.open(item.path)
            size_list.append(img.size)
        info_item.append(list(set(size_list)))
        info.append(info_item)
    info.sort(key=lambda x:eval(x[0]))
    col = ['class','size_list']
    csv_file = pd.DataFrame(columns=col,data=info)
    csv_file.to_csv(save_path,index=False)


def cal_mean_std(data_path):
    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    for item in tqdm(data_path):
        img = Image.open(item).convert('RGB').split()
        r_img = (np.array(img[0].resize((512,512))).astype(np.float32)/255.0).flatten()
        r_mean.append(np.mean(r_img))
        g_img = (np.array(img[1].resize((512,512))).astype(np.float32)/255.0).flatten()
        g_mean.append(np.mean(g_img))
        b_img = (np.array(img[2].resize((512,512))).astype(np.float32)/255.0).flatten()
        b_mean.append(np.mean(b_img))
    
    r_mean = np.mean(r_mean)
    g_mean = np.mean(g_mean)
    b_mean = np.mean(b_mean)

    for item in tqdm(data_path):
        img = Image.open(item).convert('RGB').split()
        r_img = (np.array(img[0].resize((512,512))).astype(np.float32)/255.0).flatten()
        r_std.append(np.mean(np.power(r_img - r_mean,2)))
        g_img = (np.array(img[1].resize((512,512))).astype(np.float32)/255.0).flatten()
        g_std.append(np.mean(np.power(g_img - g_mean,2)))
        b_img = (np.array(img[2].resize((512,512))).astype(np.float32)/255.0).flatten()
        b_std.append(np.mean(np.power(b_img - b_mean,2)))

    r_std = np.sqrt(np.mean(r_std))
    g_std = np.sqrt(np.mean(g_std))
    b_std = np.sqrt(np.mean(b_std))

    print('r mean:%.3f' % r_mean)
    print('r std:%.3f' % r_std)
    print('g mean:%.3f' % g_mean)
    print('g std:%.3f' % g_std)
    print('b mean:%.3f' % b_mean)
    print('b std:%.3f' % b_std)


if __name__ == '__main__':
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    # save_path = './adver_material_count.csv'
    # sample_count(input_path,save_path)
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    # save_path = './adver_material_size.csv'
    # size_count(input_path,save_path)
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Leve_Disease/train'
    # cal_mean_std(input_path)


    input_csv = './csv_file/leve_disease.csv'
    path_list = pd.read_csv(input_csv)['id'].values.tolist()
    test_path = '../dataset/Leve_Disease/test/'
    path_list += [os.path.join(test_path,case) for case in os.listdir(test_path)]
    cal_mean_std(path_list)