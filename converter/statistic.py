import os
import pandas as pd
from PIL import Image

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



if __name__ == '__main__':
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    # save_path = './adver_material_count.csv'
    # sample_count(input_path,save_path)
    input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    save_path = './adver_material_size.csv'
    size_count(input_path,save_path)