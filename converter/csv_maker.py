import os 
import pandas as pd 
import glob
import random


def make_label_csv(input_path,csv_path,target_list=None):

    info = []
    for subdir in os.scandir(input_path):
        index = subdir.name
        path_list = glob.glob(os.path.join(subdir.path,"*.*g"))
        sub_info = [[item,eval(index)-1] for item in path_list]
        if target_list is not None:
            if eval(index) in target_list:
                info.extend(sub_info)
            else:
                index = len(target_list) + 1
                sub_info = [[item,index] for item in path_list]
                info.extend(sub_info)
        else:
            info.extend(sub_info)
    
    random.shuffle(info)
    # print(len(info))
    col = ['id','label']
    info_data = pd.DataFrame(columns=col,data=info)
    info_data.to_csv(csv_path,index=False)





if __name__ == "__main__":
  
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    # csv_path = './csv_file/adver_material_30.csv'
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Crop_Growth/train'
    # csv_path = './csv_file/crop_growth.csv'

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Photo_Guide/flip_train'
    # csv_path = './csv_file/photo_guide_flip_exclude.csv'
    input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Leve_Disease/train'
    csv_path = './csv_file/leve_disease.csv'
    make_label_csv(input_path,csv_path)
