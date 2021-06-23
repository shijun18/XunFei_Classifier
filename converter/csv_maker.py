import os 
import pandas as pd 
import glob
import random


def make_label_csv(input_path,csv_path):
  '''
  Make label csv file.
  label rule: AD->0, NC->1, MCI->2
  '''
  info = []
  for subdir in os.scandir(input_path):
    index = subdir.name
    path_list = glob.glob(os.path.join(subdir.path,"*.*g"))
    sub_info = [[item,index] for item in path_list]
    info.extend(sub_info)
  
  random.shuffle(info)
  # print(len(info))
  col = ['id','label']
  info_data = pd.DataFrame(columns=col,data=info)
  info_data.to_csv(csv_path,index=False)





if __name__ == "__main__":
  
  # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
  # csv_path = './csv_file/adver_material.csv'
  input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Crop_Growth/train'
  csv_path = './csv_file/crop_growth.csv'
  make_label_csv(input_path,csv_path)
