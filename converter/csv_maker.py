import os 
import pandas as pd 
import glob
import random

RULE = {"AD":0,
        "NC":1,
        "MCI":2
        }



def make_label_csv(input_path,csv_path):
  '''
  Make label csv file.
  label rule: AD->0, NC->1, MCI->2
  '''
  info = []
  for subdir in os.scandir(input_path):
    # print(subdir.name)
    for subsubdir in os.scandir(subdir.path):
      index = RULE[subsubdir.name]
      path_list = glob.glob(os.path.join(subsubdir.path,"*.*g"))
      sub_info = [[item,index] for item in path_list]
      info.extend(sub_info)
  
  random.shuffle(info)
  # print(len(info))
  col = ['id','label']
  info_data = pd.DataFrame(columns=col,data=info)
  info_data.to_csv(csv_path,index=False)





if __name__ == "__main__":
  
  input_path = '/staff/shijun/torch_projects/AD_CLS/dataset/pre_data/train'
  csv_path = './pre_shuffle_label.csv'
  make_label_csv(input_path,csv_path)

  input_path = '/staff/shijun/torch_projects/AD_CLS/dataset/pre_crop_data/train'
  csv_path = './pre_shuffle_crop_label.csv'

  make_label_csv(input_path,csv_path)