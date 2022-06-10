import os 
import pandas as pd 
import glob
import random
import shutil

def make_label_csv(input_path,csv_path,target_list=None):

    info = []
    for subdir in os.scandir(input_path):
        # if subdir.name == '1':
        #     continue
        index = subdir.name
        path_list = glob.glob(os.path.join(subdir.path,"*.*g"))
        # sub_info = [[item,eval(index)] for item in path_list]
        sub_info = [[item,index] for item in path_list]
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



def make_csv(input_path,csv_path):
    id_list = glob.glob(os.path.join(input_path,'*.*g'))
    print(len(id_list))
    info = {'id':[]}
    info['id'] = id_list
    df = pd.DataFrame(data=info)
    df.to_csv(csv_path,index=False)



if __name__ == "__main__":
  
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Adver_Material/train'
    # csv_path = './csv_file/adver_material_30.csv'
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Crop_Growth/train'
    # csv_path = './csv_file/crop_growth.csv'

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Photo_Guide/h_after_v_flip_train'
    # csv_path = './csv_file/photo_guide_flip_vertical_horizontal.csv'
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Temp_Freq/train'
    # csv_path = './csv_file/temp_freq.csv'
    # make_label_csv(input_path,csv_path)

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Farmer_Work/post-test-lite'
    # csv_path = './csv_file/farmer_work_post_test_lite.csv'
    # make_label_csv(input_path,csv_path)

    
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Leve_Disease/processed_train'
    # csv_path = './csv_file/processed_leve_disease.csv'
    # make_label_csv(input_path,csv_path)


    input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Crop_Growth/test-fake'
    csv_path = './csv_file/crop_growth_test_fake.csv'
    make_label_csv(input_path,csv_path)