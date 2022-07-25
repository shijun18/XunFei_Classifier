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
        sub_info = [[item,eval(index)-1] for item in path_list]
        # sub_info = [[os.path.basename(item),index] for item in path_list]
        # sub_info = [[item,index] for item in path_list]
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
  
    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Photo_Guide_V2/raw_data/fliped_v_train'
    # csv_path = './csv_file/photo_guide_v2_flip_v.csv'
    # make_label_csv(input_path,csv_path)

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Photo_Guide_V2/raw_data/fliped_h_train'
    # csv_path = './csv_file/photo_guide_v2_flip_h.csv'
    # make_label_csv(input_path,csv_path)

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Photo_Guide_V2/raw_data/test_fake'
    # csv_path = './csv_file/photo_guide_v2_fake.csv'
    # make_label_csv(input_path,csv_path)

    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/LED/train'
    # csv_path = './csv_file/led.csv'
    # make_label_csv(input_path,csv_path)


    # input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Package/train-post-five/c1'
    # csv_path = './csv_file/package_post_c1.csv'
    # make_label_csv(input_path,csv_path)


    input_path = '/staff/shijun/torch_projects/XunFei_Classifier/dataset/Family_Env_V2/train'
    csv_path = './csv_file/family_env_v2.csv'
    make_label_csv(input_path,csv_path)