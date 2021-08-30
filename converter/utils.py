import os 
import shutil
import pandas as pd

# def rename(input_path):

#     for subdir in os.scandir(input_path):
#         for i,item in enumerate(os.scandir(subdir.path)):
#             new_name = os.path.join(os.path.dirname(item.path),str(i) + os.path.splitext(item.name)[1])
#             os.rename(item.path,new_name)

# rename('/staff/shijun/torch_projects/XunFei_Classifier/dataset/Farmer_Work/train-external-v2')

def disassemble_test(csv_path,source_path,dest_path):
    image_id = pd.read_csv(csv_path)['image_id'].values.tolist()
    category_id = pd.read_csv(csv_path)['category_id'].values.tolist()

    for img,lab in zip(image_id,category_id):
        save_dir = os.path.join(dest_path,str(lab))
        # print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        old_path = os.path.join(source_path,img)
        new_path = os.path.join(save_dir,img)
        shutil.copy(old_path,new_path)

csv_path = '../analysis/result/Farmer_Work/v6.0-crop-external-pretrained-lite/submission_ave.csv'
source_path = '../dataset/Farmer_Work/test'
dest_path = '../dataset/Farmer_Work/pre-test'

disassemble_test(csv_path,source_path,dest_path)