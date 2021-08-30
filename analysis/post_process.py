import sys
sys.path.append("..")
from analysis.cluster import union_find,read_csv,merge_class

import pandas as pd 
import os 


def post_process(input_df,cluster_dict,save_path):
    id_list = input_df['image_id'].values.tolist()
    post_label = input_df['category_id'].values.tolist()
    
    for value in cluster_dict.values():
        distrubution = [post_label[id_list.index(f'a{str(index)}.jpg')]  for index in value]
        label = max(distrubution,key=distrubution.count)
        for index in value:
            post_label[id_list.index(f'a{str(index)}.jpg')] = label  

     
    input_df['post_label'] = post_label
    input_df.to_csv(save_path,index=False)


def diff_csv(pred_csv,target_csv,pred_key='post_label',target_key='category_id'):
    pred_list = pd.read_csv(pred_csv)[pred_key].values.tolist() 
    target_list = pd.read_csv(target_csv)[target_key].values.tolist() 
    count = 0
    for i,j in zip(pred_list,target_list):
        if i!=j:
            count += 1
    print('diff with target = %d'%count)        
    
    return count


if __name__ == "__main__":

    threshold = 0.40
    threshold_list = []
    for i in range(100):
        print("*****%.3f*****"%threshold)
        sim_path = "/staff/honeyk/project/XunFei_Classifier-main/analysis/sim_csv/farmer_worker/test_sim.csv"
        input_path = './result/Farmer_Work/v6.0-crop-external-pretrained-new/submission_ave.csv'

        data = read_csv(sim_path)
        class_result = union_find(data,threshold=threshold)
        print("types = %d"%len(set(class_result)))
        
        input_df = pd.read_csv(input_path)
        raw_label = input_df['category_id'].values.tolist() 
        id_list = input_df['image_id'].values.tolist()

        cluster_dict = merge_class(class_result) 
        save_path = '{}/post_{}'.format(os.path.dirname(input_path),os.path.basename(input_path))   

        post_process(input_df,cluster_dict,save_path)

        post_label = pd.read_csv(save_path)['post_label'].values.tolist() 

        for i in range(len(id_list)):
            if raw_label[i] != post_label[i]:
                print(id_list[i])
                print('raw:',raw_label[i])
                print('post:',post_label[i])
                
        threshold_list.append(threshold)
        target_csv = input_path # It is an option to choose the csv that you have submitted
        diff = diff_csv(save_path,target_csv)
        
        threshold += 0.01
        # break
    print(threshold_list)