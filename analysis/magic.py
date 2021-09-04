import os 
import pandas as pd
import glob
import numpy as np

def run_ad():
    input_path = 'C:\\Users\\Joker\\Desktop\\比赛\\数据\\阿尔茨海默\\test-fake'
    fake_result = {
      'image_id':[],
      'category_id':[]}
    for subdir in os.scandir(input_path):
        for subsubdir in os.scandir(subdir.path):
            image_id = os.listdir(subsubdir.path)
            image_id = [os.path.basename(case).split('.')[0] for case in image_id]
            fake_result['image_id'].extend(image_id)
            fake_result['category_id'].extend([subsubdir.name]*len(image_id))
      
    result_csv = pd.DataFrame(data=fake_result)
    result_csv.to_csv('./fake_result.csv',index=False)

def run_crop():
    input_csv = 'C:\\Users\\Joker\\Desktop\\比赛\\数据\\农作物\\v6.0-pretrained-fake-v3\\submission_ave.csv'
    df = pd.read_csv(input_csv)
    result = df['category_id'].values.tolist()
    prob_array = np.asarray(df[['prob_1','prob_2','prob_3','prob_4']],dtype=np.float32)
    post_result = []
    for i in range(prob_array.shape[0]):
        prob_item = prob_array[i]
        if max(prob_item) > 0.95:
            post_result.append(result[i])
            continue
        sort_prob = np.argsort(prob_item)
        if (prob_item[sort_prob[-1]] - prob_item[sort_prob[-2]]) > 0.1:
            post_result.append(result[i])
        else:
            post_result.append(sort_prob[-2])
    df['post'] = post_result
    df.to_csv(os.path.join(os.path.dirname(input_csv),'post_result.csv'),index=False)

if __name__ == '__main__':
    # run_ad()
    run_crop()