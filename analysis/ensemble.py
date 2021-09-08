import os
import pandas as pd
import numpy as np


def vote_ensemble(csv_path_list, save_path, col='category_id'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(csv_file[col].values.tolist())

    result['image_id'] = csv_file['image_id'].values.tolist()
    vote_array = np.array(ensemble_list)
    result['category_id'] = [
        max(list(vote_array[:, i]), key=list(vote_array[:, i]).count)
        for i in range(vote_array.shape[1])
    ]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)

def diff(csv_a,csv_b,col='category_id'):
    col_a = np.array(pd.read_csv(csv_a)[col])
    col_b = np.array(pd.read_csv(csv_b)[col])
    return len(col_a) - np.sum(col_a==col_b)

if __name__ == "__main__":

    save_path = './result/Crop_Growth/fusion.csv'
    if os.path.exists(save_path):
        os.remove(save_path)
    dir_list = os.listdir('./result/Crop_Growth/')
    dir_list.sort()
    csv_path_list = [os.path.join('./result/Crop_Growth/',case + '/submission_ave.csv') for case in dir_list]
    print(csv_path_list)
    vote_ensemble(csv_path_list, save_path)
    print('diff %d with target'%(diff(save_path,'../converter/csv_file/crop_growth_fake_result.csv')))
