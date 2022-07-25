import os
import pandas as pd
import numpy as np


def vote_ensemble(csv_path_list, save_path, key='file', col='label'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(csv_file[col].values.tolist())

    result[key] = csv_file[key].values.tolist()
    vote_array = np.array(ensemble_list)
    result[col] = [
        max(list(vote_array[:, i]), key=list(vote_array[:, i]).count)
        for i in range(vote_array.shape[1])
    ]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)


def prob_ensemble(csv_path_list, save_path, key='file',col='label'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(np.asarray(csv_file[[col_name for col_name in csv_file.columns if col_name not in [col,key]]]))
    ensemble_array = np.asarray(ensemble_list)
    print(ensemble_array.shape)
    ensemble_array = np.mean(ensemble_array,axis=0)
    ensemble_result = np.argmax(ensemble_array,axis=1)
    result[key] = csv_file[key].values.tolist()
    result[col] = list(ensemble_result)

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)

def diff(csv_a,csv_b,col='category_id'):
    col_a = np.array(pd.read_csv(csv_a)[col])
    col_b = np.array(pd.read_csv(csv_b)[col])
    return len(col_a) - np.sum(col_a==col_b)

if __name__ == "__main__":

    save_path = './result/Package/fusion_vote_vote_tta11.csv'
    if os.path.exists(save_path):
        os.remove(save_path)
    dir_list = [case for case in os.listdir('./result/Package/') if case not in ['v6.0-pretrained']]
    dir_list.sort(reverse=True)
    csv_path_list = [os.path.join('./result/Package/',case + '/submission_vote.csv') for case in dir_list[1:]]
    print(csv_path_list)
    vote_ensemble(csv_path_list, save_path)
    # prob_ensemble(csv_path_list, save_path)
    # print('diff %d with target'%(diff(save_path,'../converter/csv_file/temp_freq_fake_result.csv')))
