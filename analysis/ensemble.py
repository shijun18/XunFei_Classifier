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


if __name__ == "__main__":

    save_path = './result/Leve_Disease/fusion.csv'
    csv_path_list = ['./result/Leve_Disease/v6.2-pretrained/submission_ave.csv', \
                     './result/Leve_Disease/v7.2-pretrained/submission_ave.csv', \
                     './result/Leve_Disease/v8.0-pretrained/submission_ave.csv']
    vote_ensemble(csv_path_list, save_path)
