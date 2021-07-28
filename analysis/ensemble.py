import pandas as pd
import numpy as np


def vote_ensemble(csv_path_list, save_path, col='label'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(csv_file[col].values.tolist())

    result['image'] = csv_file['image'].values.tolist()
    vote_array = np.array(ensemble_list)
    result['label'] = [
        max(list(vote_array[:, i]), key=list(vote_array[:, i]).count)
        for i in range(vote_array.shape[1])
    ]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)


if __name__ == "__main__":

    save_path = './fusion.csv'
    csv_path_list = ['./result/Photo_Guide/v5.2-alldata/submission_ave.csv', \
                     './result/Photo_Guide/v5.2-merge/submission_ave.csv', \
                     './result/Photo_Guide/v5.0/submission_ave.csv']
    vote_ensemble(csv_path_list, save_path)
