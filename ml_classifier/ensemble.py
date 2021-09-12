import os
import pandas as pd

def plastic_ensemble():
    result_path = './result/plastic_drum'
    for subdir in os.scandir(result_path):
        if subdir.is_dir():
            save_path = os.path.join(result_path,subdir.name + '.csv')
            info = {}
            for csv_file in os.scandir(subdir.path):
                df = pd.read_csv(csv_file.path)
                info['group'] = df['group'].values.tolist()
                info['image'] = df['image'].values.tolist()
                info[csv_file.name.split('.')[0]] = df[f'scale_{subdir.name}'].values.tolist()
            new_df = pd.DataFrame(data=info)
            new_df.to_csv(save_path,index=False)

    # merge
    info = {}
    subitem_list = os.listdir(result_path)
    subitem_list.sort()
    subitem_list = [os.path.join(result_path,case) for case in subitem_list]
    for subitem in subitem_list:
        if os.path.isfile(subitem):
            df = pd.read_csv(subitem)
            for col in df.columns[2:]:
                if col not in info.keys():
                    info[col] = {}
                    info[col]['group'] = df['group']
                    info[col]['image'] = df['image']
                info[col][os.path.basename(subitem).split('.')[0]] = df[col].values.tolist()
    
    for item in info:
        df = pd.DataFrame(data=info[item])
        df.to_csv(f'./result/plastic_drum/{item}.csv',index=False)


def plastic_make_csv():
    test_path = '../dataset/Plastic_Drum/test'
    info = {
        'group':[],
        'image':[],
        'index_1':[],
        'index_2':[],
        'index_3':[],
        'index_4':[],
        'index_5':[],
        'index_6':[]
        }
    subdir_list = os.listdir(test_path)
    subdir_list.sort()
    subdir_list = [os.path.join(test_path,case) for case in subdir_list]
    for subdir in subdir_list:
        subitem = os.listdir(subdir)
        subitem.sort(key=lambda x:int(x.split('.')[0]))
        info['group'].extend([int(os.path.basename(subdir))]*len(subitem))
        info['image'].extend(subitem)
        for i in range(1,7):
            info[f'index_{i}'].extend([-1]*len(subitem))
    
    df = pd.DataFrame(data=info)
    df.to_csv('./result/plastic_drum/test_base_result.csv',index=False)


def plastic_merge():
    test_base_csv = './result/plastic_drum/test_base_result.csv'
    pred_csv = './result/plastic_drum/lr_result.csv'
    test_df = pd.read_csv(test_base_csv)
    pred_df = pd.read_csv(pred_csv)
    test_info = {}
    for row in test_df.iterrows():
        if row[1][0] not in test_info.keys():
            test_info[row[1][0]] = {}
        test_info[row[1][0]][row[1][1]] = list(row[1][2:8])
    for row in pred_df.iterrows():
        test_info[row[1][0]][row[1][1]] = list(row[1][2:8])
    # print(test_info)

    new_info = []
    col = ['group','image','index_1','index_2','index_3','index_4','index_5','index_6']
    for group in test_info:
        for item in test_info[group]:
            info_item = []
            info_item.append(group)
            info_item.append(item)
            info_item.extend(test_info[group][item])
            new_info.append(info_item)
    
    new_df = pd.DataFrame(data=new_info,columns=col)
    new_df.to_csv('./result/plastic_drum/merge_lr_result.csv',index=False)

if __name__ == '__main__':
   
    # plastic_ensemble()
    # plastic_make_csv()
    plastic_merge()