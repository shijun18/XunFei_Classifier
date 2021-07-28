import random
import pandas as pd


def csv_reader_single(csv_file,key_col=None,value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
      target_dict[key_item] = value_item

    return target_dict

def csv_merge(csv_list,save_path,key_col=None,value_col=None):
    key_val_list = []
    for csv_file in csv_list:
        file_csv = pd.read_csv(csv_file)
        key_list = file_csv[key_col].values.tolist()
        value_list = file_csv[value_col].values.tolist()
        key_val_list.extend([[key,val] for key,val in zip(key_list,value_list)])
    random.shuffle(key_val_list)
    print(len(key_val_list))
    target_dict = {
      'id':[],
      'label':[]
    }
    for item in key_val_list:
        target_dict['id'].append(item[0])
        target_dict['label'].append(item[1])
    print(target_dict)
    merge_file = pd.DataFrame(data=target_dict)
    merge_file.to_csv(save_path,index=False)

if __name__ == '__main__':
    
    csv_list = ['../converter/csv_file/photo_guide.csv','../converter/csv_file/photo_guide_flip_exclude.csv', \
                '../converter/csv_file/photo_guide_flip_vertical.csv','../converter/csv_file/photo_guide_flip_vertical_horizontal.csv']
    
    merge_csv = '../converter/csv_file/photo_guide_merge_all_exclude.csv'
    csv_merge(csv_list,merge_csv,key_col='id',value_col='label')