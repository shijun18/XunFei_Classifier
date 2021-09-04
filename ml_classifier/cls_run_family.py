import os
import pandas as pd 
import pickle
from cls_trainer import ML_Classifier,params_dict
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.preprocessing import LabelEncoder
import random

# _AVAIL_CLF = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
# _AVAIL_CLF = ['random forest','extra trees','bagging','mlp','xgboost']
_AVAIL_CLF = ['random forest','extra trees','bagging']
# _AVAIL_CLF = ['lasso','knn','decision tree']

METRICS_CLS = {
  'Accuracy':make_scorer(accuracy_score),
  'Recall':make_scorer(recall_score,average='macro',zero_division=0),
  'Precision':make_scorer(precision_score,average='macro',zero_division=0),
  'F1':make_scorer(f1_score,average='macro',zero_division=0),
  }

SETUP_TRAINER = {
  'target_key':'tag',
  'random_state':21,
  'metric':METRICS_CLS,
  'k_fold':5,
  'sub_col':['_id','tag'],
  'id_name':['_id']
}



def fea_encoder(train_df,test_df,fea_name=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data = data.fillna(0)
    le = LabelEncoder()
    data[fea_name] = le.fit_transform(data[fea_name])

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df

def time_encoder(train_df,test_df,fea_name=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    # data = data.fillna(0)
    # data[fea_name] = data[fea_name].apply(lambda x:int(x.split(' ')[-1].split(':')[0])*60 + int(x.split(' ')[-1].split(':')[1]))
    data['datetime'] = pd.to_datetime(data[fea_name])
    # data['year'] = data["datetime"].dt.year
    data['month'] = data["datetime"].dt.month/12.0
    data['day'] = data["datetime"].dt.day/31.0
    data['hour'] = data["datetime"].dt.hour/24.0
    data['minute'] = data["datetime"].dt.minute/60.0
    data['second'] = data["datetime"].dt.second/60.0
    # print(data)
    del data['datetime']
    del data[fea_name]
    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    
    return train_df,test_df

def state_machine(df,fea_list,diff_fea='_d'):
    def compute_diff_factor(time_array):
        factor_list = [1.0]*time_array.shape[1]
        for i in range(len(factor_list)-1):
            item = time_array[...,i] 
            item_next = time_array[...,i+1]
            sub_item = item_next - item
            if sub_item[0] != 0 or sub_item[1] != 0:
                factor_list[i] = 0
            elif sub_item[2] != 0:
                factor_list[i] = 1.0 - (sub_item[2]/59)
            elif sub_item[2] != 3:
                factor_list[i] = 0.5 - (sub_item[2]/59)/2
        return factor_list
    datetime = pd.to_datetime(df[diff_fea])
    day = datetime.dt.day
    hour = datetime.dt.hour
    minute = datetime.dt.minute
    second = datetime.dt.second 
    time_array = np.asarray([day,hour,minute,second]).astype(np.float32)
    diff_factor = compute_diff_factor(time_array)
    print(len(diff_factor))
    for fea in fea_list:
        tmp_fea = df[fea].values.tolist()
        for i in range(1,len(tmp_fea)):
            tmp_fea[i] += diff_factor[i-1]*tmp_fea[i-1]
                
        df[fea] = tmp_fea
    return df

if __name__ == "__main__":

    train_path = './dataset/family/pre_train.csv'
    test_path = './dataset/family/pre_test.csv'
    stream_list = [f'stream_{str(i)}' for i in range(8)]

    # train_path = './dataset/family/pre_train_v2.csv'
    # test_path = './dataset/family/pre_test_v2.csv'
    # stream_list = [f'stream_{str(i)}' for i in range(14)]

    train_df = pd.read_csv(train_path)[::3]
    test_df = pd.read_csv(test_path)

    # train_df = state_machine(train_df,[f'stream_{str(i)}' for i in range(14)])
    # test_df = state_machine(test_df,[f'stream_{str(i)}' for i in range(14)])

    # print(np.asarray(train_df)[1,5:])

    # exclude_list = ['_id'] 
    # exclude_list = ['_id','loginname'] + stream_list
    exclude_list = ['_id','loginname','f_i','_d']
    

    fea_list = [f for f in train_df.columns if f not in ['tag'] + exclude_list] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['tag']]

    for fea_name in fea_list:
        if fea_name not in stream_list + ['_d']:
            train_df,test_df = fea_encoder(train_df,test_df,fea_name,'tag')

    # train_df,test_df = time_encoder(train_df,test_df,'_d','tag')
    
    save_path = './result/family'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # clf_name = 'xgboost' 
    # classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
    # model = classifier.trainer(train_df=train_df,**SETUP_TRAINER,pred_flag=True,test_df=test_df,test_csv=test_path,save_path=save_path)
    
    for clf_name in _AVAIL_CLF:
      import copy
      tmp_train_df = copy.copy(train_df)
      tmp_test_df = copy.copy(test_df)
      print('********** %s **********'%clf_name)
      classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
      model = classifier.trainer(train_df=tmp_train_df,**SETUP_TRAINER,pred_flag=True,test_df=tmp_test_df,test_csv=test_path,save_path=save_path,scale_flag=False)
    
    # save model
    # pkl_filename = "./save_model/{}.pkl".format(clf_name.replace(' ','_'))
    # with open(pkl_filename, 'wb') as file:
    #   pickle.dump(model, file)

  