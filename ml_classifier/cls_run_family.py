import os
import pandas as pd 
import pickle
from cls_trainer import ML_Classifier,params_dict

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.preprocessing import LabelEncoder


# _AVAIL_CLF = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
# _AVAIL_CLF = ['random forest','extra trees','bagging','mlp','xgboost']
_AVAIL_CLF = ['decision tree','random forest']

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
    data = data.fillna(0)
    data[fea_name] = data[fea_name].apply(lambda x:int(x.split(' ')[-1].split(':')[0])*60 + int(x.split(' ')[-1].split(':')[1]))

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df

if __name__ == "__main__":

    train_path = './dataset/family/train.csv'
    train_df = pd.read_csv(train_path,encoding='gbk')
    

    test_path = './dataset/family/test.csv'
    test_df = pd.read_csv(test_path,encoding='gbk')
    
    exclude_list = []  

    fea_list = [f for f in train_df.columns if f not in ['tag','c','f_i','_id','loginname']] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['tag']]

    # train_df,test_df = fea_encoder(train_df,test_df,'loginname','tag')
    train_df,test_df = time_encoder(train_df,test_df,'_d','tag')
    
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
      model = classifier.trainer(train_df=tmp_train_df,**SETUP_TRAINER,pred_flag=True,test_df=tmp_test_df,test_csv=test_path,save_path=save_path)
    
    # save model
    # pkl_filename = "./save_model/{}.pkl".format(clf_name.replace(' ','_'))
    # with open(pkl_filename, 'wb') as file:
    #   pickle.dump(model, file)

    