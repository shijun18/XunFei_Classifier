import os
import pandas as pd 
import pickle
from cls_trainer import ML_Classifier,params_dict
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.preprocessing import LabelEncoder


# _AVAIL_CLF = ['lasso','knn','decision tree','random forest','extra trees','bagging','mlp','xgboost']
# _AVAIL_CLF = ['decision tree','random forest']
_AVAIL_CLF = ['lgb']
# _AVAIL_CLF = ['random forest']
# _AVAIL_CLF = ['xgboost']

METRICS_CLS = {
    'Accuracy':make_scorer(accuracy_score),
    'Recall':make_scorer(recall_score,average='macro',zero_division=0),
    'Precision':make_scorer(precision_score,average='macro',zero_division=0),
    'F1':make_scorer(f1_score,average='macro',zero_division=0),
  }

SETUP_TRAINER = {
  'target_key':'患有糖尿病标识',
  'random_state':21,
  'metric':METRICS_CLS,
  'k_fold':5,
  'scale_flag':False,
  'sub_col':['uuid','label'],
  'id_name':['编号']
}


def fea_encoder(train_df,test_df,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    data['糖尿病家族史'] = data['糖尿病家族史'].apply(
        lambda x:'叔叔或姑姑有一方患有糖尿病' if x=='叔叔或者姑姑有一方患有糖尿病' else x)
    df = pd.get_dummies(data['糖尿病家族史']).astype('int')
    data = pd.concat([data,df],axis=1)
    #修正
    data['肱三头肌皮褶厚度'] = data['肱三头肌皮褶厚度'].apply(lambda x:x/10.0 if x>10 else x)

    for i in ['口服耐糖量测试','胰岛素释放实验','肱三头肌皮褶厚度','体重指数']:
        data[i] = data[i].apply(lambda x:np.nan if x==0 or x==-1 else x)

    # data.fillna(data.median(axis=0), inplace=True) 
    
    
    data['age'] = data['出生年份'].apply(lambda x:int(2022 - x))

    data['BMI>25'] = data['体重指数'].apply(lambda x:int(x > 25))
    data['30>BMI>=25'] = data['体重指数'].apply(lambda x:int(x >= 25 and x < 30))
    data['BMI>=30'] = data['体重指数'].apply(lambda x:int(x >= 30))
    data['BMI>50'] = data['体重指数'].apply(lambda x:int(x > 50))

    data['OGTT<7.8'] = data['口服耐糖量测试'].apply(lambda x:int(x < 7.8))
    # data['OGTT>7.1'] = data['口服耐糖量测试'].apply(lambda x:int(x > 7.1))
    data['OGTT>8.5'] = data['口服耐糖量测试'].apply(lambda x:int(x > 8.5))
    data['OGTT>11.1'] = data['口服耐糖量测试'].apply(lambda x:int(x > 11.1))
    # print(data['age'])

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df

'''
def fea_encoder(train_df,test_df,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    data['糖尿病家族史'] = data['糖尿病家族史'].apply(
        lambda x:'叔叔或姑姑有一方患有糖尿病' if x=='叔叔或者姑姑有一方患有糖尿病' else x)
    df = pd.get_dummies(data['糖尿病家族史']).astype('int')
    data = pd.concat([data,df],axis=1)

    #修正
    data['肱三头肌皮褶厚度'] = data['肱三头肌皮褶厚度'].apply(lambda x:x/10.0 if x>10 else x)

    data.fillna(data.median(axis=0), inplace=True) 
    # for i in ['口服耐糖量测试','胰岛素释放实验','肱三头肌皮褶厚度','体重指数']:
        # data[i] = data[i].apply(lambda x:np.nan if x<=0 else x)
        # data[i] = data[i].apply(lambda x:np.nan if x==0 else x)
    
    data['age'] = data['出生年份'].apply(lambda x:int(2022 - x))
    data['BMI>50'] = data['体重指数'].apply(lambda x:int(x > 50))
    data['OGTT>7.1'] = data['口服耐糖量测试'].apply(lambda x:int(x > 7.1))
    data['OGTT>8.5'] = data['口服耐糖量测试'].apply(lambda x:int(x > 8.5))
    data['OGTT>11.1'] = data['口服耐糖量测试'].apply(lambda x:int(x > 11.1))
    # print(data['age'])

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df
'''

if __name__ == "__main__":

    train_path = './dataset/diabetes/train.csv'
    train_df = pd.read_csv(train_path,encoding='gbk')

    # test_fake_path = './dataset/diabetes/test_fake.csv'
    # test_fake_df = pd.read_csv(test_fake_path,encoding='gbk')

    # train_df = pd.concat([train_df,test_fake_df], axis=0, ignore_index=True)

    test_path = './dataset/diabetes/test.csv'
    test_df = pd.read_csv(test_path,encoding='gbk')

    train_df,test_df = fea_encoder(train_df,test_df,'患有糖尿病标识')

    fea_list = [f for f in train_df.columns if f not in ['编号','患有糖尿病标识','糖尿病家族史']] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['患有糖尿病标识']]
    
    save_path = './result/diabetes_fake'
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

    