import os
from numpy import fabs
from numpy.core.fromnumeric import mean
import pandas as pd 
import pickle
from reg_trainer import ML_Classifier,params_dict
import math
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# _AVAIL_CLF = ['lr','xgboost','lgb','mlp','random_forest','extra_trees','bagging']
# _AVAIL_CLF = ['random_forest','extra_trees','bagging']
_AVAIL_CLF = ['mlp']
# _AVAIL_CLF = ['lr']
# _AVAIL_CLF = ['poly']
# _AVAIL_CLF = ['mlp']

def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def neg_rmse(y_true,y_pred):
    return -1.0*mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def log_rmse(y_true,y_pred):
    return -1.0*math.log(mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5)


def date_encoder(df,key='日期'):
    # df["year"] = pd.to_datetime(df[key]).dt.year
    df["month"] = pd.to_datetime(df[key]).dt.month
    df["day"] = pd.to_datetime(df[key]).dt.day
    del df[key]
    return df

def onehot_encoder(df,key='质量等级',label_list=['重度污染', '良', '中度污染', '轻度污染', '严重污染']):
    ff = pd.get_dummies(df[key].values)
    for label in label_list:
        df[label] = ff[label]
    del df[key]
    return df


def scaler_normalize(train_df,test_df,scale_list=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data = data.fillna(0)
    data = date_encoder(data)
    data = onehot_encoder(data)


    if scale_list is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        for col in scale_list:
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df


def new_normalize(df,factor_dict):
    for col,factor in factor_dict.items():
        df[col] = df[col].apply(lambda x : x / factor)
    return df

factor_dict = {
    'AQI':500,
    "PM2.5":500, 
    "PM10":600
}




METRICS_REG= {
#   'mse':make_scorer(mean_squared_error,greater_is_better=False),
#   'rmse':make_scorer(rmse,greater_is_better=False),
  'neg_rmse':make_scorer(neg_rmse),
  'log_rmse':make_scorer(log_rmse)
  }

SETUP_TRAINER = {
  'target_key':'IPRC',
  'random_state':21,
  'metric':METRICS_REG,
  'k_fold':10,
  'scaler':False
}


if __name__ == "__main__":

    train_path = './dataset/air/pre_train/保定2016年.csv'
    train_df = pd.read_csv(train_path)

    test_path = './dataset/air/pre_test/石家庄20160701-20170701.csv'
    test_df = pd.read_csv(test_path)

    # train_path = './dataset/air/pre_train/保定2016年_IAQI.csv'
    # train_df = pd.read_csv(train_path)

    # test_path = './dataset/air/pre_test/石家庄20160701-20170701_IAQI.csv'
    # test_df = pd.read_csv(test_path)

    
    raw_list = ["AQI","PM2.5", "PM10"]
    iaqi_list = ["PM2.5_IAQI", "PM10_IAQI"]
    exclude_iaqi_list = ["SO2_IAQI", "CO_IAQI", "NO2_IAQI", "O3_8h_IAQI"]
    extra_list = ['month','day'] 
    serious_list =['重度污染', '良', '中度污染', '轻度污染', '严重污染']
    
    scale_list = raw_list
    # # scale_list = None
    train_df,test_df = scaler_normalize(train_df,test_df,scale_list,'IPRC')
    
    # train_df = new_normalize(train_df,factor_dict)
    # test_df = new_normalize(test_df,factor_dict)

    fea_list = [f for f in train_df.columns if f not in ['IPRC','日期','质量等级'] + ["PM10","SO2", "CO", "NO2", "O3_8h"]] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['IPRC']]

    save_path = './result/air/pre'
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