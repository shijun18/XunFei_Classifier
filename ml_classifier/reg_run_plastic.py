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

_AVAIL_CLF = ['lr','xgboost','random_forest','extra_trees','bagging']

def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def neg_rmse(y_true,y_pred):
    return -1.0*mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def log_rmse(y_true,y_pred):
    return -1.0*math.log(mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5)

METRICS_REG= {
#   'mse':make_scorer(mean_squared_error,greater_is_better=False),
#   'rmse':make_scorer(rmse,greater_is_better=False),
  'neg_rmse':make_scorer(neg_rmse),
  'log_rmse':make_scorer(log_rmse)
  }

SETUP_TRAINER = {
  'target_key':'scale_index_2',
  'random_state':21,
  'metric':METRICS_REG,
  'k_fold':10,
  'scale_factor':1240
}


if __name__ == "__main__":

    for index in [1,5,6]:
        SETUP_TRAINER['target_key'] = f'scale_index_{index}'
        if index > 4:
            SETUP_TRAINER['scale_factor'] = 1624
        else:
            SETUP_TRAINER['scale_factor'] = 1240

        train_path = './dataset/plastic_drum/train_xy.csv'
        train_df = pd.read_csv(train_path)

        test_path = './dataset/plastic_drum/test_xy.csv'
        test_df = pd.read_csv(test_path)

        index_list = [f'index_{i}' for i in range(1,7)]
        scale_index_list = [f'scale_index_{i}' for i in range(1,7)]

        fea_list = [f for f in train_df.columns if f not in ['group','image'] + index_list + scale_index_list] 
        print(fea_list)
        test_df = test_df[fea_list]
        train_df = train_df[fea_list + [f'scale_index_{index}']]

        save_path = f'./result/plastic_drum/index_{index}'
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