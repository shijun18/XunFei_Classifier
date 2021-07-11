import os
import pandas as pd 
import pickle
from reg_trainer import ML_Classifier,params_dict

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



_AVAIL_CLF = ['lr','xgboost','lgb','mlp','random_forest','extra_trees','bagging']


def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def neg_rmse(y_true,y_pred):
    return -1.0*mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def preprocess(train_df,test_df,encoder_list=['qua']):
    train_df['日期'] = pd.to_datetime(train_df['日期'], format='%Y/%m/%d')
    train_df["month"]= train_df["日期"].apply(lambda x : x.month)
    
    test_df['日期'] = pd.to_datetime(test_df['日期'], format='%Y/%m/%d')
    test_df["month"]= test_df["日期"].apply(lambda x : x.month)
    label_coder = LabelEncoder()
    for col in encoder_list:
        train_df[col] = label_coder.fit_transform(train_df[col])
        test_df[col] = label_coder.transform(test_df[col])
    return train_df,test_df



METRICS_REG= {
#   'mse':make_scorer(mean_squared_error,greater_is_better=False),
#   'rmse':make_scorer(rmse,greater_is_better=False),
  'neg_rmse':make_scorer(neg_rmse)
  }

SETUP_TRAINER = {
  'target_key':'IPRC',
  'random_state':21,
  'metric':METRICS_REG,
  'k_fold':5
}


if __name__ == "__main__":

    train_path = './dataset/air/pre_train/保定2016年.csv'
    train_df = pd.read_csv(train_path)

    test_path = './dataset/air/pre_test/石家庄20160701-20170701.csv'
    test_df = pd.read_csv(test_path)

    train_df,test_df = preprocess(train_df,test_df,encoder_list=['质量等级'])

    save_path = './result/air/pre'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    del train_df["日期"]
    del test_df["日期"] 
    
    # clf_name = 'xgboost' 
    # classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
    # model = classifier.trainer(train_df=train_df,**SETUP_TRAINER,pred_flag=True,test_df=test_df,test_csv=test_path,save_path=save_path)
    for clf_name in _AVAIL_CLF[2:]:
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