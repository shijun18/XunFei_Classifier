import os
import pandas as pd 
import numpy as np
from cls_trainer import ML_Classifier,params_dict

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.preprocessing import LabelEncoder


# _AVAIL_CLF = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
# _AVAIL_CLF = ['random forest','extra trees','bagging','mlp','xgboost']
# _AVAIL_CLF = ['decision tree','random forest']

_AVAIL_CLF = ['random forest']

METRICS_CLS = {
  'Accuracy':make_scorer(accuracy_score),
  'Recall':make_scorer(recall_score,average='macro',zero_division=0),
  'Precision':make_scorer(precision_score,average='macro',zero_division=0),
  'F1':make_scorer(f1_score,average='macro',zero_division=0),
  }

SETUP_TRAINER = {
  'target_key':'label',
  'random_state':21,
  'metric':METRICS_CLS,
  'k_fold':5,
  'sub_col':['sample_id','category_id'],
  'id_name':['id']
}


if __name__ == "__main__":

    train_path = '../dataset/Covid19/train.csv'
    train_df = pd.read_csv(train_path)

    test_path = '../dataset/Covid19/test.csv'
    test_df = pd.read_csv(test_path)

    # exclude_list = [f for f in train_df.columns][1:][::2]
    exclude_list = [f for f in train_df.columns][2:][:80]
    # exclude_list = [None]
    print(exclude_list)

    fea_list = [f for f in train_df.columns if f not in ['id','label'] + exclude_list] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['label']]


    save_path = './result/covid19'
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
      model = classifier.trainer(train_df=tmp_train_df,**SETUP_TRAINER,pred_flag=True,test_df=tmp_test_df,test_csv=test_path,save_path=save_path,encoder_flag=True,scale_flag=True)
    
    # save model
    # pkl_filename = "./save_model/{}.pkl".format(clf_name.replace(' ','_'))
    # with open(pkl_filename, 'wb') as file:
    #   pickle.dump(model, file)

    