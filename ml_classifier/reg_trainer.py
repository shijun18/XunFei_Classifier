import os 
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

params_dict = {
    'random_forest':{
        'n_estimators':range(10,150,10),
        'criterion':['mse']
    },
    'extra_trees':{
        'n_estimators':range(10,150,10),
        'criterion':['mse']
    },
    'bagging':{
        'n_estimators':range(10,150,10),
    },
    'mlp':{
        'alpha':[0.001,0.01,0.1],
        'hidden_layer_sizes':[(5,),(10,),(15,),(20,),(10,10),(7,7,7),(5,5,5,5)],
        'solver':['lbfgs','sgd', 'adam'],
        'activation':['identity'],
        'learning_rate':['constant','invscaling']
    },
    # 'xgboost':{
    #     'n_estimators':range(100,150,5),
    #     'max_depth':range(10,15,2),
    #     'learning_rate':np.linspace(0.05,0.1,2),
    #     'subsample':np.linspace(0.7,0.9,2)
    # }
    'xgboost':{
        'n_estimators':[10000],
        'max_depth':[15],
        'learning_rate':[0.05],
        'subsample':[0.5]
    },
    'lgb':{
        'max_depth':[15],
        'num_leaves':[35],
        'learning_rate':[0.03],
        'n_estimators':[5000]
    },
    'lr':{
        'normalize':[False]
    }
}


class ML_Classifier(object):
  '''
  Machine Learning Classifier for the classification
  Args:
  - clf_name, string, __all__ = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
  - params_dict, dict, parameters setting of the specified classifer
  '''
  def __init__(self,clf_name=None,params=None): 
    super(ML_Classifier,self).__init__()  
    self.clf_name = clf_name
    self.params = params
    self.clf = self._get_clf()  
  

  def trainer(self,train_df,target_key,random_state=21,metric=None,k_fold=5,pred_flag=False,test_df=None,test_csv=None,save_path=None):
    params = self.params
    fea_list= [f for f in train_df.columns if f != target_key]
    # print('feature list:',fea_list)
    kfold = KFold(n_splits=k_fold,shuffle=True,random_state=random_state)
    
    answers = []
    for fold_num,(train_index,val_index) in enumerate(kfold.split(train_df)):
        print(f'***********fold {fold_num+1} start!!***********')
        x_train, x_val = train_df[fea_list].iloc[train_index], train_df[fea_list].iloc[val_index]
        y_train, y_val = train_df[target_key][train_index], train_df[target_key][val_index]
        # print(x_val,y_val)
        model = GridSearchCV(estimator=self.clf,
                            param_grid=params,
                            cv=kfold,
                            scoring=metric,
                            refit='neg_rmse',
                            verbose=True,
                            return_train_score=True)
        model = model.fit(x_train,y_train)

        best_score = -1.0*model.best_score_
        best_model = model.best_estimator_
        train_pred = model.predict(x_train)
        train_score = rmse(y_train,train_pred)
        # test_score = best_model.score(x_val,y_val)
        test_pred = model.predict(x_val)
        test_score = rmse(y_val,test_pred)
        print("MSE Evaluation:")
        print("fold {} Best score:{}".format(fold_num + 1,best_score))
        print("fold {} Train score:{}".format(fold_num + 1,train_score))
        print("fold {} Test score:{}".format(fold_num + 1,test_score))
        print("fold {} Best parameter:\n".format(fold_num + 1))
        for key in params.keys():
            print('%s:'%key)
            print(best_model.get_params()[key])

        '''
        if self.clf_name == 'random_forest' or self.clf_name == 'extra_trees':
            if self.clf_name == 'random_forest':
                new_grid = RandomForestRegressor(random_state=0,bootstrap=True)
            elif self.clf_name == 'extra_trees':
                new_grid = ExtraTreesRegressor(random_state=0,bootstrap=True)
            new_grid.set_params(**grid.best_params_)

            new_grid = new_grid.fit(x_train,y_train) 
            importances = new_grid.feature_importances_
            feat_labels = fea_list
            # print(feat_labels)
            indices = np.argsort(importances)[::-1]

            # print(indices)
            for f in range(x_train.shape[1]):
                print("%2d) %-*s %f" % (f + 1,30, feat_labels[indices[f]], importances[indices[f]]))
        '''
        if pred_flag and test_df is not None:
            pred = model.predict(test_df)
            answers.append(pred)
    
    pred_df = pd.read_csv(test_csv)
    pred_df['date'] = pred_df['日期']
    pred_df[target_key] = sum(answers) / k_fold
    pred_df[['date',target_key]].to_csv(f'{save_path}/{self.clf_name}_result.csv',index=False)
    return best_model

    
  def _get_clf(self):
    if self.clf_name == 'xgboost':
        classifier = XGBRegressor()
    elif self.clf_name == 'mlp':
        classifier = MLPRegressor(max_iter=2000,warm_start=True,random_state=2021)
    elif self.clf_name == 'random_forest':
        classifier = RandomForestRegressor(random_state=2021,bootstrap=True)
    elif self.clf_name == 'extra_trees':
        classifier = ExtraTreesRegressor(random_state=2021,bootstrap=True)
    elif self.clf_name == 'bagging':
        classifier = BaggingRegressor(random_state=2021)
    elif self.clf_name == 'lgb':
        classifier = lgb.LGBMRegressor(seed=2021)
    elif self.clf_name == 'lr':
        classifier = LinearRegression()


    return classifier  



