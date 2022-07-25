import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def f1(true,pred):
    return f1_score(true,pred,average='macro',zero_division=0)


params_dict = {
  'lasso':{
    'C':[0.001,0.01,0.1,1,10]
  },
  'knn':{
    'n_neighbors':range(1,12)
  },
  'svm':{
    'C':[0.001,0.01,0.1,1,10],
    'gamma':[0.001,0.01,0.1,1,10]
  },
  'decision tree':{
    'max_depth':range(3,15),
  },
  'random forest':{
    'n_estimators':[50,100,200],
    'criterion':['entropy']
  },
  'extra trees':{
    'n_estimators':[50,100,200],
    'criterion':['entropy']
  },
  'bagging':{
    'n_estimators':[100],
  },
  'mlp':{
    'alpha':[0.001],
    'hidden_layer_sizes':[(10,10)],
    'solver':['adam'],
    'activation':['logistic'],
    'learning_rate':['constant','invscaling']
  },
  'xgboost':{
    'booster':['dart'],
    'objective':['binary:logistic'],
    'n_estimators':[500],
    'max_depth':[6],
    'min_child_weight':[6],
    'learning_rate':[0.01],
    'subsample':[0.8]
  },
  'lgb':{
      'boosting_type': ['gbdt'],
      'min_child_weight':[4],
      'num_leaves':[64],
      'tree_learner':['serial'],
      'num_threads':[20],
      'min_data_in_leaf': [10],
      'learning_rate': [0.01],
      "min_child_samples": [10],
      "feature_fraction": [0.9],
      "bagging_fraction": [0.8],
      'bagging_freq':[4],
    },
}


METRICS_CLS = {
  'Accuracy':'accuracy',
  'Recall':'recall',
  'Precision':'precision',
  'F1':'f1',
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
    

    def trainer(self,train_df,target_key,random_state=21,metric=None,k_fold=5,pred_flag=False,test_df=None,test_csv=None,save_path=None,encoder_flag=False,scale_flag=False,sub_col=None,id_name=None):
        params = self.params
        fea_list= [f for f in train_df.columns if f != target_key]
        print(fea_list)
        if encoder_flag:
          le = LabelEncoder()
          # train_df[target_key] = train_df[target_key].apply(lambda x:x)
          train_df[target_key] = le.fit_transform(train_df[target_key])
        # print(len(set(train_df[target_key])))

        Y = np.asarray(train_df[target_key])
        num_classes = len(set(Y))
        X = np.asarray(train_df[fea_list])
        test = np.asarray(test_df[fea_list])

        if scale_flag:
            X_len = X.shape[0]
            data_scaler = StandardScaler()
            cat_data = np.concatenate([X,test],axis=0)
            cat_data= data_scaler.fit_transform(cat_data)

            X = cat_data[:X_len]
            test = cat_data[X_len:]
      
        kfold = KFold(n_splits=k_fold,shuffle=True,random_state=random_state)

        print(X)
        print(test)
        print(X.shape,test.shape)
        test_score_list = []
        predictions = []
        predictions_prob = []
        for fold_num,(train_index,val_index) in enumerate(kfold.split(X)):
            print(f'***********fold {fold_num+1} start!!***********')
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            model = GridSearchCV(estimator=self.clf,
                                param_grid=params,
                                cv=kfold,
                                scoring=metric,
                                refit='F1',
                                verbose=True,
                                return_train_score=True)
            model = model.fit(x_train,y_train)

            best_score = model.best_score_
            best_model = model.best_estimator_
            train_pred = model.predict(x_train)
            train_score = f1(y_train,train_pred)
            test_pred = model.predict(x_val)
            test_score = f1(y_val,test_pred)
            test_score_list.append(test_score)
            print("F1 Evaluation:")
            print("fold {} Best score:{}".format(fold_num + 1,best_score))
            print("fold {} Train score:{}".format(fold_num + 1,train_score))
            print("fold {} Test score:{}".format(fold_num + 1,test_score))
            print("fold {} Best parameter:".format(fold_num + 1))
      
            for key in params.keys():
                print('%s:'%key)
                print(best_model.get_params()[key])
            
            if self.clf_name == 'random forest' or self.clf_name == 'extra trees':
                if self.clf_name == 'random forest':
                    new_grid = RandomForestClassifier(random_state=21,bootstrap=True)
                elif self.clf_name == 'extra trees':
                    new_grid = ExtraTreesClassifier(random_state=21,bootstrap=True)
                new_grid.set_params(**model.best_params_)

                new_grid = new_grid.fit(x_train,y_train) 
                importances = new_grid.feature_importances_
                feat_labels = fea_list
                # print(feat_labels)
                indices = np.argsort(importances)[::-1]

                # print(indices)
                for f in range(x_train.shape[1]):
                    print("%2d) %-*s %f" % (f + 1,30, feat_labels[indices[f]], importances[indices[f]]))
            
            if pred_flag and test is not None:
                pred = model.predict(test)
                predictions.append(pred)
                # prob
                pred_prob = model.predict_proba(test)
                predictions_prob.append(pred_prob)
        
        final_result = []
        vote_array = np.asarray(predictions).astype(int)
        final_result.extend([max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])])
        
        prob_array = np.mean(predictions_prob,axis=0) # N*C
        prob_final_result = np.argmax(prob_array,axis=1).tolist() # N

        if encoder_flag:
            final_result = le.inverse_transform(final_result)
            prob_final_result = le.inverse_transform(prob_final_result)

        
        test_df = pd.read_csv(test_csv,encoding='gbk')
        
        pred_df = {
          sub_col[0]:[case for case in test_df[id_name].values.tolist()]
        }
        pred_df = pd.DataFrame(data=pred_df)

        pred_df[sub_col[0]] = test_df[id_name]
        # vote
        pred_df[sub_col[1]] = final_result
        pred_df.to_csv(f'{save_path}/{self.clf_name}_vote_result.csv',index=False)
        
        # prob
        for i in range(num_classes):
            pred_df[f'prob_{str(i)}'] = prob_array[:,i].tolist()
        pred_df[sub_col[1]] = prob_final_result
        pred_df.to_csv(f'{save_path}/{self.clf_name}_prob_result.csv',index=False)
        return best_model

    def _get_clf(self):
        if self.clf_name == 'lasso':
            classifier = LogisticRegression(penalty='l2',random_state=0)
        elif self.clf_name == 'knn':
            classifier = KNeighborsClassifier()
        elif self.clf_name == 'svm':
            classifier = SVC(kernel='rbf',random_state=21)
        elif self.clf_name == 'decision tree':
            classifier = DecisionTreeClassifier(random_state=21)
        elif self.clf_name == 'random forest':
            classifier = RandomForestClassifier(random_state=21,bootstrap=True)
        elif self.clf_name == 'extra trees':
            classifier = ExtraTreesClassifier(random_state=21,bootstrap=True)
        elif self.clf_name == 'bagging':
            classifier = BaggingClassifier(random_state=21)
        elif self.clf_name == 'mlp':
            classifier = MLPClassifier(max_iter=2000,warm_start=True,random_state=0)
        elif self.clf_name == 'xgboost':
            classifier = XGBClassifier()
        elif self.clf_name == 'lgb':
            classifier = lgb.LGBMClassifier()

        return classifier  

