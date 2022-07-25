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
from scipy import stats
# _AVAIL_CLF = ['lr','xgboost','lgb','mlp','random_forest','extra_trees','bagging']
# _AVAIL_CLF = ['random_forest','extra_trees','bagging']
_AVAIL_CLF = ['mlp']
# _AVAIL_CLF = ['lr']
# _AVAIL_CLF = ['lr']
# _AVAIL_CLF = ['random_forest']

def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def neg_rmse(y_true,y_pred):
    return -1.0*mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5

def log_rmse(y_true,y_pred):
    return -1.0*math.log(mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5)

def neg_mse(y_true,y_pred):
    return -1.0*mean_squared_error(y_true=y_true,y_pred=y_pred)

def make_feat(train_df, test_df, drop_feats):

    data = pd.concat([train_df, test_df],axis=0, ignore_index=True)

    #性别处理
    data['性别'] = data['性别'].map({'男': 1, '女': 0, '??':0})

    #对缺少一部分的数据进行填充
    # data.fillna(data.median(axis=0), inplace=True)

    #对缺失值和无意义值进行处理
    data = data.drop(drop_feats,axis=1)

    train_df = data[:train_df.shape[0]]
    test_df = data[train_df.shape[0]:]

    train_df.fillna(train_df.median(axis=0), inplace=True)
    test_df.fillna(test_df.median(axis=0), inplace=True)

    '''
    #删除离群值
    train_df = train_df.drop(train_df[train_df['*r-谷氨酰基转换酶'] > 600 ].index)
    train_df = train_df.drop(train_df[train_df['白细胞计数'] > 20.06].index)
    train_df = train_df.drop(train_df[train_df['*丙氨酸氨基转换酶'] == 498.89].index)
    train_df = train_df.drop(train_df[train_df['单核细胞%'] > 20 ].index)
    train_df = train_df.drop(train_df[train_df['*碱性磷酸酶'] > 340].index)    #有待调整
    train_df = train_df.drop(train_df[train_df['*球蛋白'] > 60].index)
    train_df = train_df.drop(train_df[train_df['嗜酸细胞%'] > 20].index)
    train_df = train_df.drop(train_df[train_df['*天门冬氨酸氨基转换酶'] > 300].index)
    train_df = train_df.drop(train_df[train_df['血小板计数'] > 700].index)
    train_df = train_df.drop(train_df[train_df['*总蛋白'] > 100].index)
    

    
    #对训练数据进行平滑处理
    # train_df['甘油三酯'], a = stats.boxcox(train_df['甘油三酯'])
    train_df['*r-谷氨酰基转换酶'], b = stats.boxcox(train_df['*r-谷氨酰基转换酶'])
    # train_df['白球比例'], c = stats.boxcox(train_df['白球比例'])
    train_df['*天门冬氨酸氨基转换酶'], d = stats.boxcox(train_df['*天门冬氨酸氨基转换酶'])

    test_df['甘油三酯'], a1 = stats.boxcox(test_df['甘油三酯'])
    test_df['*r-谷氨酰基转换酶'], a1 = stats.boxcox(test_df['*r-谷氨酰基转换酶'])
    # test_df['白球比例'], a1 = stats.boxcox(test_df['白球比例'])
    test_df['*天门冬氨酸氨基转换酶'], a1 = stats.boxcox(test_df['*天门冬氨酸氨基转换酶'])
    #train_df['甘油三酯'] += 2
    #test_df['甘油三酯'] += 2
    '''
    
    return train_df, test_df




def scaler_normalize(train_df,test_df,scale_list=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data = data.fillna(0)

    if scale_list is not None:
        # scaler = MinMaxScaler(feature_range=(0, 1))

        scaler = StandardScaler()
        
        for col in scale_list:
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df





METRICS_REG= {
  'mse':make_scorer(mean_squared_error,greater_is_better=False),
  'neg_mse':make_scorer(neg_mse),
  'rmse':make_scorer(rmse,greater_is_better=False),
  'neg_rmse':make_scorer(neg_rmse),
  'log_rmse':make_scorer(log_rmse)
}

SETUP_TRAINER = {
  'target_key':'血糖',
  'random_state':21,
  'metric':METRICS_REG,
  'k_fold':5,
  'scale_flag':False
}


if __name__ == "__main__":

    train_path = './dataset/diabetes/train_add.csv'
    train_df = pd.read_csv(train_path, encoding='gbk')

    test_path = './dataset/diabetes/test.csv'
    test_df = pd.read_csv(test_path, encoding='gbk')

    drop_feats = ['id','体检日期','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体']

    train_df, test_df = make_feat(train_df, test_df, drop_feats)
    feats = [f for f in train_df.columns if f not in ['血糖']]
    print(feats)
    
    # scale_list = [f for f in feats if f not in ['性别']]
    scale_list = ['*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*r-谷氨酰基转换酶','白球比例','甘油三酯']
    # train_df,test_df = scaler_normalize(train_df,test_df,scale_list,'血糖')
    

    # fea_list = ['*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*r-谷氨酰基转换酶','白球比例','甘油三酯']
    fea_list = feats
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['血糖']]

    save_path = './result/diabetes/'
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