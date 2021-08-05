import pickle

from numpy.core.fromnumeric import mean
from numpy.lib.financial import fv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np

def select_feature_linesvc(fvecs, labels, save_path):
    '''
    fvecs: (n, d) d-dimentional feature vectors of n sequences
    labels: (n, 1) scalar class label of n sequence
    save_path: str
    
    select_model : sklearn model to select features from vectors
    '''
    fvecs, labels = np.array(fvecs), np.array(labels)
    scaler = StandardScaler().fit(fvecs)
    fvecs = scaler.transform(fvecs)
    clf=LinearSVC(penalty='l1', C=0.1, dual=False, random_state=0)
    clf.fit(fvecs, labels)
    importance = np.linalg.norm(clf.coef_, axis=0, ord=1)
    mean = np.mean(importance)
    select_model = SelectFromModel(clf, prefit=True, threshold=1.25*mean)
    # selected_features = select_model.transform(fvecs)
    with open(save_path, 'wb') as f:
        pickle.dump(select_model, f)
    return select_model