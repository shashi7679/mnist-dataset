from joblib import dump,load
from numpy.core.getlimits import _fr1
import sklearn
import numpy as np
from sklearn import metrics
import os

best_svm_path = "../models/tt_0.2_val_0.1_round_4_svm_hyper_0.0001/model.joblib"

best_decisiontree_path = "../models/tt_0.2_val_0.1_round_3_tree_hyper_15/model.joblib"

def svm_load_predict(test_X,test_Y):
    svm = load(best_svm_path)
    pred = svm.predict(test_X)
    #acc = metrics.accuracy_score(y_true = test_Y,y_pred = pred)
    return pred

def tree_load_predict(test_X,test_Y):
    svm = load(best_decisiontree_path)
    pred = svm.predict(test_X)
    #acc = metrics.accuracy_score(y_true = test_Y,y_pred = pred)
    return pred