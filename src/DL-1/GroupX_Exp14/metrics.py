from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def get_ACC(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_Precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def get_Recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def get_F1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def get_ROC(y_true, y_pred):
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    auc1 = auc(fpr, tpr)
    return auc1

def get_auc_data(y_true, y_pred):
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    return fpr, tpr
