import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# os.system("taskset -p 0xff %d" % os.getpid())

def getStandardScaler(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def getPCA(X,my_components):
    # my_components: the dimension to be reduced to
    pca = PCA(n_components=my_components)
    pca.fit(X)
    X_reduction = pca.transform(X)
    X_reduction = getStandardScaler(X_reduction)
    return X_reduction

def get_data(data_path):
    data_set = pd.read_excel(data_path,sheet_name=0)
    train_data = data_set.iloc[:,2:]
    train_target = data_set.iloc[:,0]
    
    train_target = np.array(train_target)
    activity_list = np.zeros(train_target.shape[0],dtype=np.int)
    j = 0 
    for i in train_target:
        t = float(i)
        if t<=0.5:
            activity_list[j] = 1
        else:
            activity_list[j] = 0
        j += 1

    train_data = np.array(train_data)

    ecfp0 = np.array(train_data[:,361:2409],dtype=np.int)
    new_train_data = getPCA(ecfp0,128)

    test_x = new_train_data[:531,:]
    test_y = activity_list[:531]
    train_x = new_train_data[531:,:]
    train_y = activity_list[531:]
    print("data got!")
    
    return train_x,train_y,test_x,test_y

def bi_model_evaluation(y_true, y_pred, y_score):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_score)
    auc1 = auc(fpr, tpr)

    return acc,f1,auc1

def main(model,train_x,train_y,test_x,test_y,sheet_name,excel_writer):
    seed_list = [32416,31764,31861,32342,32486,32249,32313,31691,
                 32289,32538,32487,31673,32140,31632,31732,31607,
                 31786,31687,32397,31948,31924,32543,32479,31956,
                 31690,31677,32200,32168,32230,31692]
    acc_list = []
    f1_list = []
    auc_list = []
    best_para  = []
    for seed in seed_list:
        print(seed)
        scores = {
            'acc': 'accuracy',
            'f1': 'f1',
            'auc': 'roc_auc'
        }
        if model == 'svc':
            parameters = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': [1e-3, 1e-4]
            }
            clf = SVC(random_state=seed)

        if model == 'rf':
            parameters = {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy']
            }
            clf = RandomForestClassifier(random_state=seed)

        if model == 'knn':
            parameters = [
                {
                    'weights': ['uniform'],
                    'n_neighbors': [i for i in range(4, 8)]
                },
                {
                    'weights': ['distance'],
                    'n_neighbors': [i for i in range(4, 8)],
                    'p': [i for i in range(2, 5)]
                }]
            clf = KNeighborsClassifier()

        if model == 'dt':
            parameters = {
                'criterion': ['entropy', 'gini'],
                'max_depth': [50, 60, 100]
            }
            clf = DecisionTreeClassifier(random_state=seed)

        if model == 'gdbc':
            parameters = {
                'criterion': ["friedman_mse", "squared_error"],
                'n_estimators': [50, 100, 150]
            }
            clf = GradientBoostingClassifier(random_state=seed)

        if model == 'ab':
            parameters = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 1]
            }
            clf = AdaBoostClassifier(random_state=seed)

        if model == 'gnb':
            parameters = {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
            clf = GaussianNB()

        if model == 'sgd':
            parameters = {
                'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                'penalty': ['l2', 'l1', 'elasticnet']
            }
            clf = SGDClassifier( random_state = seed,max_iter=5000)

        if model == 'lr':
            parameters = {
                'max_iter': [5000, 6000, 7000],
                'penalty': ['l2']
            }
            clf = LogisticRegression(random_state=seed)

        gs = RandomizedSearchCV(clf,parameters, cv=5, scoring=scores, refit='f1', verbose=1, n_jobs=14, n_iter=10)
        gs.fit(train_x,train_y)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(test_x)
        y_score = best_model.predict_proba(test_x)
        acc, f1, auc1 = bi_model_evaluation(test_y, y_pred, y_score[:,1])

        best_para.append(best_model.get_params())
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc1)
        print(model)

    acc_np = np.array(acc_list).reshape((30,1))
    f1_np = np.array(f1_list).reshape((30,1))
    auc_np = np.array(auc_list).reshape((30,1))
    seed_np = np.array(seed_list).reshape((30,1))
    best_para_np = np.array(best_para).reshape((30,1))
    com_vec = np.concatenate((seed_np,acc_np,f1_np,auc_np,best_para_np),1)
    com_pd = pd.DataFrame(com_vec)
    com_pd.columns = ['Seed', 'ACC', 'F1', 'AUC', 'Best_model']
    com_pd.to_excel(excel_writer, index=False, sheet_name=str(model))


if __name__ == "__main__":

    model_list = ['svc','rf','knn','dt','gdbc','ab',
                  'gnb','sgd','lr']
    data_path = r'../data_cleaned.xlsx'
    train_x,train_y,test_x,test_y = get_data(data_path)
    excel_writer = pd.ExcelWriter('./2.xlsx')
    sheet_name = 0
    for i in model_list:
        main(i,train_x,train_y,test_x,test_y,sheet_name,excel_writer)
        sheet_name = sheet_name + 1
    excel_writer.close()
    print("finished")
