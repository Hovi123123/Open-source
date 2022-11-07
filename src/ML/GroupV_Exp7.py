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
from sklearn.model_selection import RandomizedSearchCV

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
        if t<=10:
            activity_list[j] = 1
        else:
            activity_list[j] = 0
        j += 1

    train_data = np.array(train_data)
    
    #Molecular descriptors
    dp0 = getStandardScaler(train_data[:,0:194])
    dp1 = getPCA(dp0,128)
    #MACCS
    macc0 = getStandardScaler(np.array(train_data[:,194:361],dtype=np.int))
    macc1 = getPCA(macc0,128)
    #ecfp4
    ecfp0 = getStandardScaler(np.array(train_data[:,361:2409],dtype=np.int))
    ecfp1 = getPCA(ecfp0,128)

    new_train_data = np.concatenate((dp1,macc1,ecfp1),axis=1)

    test_x = new_train_data[:531,:]
    test_y = activity_list[:531]
    train_x = new_train_data[531:,:]
    train_y = activity_list[531:]

    return train_x,train_y,test_x,test_y

def bi_model_evaluation(y_true, y_pred, y_score):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_score)
    auc1 = auc(fpr, tpr)

    return acc,f1,auc1

def main(model,train_x,train_y,test_x,test_y):
    seed_list = [32416]
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
            clf = SVC(random_state=seed,probability=True)

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
                'loss': ['log', 'modified_huber'],
                'penalty': ['l2', 'l1', 'elasticnet']
            }
            clf = SGDClassifier( random_state = seed,max_iter=5000)

        if model == 'lr':
            parameters = {
                'max_iter': [5000, 6000, 7000],
                'penalty': ['l2']
            }
            clf = LogisticRegression(random_state=seed)

        gs = RandomizedSearchCV(clf,parameters, cv=5, scoring=scores, refit='f1', verbose=1, n_jobs=5,n_iter=10)
        gs.fit(train_x,train_y)
        best_model = gs.best_estimator_
        y_score = best_model.predict_proba(test_x)
        y_pred = best_model.predict(test_x)
        acc, f1,auc1 = bi_model_evaluation(test_y, y_pred, y_score[:,1])
        print(model)
        print("ACC:",acc)
        print("F1:",f1)
        print("AUC:",auc1)
        # print("Params:",best_model.get_params())


if __name__ == "__main__":

    model_list = ['svc','rf','knn','dt','gdbc','ab',
                  'gnb','sgd','lr']
    data_path = r'./data_cleaned_decoy.xlsx'
    train_x,train_y,test_x,test_y = get_data(data_path)
    for i in model_list:
        main(i,train_x,train_y,test_x,test_y)
    print("finished")
