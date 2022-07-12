from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def getStandardScaler(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def getPCA(X,my_components):
    pca = PCA(n_components=my_components)
    pca.fit(X)
    X_reduction = pca.transform(X)
    X_reduction = getStandardScaler(X_reduction)
    return X_reduction

class MyDataset(Dataset):
    def __init__(self,phase, data_path,  dp_len, maccs_len, ecfp4_len):
        data_path = Path(data_path)
        self.dp_len = dp_len
        self.maccs_len = maccs_len
        self.ecfp4_len = ecfp4_len

        data_set = pd.read_excel(data_path,sheet_name=0)
        train_data = data_set.iloc[:,1:]
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

        dp0 = getStandardScaler(train_data[:,1:195])
        dp1 = getPCA(dp0,128)

        macc0 = np.array(train_data[:,195:362],dtype=np.int)
        macc1 = getPCA(macc0,128)

        ecfp0 = np.array(train_data[:,362:2410],dtype=np.int)
        ecfp1 = getPCA(ecfp0,128)


        new_train_data = np.concatenate((np.reshape(train_data[:,0],(train_data[:,0].shape[0],1)),dp1,macc1,ecfp1),axis=1)


        X_test = new_train_data[:531,:]
        y_test = activity_list[:531]

        X_train,X_dev, y_train, y_dev =train_test_split(new_train_data[531:,:],activity_list[531:],test_size=0.06,
                                                          random_state=8888,stratify=activity_list[531:],shuffle=True)

        if phase == 'training':
            self.activity_list = y_train
            self.smi_list = X_train[:,0]
            self.dp_value = X_train[:,1:129] 
            self.macc_values = X_train[:,129:257] 
            self.ecfp4_values = X_train[:,257:385]
            self.length = X_train.shape[0]


        elif phase == 'validation':
            self.activity_list = y_dev
            self.smi_list = X_dev[:,0]
            self.dp_value = X_dev[:,1:129]  
            self.macc_values = X_dev[:,129:257] 
            self.ecfp4_values = X_dev[:,257:385]
            self.length = X_dev.shape[0]

        elif phase == 'test':
            self.activity_list = y_test
            self.smi_list = X_test[:,0]
            self.dp_value = X_test[:,1:129]
            self.macc_values = X_test[:,129:257]  
            self.ecfp4_values = X_test[:,257:385]
            self.length = y_test.shape[0]



    def __getitem__(self, idx):

        activity_value = self.activity_list[idx]
        single_dp = self.dp_value[idx]
        single_maccs = self.macc_values[idx]
        single_ecfp4 = self.ecfp4_values[idx]


        return (np.array(activity_value,dtype=np.int),
                np.array(single_dp,dtype=np.float64),
                np.array(single_maccs,dtype=np.float64),
                np.array(single_ecfp4,dtype=np.float64)
                )

    def __len__(self):
        length = self.length
        return length
