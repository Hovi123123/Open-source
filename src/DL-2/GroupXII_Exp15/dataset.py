from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]
    return X

class MyDataset(Dataset):
    def __init__(self,phase, data_path, max_smi_len):
        data_path = Path(data_path)
        self.max_smi_len = max_smi_len


        data_set = pd.read_excel(data_path,sheet_name=0)
        train_data = data_set.iloc[:,1:]
        train_target = data_set.iloc[:,0]
        
        train_target = np.array(train_target)
        activity_list = np.zeros(train_target.shape[0],dtype=np.int)
        j = 0 
        for i in train_target:
            t = float(i)
            if t<=1:
                activity_list[j] = 1
            else:
                activity_list[j] = 0
            j += 1

        new_train_data = np.array(train_data)

        X_test = new_train_data[:531,:]
        y_test = activity_list[:531]

        X_train,X_dev, y_train, y_dev =train_test_split(new_train_data[531:,:],activity_list[531:],test_size=0.06,
                                                          random_state=8888,stratify=activity_list[531:],shuffle=True)

        if phase == 'training':
            self.activity_list = y_train
            self.smi_list = X_train[:,0]
            self.length = X_train.shape[0]

        elif phase == 'validation':
            self.activity_list = y_dev
            self.smi_list = X_dev[:,0]
            self.length = X_dev.shape[0]

        elif phase == 'test':
            self.activity_list = y_test
            self.smi_list = X_test[:,0]
            self.length = y_test.shape[0]

    def __getitem__(self, idx):
        activity_value = self.activity_list[idx]

        smi = self.smi_list[idx]
        smi_padded_onehot = label_smiles(smi,self.max_smi_len)
        smi_len = self.max_smi_len

        return (np.array(activity_value,dtype=np.int),
                smi_padded_onehot,
                np.array(smi_len,dtype=np.int)
                )

    def __len__(self):
        length = self.length
        return length
    
if __name__ == '__main__':

    test = 'O=C(NCCN1CCOCC1)c1ccc'
    a = label_smiles(test, 40)

    print(a)
