Source Code of 《Multi-modal data fusion for supervised learning-based identification of USP7 inhibitors: a systematic comparison》

---

### 1. Overall architecure

![New_model.jpg](https://s2.loli.net/2022/10/25/Wqotbc93dfE8KMB.jpg)


### 2. Requirements:

- Python >= 3.6

- numpy

- pandas

- scikit_learn

- pytorch

- tqdm

  

### 3. Document Structure:

> │  LICENSE </br>
> │  README.md </br>
> │  requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Recommended Environment </br>
> ├─data&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Include the USP7 dataset, decoy dataset and SMILES enumeration dataset</br>
> ├─results&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stored the results of all experiments, and Tables in Additional file 1</br>
> └─src&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stored the main codes (cutoff values = 0.5,1,10, SMOTE, decoy selection, SMILES enumeration)</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─DL-1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   #  The source code of experiment 14 in Group VI-X. </br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  Experiment 8-13 only needs to change the inputs.</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─DL-2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   #  The source code of Experiment 15 in Group XI-XV.</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ML&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     #  The source code of experiment 7 in Group I-V.</br>
>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  Experiment 1-6 only needs to change the inputs.</br>
* The recommended hyperparameters for each ML or DL models are available in the paper`s Additional file 1.
* We provide the best model and prediction script of Experiment 15 in Group XI, and the prediction scripts of other experiments can be modified if is needed.
 

### 4. Contact:

Hovi Tang: hovi@shu.edu.cn
