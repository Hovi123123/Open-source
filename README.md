# DeepMIF
Source Code of 《DeepMIF：a deep learning based multimodal information fusion method for small molecule activity classes prediction》

---

### 1. Overall architecure of DeepMIF:

![Image text](https://raw.githubusercontent.com/Hovi123123/DeepMIF/master/DeepMIF_model.jpg?token=GHSAT0AAAAAABSZT3ISBFJEJYKGZIPNLRXKYR4RHOA)



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
> │  </br>
> ├─best_model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Store the trained model</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_model.pt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# The trained model</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;events.out.tfevents.1640322372.Hovi-laptop.40112.0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Support for using TensorBoard to visualize the training process</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Parameters of the trained model</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result.txt			 </br>
> │      </br>
> ├─predict&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stored prediction example data</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pre_labels.csv&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  Predicted results</br>
> │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TCM_inhouse_tumor.xlsx&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Data used for prediction</br>
> │      </br>
> └─src&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Stored the main codes</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dataset.py</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_cleaned.xlsx</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;main.py</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model.py</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;new_pre.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Prediction</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  </br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─logs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Training log, same content as 'best_model'</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─traning_logs_31592</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_model.pt</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;events.out.tfevents.1640322372.Hovi-laptop.40112.0</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters.txt</br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result.txt</br>

 

### 4. Contact:

Hovi Tang: hovi@shu.edu.cn
