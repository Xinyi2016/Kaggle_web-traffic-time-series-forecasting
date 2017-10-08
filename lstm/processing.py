import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

train_raw = pd.read_csv("../data/train_2.csv")
# fillna(0) # 44.9
train_df = train_raw.fillna(method='bfill')
train_df = train_df.fillna(method='ffill')

data = train_df.drop('Page',axis = 1)
data.shape

row = data.iloc[90000,:].values

X = row[0:802].astype(np.float32)
y = row[1:803].astype(np.float32)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val= train_test_split(X_tr, y_tr, test_size=0.2, random_state=1)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = np.reshape(X_train,(-1,1))
y_train = np.reshape(y_train,(-1,1))
X_val = np.reshape(X_val, (-1,1))
y_val = np.reshape(y_val, (-1,1))
X_train = sc.fit_transform(X_train)
y_train = sc.fit_transform(y_train)
X_val = sc.fit_transform(X_val)
y_val = sc.fit_transform(y_val)

X_trained = np.reshape(X_train, (X_train.shape[0],1,1))
X_valed = np.reshape(X_val, (X_val.shape[0],1,1))

y_trained= np.reshape(y_train, (y_train.shape[0],1,1))
y_valed = np.reshape(y_val, (y_val.shape[0],1,1))

X_test = np.reshape(X_test, (-1,1))
y_test = np.reshape(y_test, (-1,1))
X_test = sc.fit_transform(X_test)
y_test = sc.fit_transform(y_test)
X_tested = np.reshape(X_test, (X_test.shape[0],1,1))

numRows=5
result = []
i=1
for i in range(numRows):
	print(i)
	result.append([j for ])


