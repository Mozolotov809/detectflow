# %%
"""
# Importing libraries
"""

# %%
# this module provides functions for interacting with the operating system 
import os
import pickle
import sys

# it's used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# Pandas used for data manipulation and analysis
import pandas as pd

# data visualization library for 2D and 3D plots, built on numpy
from matplotlib import pyplot as plt
#%matplotlib inline

# Seaborn is based on matplotlib; used for plotting statistical graphics
import seaborn as sns

# plotting
import plotly.express as px

# sklearn helpful ML libraries including clasifiers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
import time

# to suppress warnings
import warnings
warnings.filterwarnings("ignore") 

# %%
"""
# Importing Data and removing useless columns, splitting data exfiltration
"""

# %%
train_path = "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download?path=%2FCSV%2FTraning%20and%20Testing%20Tets%20(5%25%20of%20the%20entier%20dataset)%2F10-best%20features%2F10-best%20Training-Testing%20split&files=UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
#train_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/pcap_files/mqtt_train_dupl.csv"
test_path = "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download?path=%2FCSV%2FTraning%20and%20Testing%20Tets%20(5%25%20of%20the%20entier%20dataset)%2F10-best%20features%2F10-best%20Training-Testing%20split&files=UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
#test_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/pcap_files/mqtt_test_dupl.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# %%
#ARP packets were excluded because they are used to convert an IP address to a MAC address and irrelevant to the proposed attacks in the dataset
train_data = train_data[train_data.proto!="arp"]
test_data = test_data[test_data.proto!="arp"]

#dropping unused columns
train_data.drop(['pkSeqID', 'proto', 'saddr', 'sport', 'daddr', 'dport',"attack","category"],axis=1,inplace=True)
test_data.drop(['pkSeqID', 'proto', 'saddr', 'sport', 'daddr', 'dport',"attack","category"],axis=1,inplace=True)
full_data = pd.concat([train_data,test_data],axis=0)
print(test_data)
print(full_data)

# %%
data_exf = train_data.loc[train_data.subcategory=="Data_Exfiltration"].sample(n=2,random_state=42)
#loc函数：通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）
test_data=test_data.append([data_exf]) # subcategory 加了 data_exfiltration
train_data=train_data.drop(data_exf.index)

# %%
# Place the class in the begining of the dataframe
class_train = train_data['subcategory']
#0          UDP
#1          TCP
#2          TCP
#Name: subcategory, Length: 2934649, dtype: object
train_data.drop(['subcategory'], axis=1, inplace=True)
train_data.insert(0, 'subcategory', class_train)


class_test = test_data['subcategory']
test_data.drop(['subcategory'], axis=1, inplace=True)
test_data.insert(0, 'subcategory', class_test)

class_full = full_data['subcategory']
full_data.drop(['subcategory'], axis=1, inplace=True)
full_data.insert(0, 'subcategory', class_full)

# %%
"""
# Checking Balance of the Data
"""

# %%
test_data.subcategory.value_counts()
print(test_data.subcategory.value_counts())
#UDP                  1584622
#TCP                  1274811
#Service_Scan           58592
#OS_Fingerprint         14267
#HTTP                    1965
#Normal                   332
#Keylogging                57
#Data_Exfiltration          3
#Name: subcategory, dtype: int64

train_data.subcategory.value_counts()
d = full_data.subcategory.value_counts()

fig = px.bar(d, x=d.index, y=d.values,title = 'Class distribution between attack subcategories on full data',labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()


# %%
"""
# Checking Missing Data and datatypes
"""

# %%
percent_missing = train_data.isnull().sum() * 100 / len(train_data)
missing_values = pd.DataFrame({'% missing': percent_missing})
missing_values.sort_values(by ='% missing' , ascending=False)

# %%
pd.DataFrame({'Data Type': full_data.dtypes})

# %%
"""
# Undersampling
"""

# %%
d = full_data.subcategory.value_counts()
fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT',labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
shuffled_df = full_data.sample(frac=1,random_state=4)
#从所选的数据的指定 axis 上返回随机抽样结果，类似于random.sample()函数。意味着frac=1返回所有行
#shuffled_df 是full_data 抽样后的结果

nontcpudp_df = shuffled_df.loc[(shuffled_df['subcategory'] != "UDP") & (shuffled_df['subcategory'] != "TCP")]
#loc函数：通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）

udp_df = shuffled_df.loc[shuffled_df['subcategory'] == "UDP"].sample(n=73122,random_state=42)
tcp_df = shuffled_df.loc[shuffled_df['subcategory'] == "TCP"].sample(n=73122,random_state=42)

normalized_full_df = pd.concat([nontcpudp_df, udp_df, tcp_df])
#pandas.concat()通常用来连接DataFrame对象，默认纵向，axis=1 横向连接

#TRAIN after undersampling
d = normalized_full_df.subcategory.value_counts()
fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT (undersampled)',labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Binary Classification
"""

# %%
normalized_full_df['subcategory'] = (normalized_full_df['subcategory'] != 'Normal').astype(int)

#numpy astype（int）转化为整型数据

"""
# Splitting into Train and Test datasets
"""

print(normalized_full_df.subcategory.value_counts())
#1    239790
#0       430
#Name: subcategory, dtype: int64
print('normalized_full_df.head()')
print(normalized_full_df.head())
#pandas.head() 函数用于访问数据帧或系列的前 n 行。default n=5

# %%
X = normalized_full_df.drop(["subcategory"], axis = 1)
X = pd.get_dummies(X, prefix_sep='_')
# one hot encode -> color['red','yellow'] -> color_red[1,0] color_yellow[0,1]
# le = LabelEncoder()
##dropping features
# X.drop(['state_number'],axis=1,inplace=True)
Y = normalized_full_df['subcategory']
#x具有输入 ( )的二维数组
#y具有输出 ( )的一维数组

# Y2 = le.fit_transform(Y)
X2 = StandardScaler().fit_transform(X)
print('X2')
print(X2)
#fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.2, random_state = 9)
#sklearn.model_selection.train_test_split随机划分训练集和测试集

# %%
print('X_Train.shape,X_Test.shape:')
print(X_Train.shape,X_Test.shape)

# %%
print('Y_Train.shape,Y_Test.shape:')
print(Y_Train.shape,Y_Test.shape)

# %%
print(Counter(Y_Test))
print(Counter(Y_Train))
#1:attack 0:Normal

# %%
d = pd.Series(data=Counter(Y_Train).values(), index = Counter(Y_Train).keys())
#Series是一种类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成。
fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT - Training',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.3s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Oversampling
"""

# %%
#%%time

sm = SMOTE(random_state=42,sampling_strategy='auto') #k_neighbors=3
#SMOTE模型默认生成一比一的数据
X_Smote_Train, Y_Smote_Train = sm.fit_resample(X_Train, Y_Train)
oversampled_train = pd.concat([pd.DataFrame(Y_Smote_Train), pd.DataFrame(X_Smote_Train)], axis=1)
oversampled_train.columns = normalized_full_df.columns

# %%
print(f"Before: {Counter(Y_Train)}, number of records: {sum(Counter(Y_Train).values())}")
print(f"After: {Counter(Y_Smote_Train)}, number of records: {sum(Counter(Y_Smote_Train).values())}")

# %%
d = oversampled_train.subcategory.value_counts()
fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT (oversampled with SMOTE)',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Feature Correlation
"""

# %%
# Sample figsize in inches
fig, ax = plt.subplots(figsize=(12,6))        
fig, ax1 = plt.subplots(figsize=(12,6))    

# Imbalanced DataFrame Correlation
corr = full_data.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Imbalanced Correlation Matrix full_data", fontsize=14)

# Imbalanced DataFrame Correlation
corr2 = normalized_full_df.corr()
sns.heatmap(corr2, cmap='YlGnBu', annot_kws={'size':30}, ax=ax1)
ax.set_title("Imbalanced Correlation Matrix normalized_full_df", fontsize=14)

plt.show()
plt.savefig( 'myfig.png' )

# %%
"""
# Random Forests
"""

# %%
#%%time
print("----------------Random Forests----------------")
rfc = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train) #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_forest = rfc.predict(X_Test)
print(classification_report(Y_Test,prediction_forest))
print(f"Accuracy: {rfc.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(rfc))} bytes")

cm_forest = confusion_matrix(Y_Test, prediction_forest, labels=[1,0])
print(cm_forest)

# %%
"""
## Feature Importance using RF
"""

# %%
feat_importances = pd.Series(rfc.feature_importances_, index= X.columns).sort_values(ascending=False)
fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance using RF',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Gradient Boosting
"""

# %%
#%%time
print("----------------Gradient Boosting----------------")
gb = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_gb = gb.predict(X_Test)
print(classification_report(Y_Test,prediction_gb))
print(f"Accuracy: {gb.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(gb))} bytes")

cm_gbm = confusion_matrix(Y_Test, prediction_gb, labels=[1,0])
print(cm_gbm)

# %%
"""
# K Nearest Neighbour
"""

# %%
#%%time
print("----------------K Nearest Neighbour----------------")
neigh = KNeighborsClassifier().fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train n_neighbors=3
prediction_neigh = neigh.predict(X_Test)
print(classification_report(Y_Test,prediction_neigh))
print(f"Accuracy: {neigh.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(neigh))} bytes")

cm_knn = confusion_matrix(Y_Test, prediction_neigh, labels=[1,0])
print(cm_knn)

# %%
"""
# Support Vector Machines (SVM)
"""

# %%
#%%time
print("----------------Support Vector Machines (SVM)----------------")
svmclf = svm.SVC(random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
predictions_svm = svmclf.predict(X_Test)
print(classification_report(Y_Test,predictions_svm))
print(f"Accuracy: {svmclf.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(svmclf))} bytes")

cm_svm = confusion_matrix(Y_Test, predictions_svm, labels=[1,0])
print(cm_svm)

# %%
"""
# Adaboost
"""

# %%
#%%time
print("----------------Adaboost----------------")
adab = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_adab = adab.predict(X_Test)
print(classification_report(Y_Test,prediction_adab))
print(f"Accuracy: {adab.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(adab))} bytes")

cm_ada = confusion_matrix(Y_Test, prediction_adab, labels=[1,0])
print(cm_ada)

# %%
"""
# Artifical Neural Networks
"""

# %%
#%%time
print("----------------Artifical Neural Networks----------------")
ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_ann = ANN.predict(X_Test)
print(classification_report(Y_Test,prediction_ann))
print(f"Accuracy: {ANN.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(ANN))} bytes")

cm_ann = confusion_matrix(Y_Test, prediction_ann, labels=[1,0])
print(cm_ann)

# %%
#%%time

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann_clf = Sequential()
#模型Sequential适用于简单的层堆栈， 其中每一层恰好有一个输入张量和一个输出张量。
# Adding the input layer and the first hidden layer
ann_clf.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
# classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
ann_clf.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
ann_clf.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#summary
ann_clf.summary()

# Fitting the ANN to the Training set
ann_clf.fit(X_Train,Y_Train , epochs = 10) #SMOTE: X_Smote_Train,Y_Smote_Train

# Predicting the Test set results
predict_x=ann_clf.predict(X_Test) 
classes_x=np.argmax(predict_x,axis=1)

print('classification_report(Y_Test, classes_x):')
print(classification_report(Y_Test, classes_x))
print(f"Model size: {sys.getsizeof(pickle.dumps(ann_clf))} bytes")