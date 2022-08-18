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
test_path = "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download?path=%2FCSV%2FTraning%20and%20Testing%20Tets%20(5%25%20of%20the%20entier%20dataset)%2F10-best%20features%2F10-best%20Training-Testing%20split&files=UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# %%


# %%
#ARP packets were excluded because they are used to convert an IP address to a MAC address and irrelevant to the proposed attacks in the dataset
train_data = train_data[train_data.proto!="arp"]
test_data = test_data[test_data.proto!="arp"]

#dropping unused columns
train_data.drop(['pkSeqID', 'proto', 'saddr', 'sport', 'daddr', 'dport',"attack","category"],axis=1,inplace=True)
test_data.drop(['pkSeqID', 'proto', 'saddr', 'sport', 'daddr', 'dport',"attack","category"],axis=1,inplace=True)
full_data = pd.concat([train_data,test_data],axis=0)

# %%
data_exf = train_data.loc[train_data.subcategory=="Data_Exfiltration"].sample(n=2,random_state=42)
test_data=test_data.append([data_exf])
train_data=train_data.drop(data_exf.index)

# %%
# Place the class in the begining of the dataframe
class_train = train_data['subcategory']
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

# %%
train_data.subcategory.value_counts()

# %%
d = full_data.subcategory.value_counts()
fig = px.bar(d, x=d.index, y=d.values,title = 'Class distribution between attack subcategories on full data',labels = {'index':'Attack','y':'Volume'},color=d.values)
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
# Grouping DDOS STUFF TOGETHER

# %%
full_data['subcategory'] = np.where((full_data['subcategory'] == 'TCP') | (full_data['subcategory'] == 'UDP') | (full_data['subcategory'] == 'HTTP') , 'DoS&DDoS', full_data['subcategory'])

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

nondos_df = shuffled_df.loc[shuffled_df['subcategory'] != "DoS&DDoS"]

dos_df = shuffled_df.loc[shuffled_df['subcategory'] == "DoS&DDoS"].sample(n=73122,random_state=42)

normalized_full_df = pd.concat([nondos_df, dos_df])

#TRAIN after undersampling
d = normalized_full_df.subcategory.value_counts()
fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT (undersampled)',labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Splitting into Train and Test datasets
"""

# %%
normalized_full_df.subcategory.value_counts()

# %%
normalized_full_df.head()

# %%
X = normalized_full_df.drop(["subcategory"], axis = 1)
##dropping features
# X.drop(['state_number'],axis=1,inplace=True)

Y = normalized_full_df['subcategory']

X = pd.get_dummies(X, prefix_sep='_')
le = LabelEncoder()
Y2 = le.fit_transform(Y)
X2 = StandardScaler().fit_transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y2, test_size = 0.2, random_state = 10)

# %%
print(X_Train.shape,X_Test.shape)

# %%
print(Y_Train.shape,Y_Test.shape)

# %%
print(Counter(Y_Test))
print(Counter(Y_Train))

# %%
d = pd.Series(data=Counter(Y_Train).values(), index = Counter(Y_Train).keys())
fig = px.bar(d, x=le.inverse_transform(d.index), y=d.values,title = 'Class Label Distribution in Bot-IoT - Training',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.3s')
fig.update_layout(title_x=0.5,width=1000, height=400)
fig.show()

# %%
"""
# Oversampling
"""

# %%
#%%time

sm = SMOTE(random_state=42,sampling_strategy='auto',k_neighbors=2) #k_neighbors=3
X_Smote_Train, Y_Smote_Train = sm.fit_resample(X_Train, Y_Train)
oversampled_train = pd.concat([pd.DataFrame(Y_Smote_Train), pd.DataFrame(X_Smote_Train)], axis=1)
oversampled_train.columns = normalized_full_df.columns

# %%
print(f"Before: {Counter(Y_Train)}, number of records: {sum(Counter(Y_Train).values())}")
print(f"After: {Counter(Y_Smote_Train)}, number of records: {sum(Counter(Y_Smote_Train).values())}")

# %%
d = oversampled_train.subcategory.value_counts()
fig = px.bar(d, x=le.inverse_transform(d.index), y=d.values,title = 'Class Label Distribution in Bot-IoT (oversampled with SMOTE)',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
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
ax.set_title("Imbalanced Correlation Matrix", fontsize=14)

# Imbalanced DataFrame Correlation
corr2 = normalized_full_df.corr()
sns.heatmap(corr2, cmap='YlGnBu', annot_kws={'size':30}, ax=ax1)
ax.set_title("Imbalanced Correlation Matrix", fontsize=14)

plt.show()

# %%
"""
# Random Forests
"""

# %%
#%%time

rfc = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train) #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_forest = rfc.predict(X_Test)
print(classification_report(Y_Test,prediction_forest,target_names = le.inverse_transform(rfc.classes_),digits=3))
print(f"Accuracy: {rfc.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(rfc))} bytes")

cm_forest = confusion_matrix(Y_Test, prediction_forest)
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
# K Nearest Neighbour
"""

# %%
#%%time

neigh = KNeighborsClassifier().fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_neigh = neigh.predict(X_Test)
print(classification_report(Y_Test,prediction_neigh,target_names = le.inverse_transform(neigh.classes_),digits=3))
print(f"Accuracy: {neigh.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(neigh))} bytes")

cm_knn = confusion_matrix(Y_Test, prediction_neigh)
print(cm_knn)

# %%
"""
# Support Vector Machines (SVM)
"""

# %%
#%%time

svmclf = svm.SVC(random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
predictions_svm = svmclf.predict(X_Test)
print(classification_report(Y_Test,predictions_svm,target_names = le.inverse_transform(svmclf.classes_),digits=3))
print(f"Accuracy: {svmclf.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(svmclf))} bytes")

cm_svm = confusion_matrix(Y_Test, predictions_svm)
print(cm_svm)

# %%
"""
# Gradient Boosting
"""

# %%
#%%time

gb = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_gb = gb.predict(X_Test)
print(classification_report(Y_Test,prediction_gb,target_names = le.inverse_transform(gb.classes_),digits=3))
print(f"Accuracy: {gb.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(gb))} bytes")

cm_gbm = confusion_matrix(Y_Test, prediction_gb)
print(cm_gbm)

# %%
"""
# Adaboost
"""

# %%
#%%time

adab = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_adab = adab.predict(X_Test)
print(classification_report(Y_Test,prediction_adab,target_names = le.inverse_transform(adab.classes_),digits=3))
print(f"Accuracy: {adab.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(adab))} bytes")

cm_ada = confusion_matrix(Y_Test, prediction_adab)
print(cm_ada)

# %%
"""
# Artifical Neural Networks
"""

# %%
#%%time

ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,2), random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
prediction_ann = ANN.predict(X_Test)
print(classification_report(Y_Test,prediction_ann,target_names = le.inverse_transform(ANN.classes_),digits=3))
print(f"Accuracy: {ANN.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(ANN))} bytes")

cm_ann = confusion_matrix(Y_Test, prediction_ann)
print(cm_ann)

# %%
#%%time

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann_clf = Sequential()

# Adding the input layer and the first hidden layer
ann_clf.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
# classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
ann_clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
ann_clf.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#summary
ann_clf.summary()

# Fitting the ANN to the Training set
ann_clf.fit(X_Train,Y_Train , epochs = 10) #SMOTE: X_Smote_Train,Y_Smote_Train

# Predicting the Test set results
predict_x=ann_clf.predict(X_Test) 
classes_x=np.argmax(predict_x,axis=1)

print(classification_report(Y_Test, classes_x,digits=3))
print(f"Model size: {sys.getsizeof(pickle.dumps(ann_clf))} bytes")