# %%
"""
# Importing libraries
"""

# %%
# this module provides functions for interacting with the operating system
import pickle
import sys

# Pandas used for data manipulation and analysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
import plotly.express as px

# sklearn helpful ML libraries including clasifiers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#import read_large_file

import time

# to suppress warnings
import warnings
import datetime
warnings.filterwarnings("ignore")

# %%
"""
# Importing Data and removing useless columns, splitting data exfiltration
"""
'''
#iotid20
full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoTID20*/iotid20_reduli.csv"
print("Dataset:IoTID20")
full_data = pd.read_csv(full_path)
full_data.drop([ 'Timestamp', 'Dst_IP', 'Src_IP','Src_Port',  'Dst_Port',"Label","Cat"],axis=1,inplace=True)
print('drop')
'''
#mqtt
'''
full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/mqtt_reduli.csv"
print("Dataset:MQTT")
full_data = pd.read_csv(full_path)
full_data.drop([ 'Timestamp', 'Dst_IP', 'Src_IP','Src_Port',  'Dst_Port',"Label","Cat"],axis=1,inplace=True)
print('drop')

'''
#botiot
'''
print("Dataset:botiot")
full_path = "/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/botiot.csv"
full_data = pd.read_csv(full_path)
full_data.drop(['Unnamed: 0'],axis=1,inplace=True)
print('drop unnamed')
'''
'''
#iot23
print("Dataset:iot23")
full_path = "/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/iot23.csv"
full_data = pd.read_csv(full_path)
full_data.drop(['Unnamed: 0'],axis=1,inplace=True)
print('drop unnamed')
'''

#combined
print("Dataset:combined")
iot23_path = "/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/iot23.csv"
botiot_path = "/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/botiot.csv"
mqtt_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/mqtt_reduli.csv"
iotid20_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoTID20*/iotid20_reduli.csv"
iot23_data = pd.read_csv(iot23_path)
botiot_data = pd.read_csv(botiot_path)
mqtt_data = pd.read_csv(mqtt_path)
iotid20_data = pd.read_csv(iotid20_path)
mqtt_data.drop(['Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Cat'],
               axis=1, inplace=True)
iotid20_data.drop([ 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Cat'],
               axis=1, inplace=True) #,'Unnamed: 0'
full_data  = pd.concat([iot23_data,botiot_data,mqtt_data, iotid20_data],axis=0)
full_data.drop(['Fwd_IAT_Std'],axis=1,inplace=True)
full_data.drop(['Unnamed: 0'],axis=1,inplace=True)
print(full_data)
print('drop unnamed')


xxx = full_data.loc[full_data.Flow_Duration=='Flow_Duration']
full_data = full_data.drop(xxx.index)
print('remove Flow_Duration')
print(full_data.shape[0])
print('remove inf')
full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
print(full_data.shape[0])
full_data = full_data.dropna()
print('remove nan')
print(full_data.shape[0])
full_data = full_data.drop_duplicates()
print(full_data.shape[0])
print(full_data)

class_full = full_data['Sub_Cat']
full_data.drop(['Sub_Cat'], axis=1, inplace=True)
full_data.insert(0, 'Sub_Cat', class_full)

# %%
"""
# Checking Balance of the Data
"""

print("full_data.Sub_Cat.value_counts():")
#synflooding       61248
#ackflooding       32332
#portos            10307
#hostport           7390
#httpflooding       4473
#hostbruteforce     3525
#arpspoofing        1335
#udpflooding         737
#Normal              431
print(full_data.Sub_Cat.value_counts())
d = full_data.Sub_Cat.value_counts()

# %%
"""
# Checking Missing Data and datatypes
"""

# %%
percent_missing = full_data.isnull().sum() * 100 / len(full_data)
print(percent_missing)
missing_values = pd.DataFrame({'% missing': percent_missing})
print(missing_values)
missing_values.sort_values(by ='% missing' , ascending=False)
print('full_data.dtypes:')
for i,v in zip(full_data.columns,full_data.dtypes):
    print(i,v)


# %%
pd.DataFrame({'Data Type': full_data.dtypes})
normalized_full_df = full_data.sample(frac=1,random_state=4)
normalized_full_df['Sub_Cat'] = (normalized_full_df['Sub_Cat'] != 'Normal').astype(int)

"""
# Splitting into Train and Test datasets
"""

X = normalized_full_df.drop(["Sub_Cat"], axis = 1)
for i in X.columns:
    print(i)
Y = normalized_full_df['Sub_Cat']
X2 = StandardScaler().fit_transform(X)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.2, random_state = 9)
print('train_test_split,finish')
#sklearn.model_selection.train_test_split随机划分训练集和测试集

fig, ax = plt.subplots(figsize=(12,6))
fig, ax1 = plt.subplots(figsize=(12,6))

"""
# Random Forests
"""

# %%
#%%time
print("----------------Random Forests----------------")
startrf = time.time()
print('program start...\n')
rfc = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train) #undersample
prediction_forest = rfc.predict(X_Test)
print(classification_report(Y_Test,prediction_forest))
print(f"Accuracy: {rfc.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(rfc))} bytes")

cm_forest = confusion_matrix(Y_Test, prediction_forest, labels=[1,0])
print(cm_forest)

endrf = time.time()

print('time cost: ')
print(endrf - startrf, 'seconds')

# %%
"""
## Feature Importance using RF
"""

# %%
feat_importances = pd.Series(rfc.feature_importances_, index= X.columns).sort_values(ascending=False)
#fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance Using Random Forest in MQTT-IoT-IDS2020',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
#fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance Using Random Forest in IoTID20',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
#fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance Using Random Forest in IoT-23',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
#fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance Using Random Forest in BoT_IoT',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
fig = px.bar(d, x=feat_importances.index, y=feat_importances.values,title = 'Feature Importance Using Random Forest in Combined Dataset',labels = {'x':'Feature','y':'Importance'},color=feat_importances.values,text_auto=False)
fig.update_layout(title_x=0.5,width=1000, height=400,font_size=6)
fig.show()

"""
# Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("----------------Logistic Regression----------------")
startlr = time.time()
print('program start...\n')
lrc = LogisticRegression(solver='liblinear', random_state=0).fit(X_Train, Y_Train)
prediction_forest = lrc.predict(X_Test)

print(classification_report(Y_Test,prediction_forest))
print(f"Accuracy: {lrc.score(X_Test, Y_Test)}")
print(f"Model size: {sys.getsizeof(pickle.dumps(lrc))} bytes")
cm_forest = confusion_matrix(Y_Test, prediction_forest, labels=[1,0])
print(cm_forest)
endlr = time.time()
print('time cost: ')
print(endlr - startlr, 'seconds')

"""
## Feature Importance using LR
"""
# get importance
importance = [abs(x) for x in lrc.coef_[0]] #
x = ['Protocol','Flow_Duration','Flow_Byts/s','Bwd_IAT_Mean.1','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Tot_Fwd_Pkts','Tot_Bwd_Pkts','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Max','Fwd_Pkt_Len_Min','Fwd_Pkt_Len_Mean','Fwd_Pkt_Len_Std','Bwd_Pkt_Len_Max','Bwd_Pkt_Len_Min','Bwd_Pkt_Len_Mean','Bwd_Pkt_Len_Std','Pkt_Len_Max','Pkt_Len_Min','Pkt_Len_Mean','Pkt_Len_Std','Pkt_Len_Var','Fwd_Header_Len','Bwd_Header_Len','Fwd_Act_Data_Pkts','Flow_Iat_Mean','Flow_Iat_Max','Flow_Iat_Min','Flow_Iat_Std','Fwd_Iat_Tot','Fwd_Iat_Max','Fwd_Iat_Min','Fwd_Iat_Mean','Bwd_Iat_Tot','Bwd_Iat_Max','Bwd_Iat_Min','Bwd_Iat_Mean','Bwd_Iat_Std','Fwd_PSH_Flags','Bwd_PSH_Flags','Fwd_URG_Flags','Bwd_URG_Flags','Fin_Flag_Cnt','Syn_Flag_Cnt','Rst_Flag_Cnt','PSH_Flag_Cnt','Ack_Flag_Cnt','URG_Flag_Cnt','Ece_Flag_Cnt','Down_Up_Ratio','Pkt_Size_Avg','Init_Fwd_Win_Byts','Init_Bwd_Win_Byts','Active_Max','Active_Min','Active_Mean','Active_Std','Idle_Max','Idle_Min','Idle_Mean','Idle_Std','Fwd_Byts_b_Avg','Fwd_Pkts_b_Avg','Bwd_Byts_b_Avg','Bwd_Pkts_b_Avg','Fwd_Blk_Rate_Avg','Bwd_Blk_Rate_Avg','Fwd_Seg_Size_Avg','Bwd_Seg_Size_Avg','Cwe_Flag_Count','Subflow_Fwd_Pkts','Subflow_Bwd_Pkts','Subflow_Fwd_Byts','Subflow_Bwd_Byts']
for i,v in zip(x,importance):
	print('Feature: {}, Score: {:.5}'.format(i,v))
for i in x:
    print(i)
print(len(importance))
# plot feature importance
fig = pyplot.figure(figsize=(14,9))
#pyplot.title("Feature Importance Using Logistic Regression in IoTID20")
#pyplot.title("Feature Importance Using Logistic Regression in IoT-23")
#pyplot.title("Feature Importance Using Logistic Regression in BoT_IoT")
#pyplot.title("Feature Importance Using Logistic Regression in MQTT-IoT-IDS2020")
pyplot.title("Feature Importance Using Logistic Regression in Combined Dataset")
pyplot.tick_params(axis='x', labelsize=6)    # 设置x轴标签大小
pyplot.bar(x, importance, width = 0.5)
plt.xticks(rotation=90)
pyplot.show()








