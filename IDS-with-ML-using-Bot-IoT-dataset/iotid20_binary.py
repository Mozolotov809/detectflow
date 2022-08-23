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
import datetime
warnings.filterwarnings("ignore")

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
import read_large_file

def read_file():

    #iotid20
    # full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoTID20*/iotid20_reduli.csv"
    # print("Dataset:IoTID20")
    # full_data = pd.read_csv(full_path)

    #mqtt
    # full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/mqtt_reduli.csv"
    # print("Dataset:MQTT")
    # full_data = pd.read_csv(full_path)

    #botiot
    #full_data = read_large_file.read_botiot()

    #iot23
    #full_data = read_large_file.read_iot23()

    #combine
    iotid20_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoTID20*/iotid20_reduli.csv"
    iotid20_data = pd.read_csv(iotid20_path)
    mqtt_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/mqtt_reduli.csv"
    mqtt_data = pd.read_csv(mqtt_path)
    mqtt_data['Sub_Cat'] = mqtt_data['Sub_Cat'].replace('scan_A','Scan')
    mqtt_data['Sub_Cat'] = mqtt_data['Sub_Cat'].replace('scan_sU','Scan')
    normalized_mqtt_data = undersampling(mqtt_data)
    #botiot_data,botiot_data_o = read_large_file.read_botiot()
    botiot_data = read_large_file.read_botiot()
    #iot23_data,iot23_data_o = read_large_file.read_iot23()
    iot23_data = read_large_file.read_iot23()
    print('iot23_data.Sub_Cat.value_counts() in binary')
    print(iot23_data.Sub_Cat.value_counts())
    #full_data_o = pd.concat([iot23_data_o, botiot_data_o, normalized_mqtt_data, iotid20_data], axis=0)
    full_data = pd.concat([iot23_data, botiot_data, normalized_mqtt_data, iotid20_data], axis=0)

    full_data = full_data[['Sub_Cat',"Bwd_Pkts/s","Flow_IAT_Max","Flow_IAT_Mean","Flow_Duration",
                           "Fwd_IAT_Max","Fwd_IAT_Tot","Fwd_Pkts/s","Bwd_IAT_Mean","Flow_IAT_Min",
                           "Fwd_Header_Len","Fwd_IAT_Mean","Fwd_IAT_Min","Init_Fwd_Win_Byts",
                           "Subflow_Fwd_Byts","Bwd_IAT_Max","Flow_Byts/s","Flow_IAT_Std","Flow_Pkts/s","Fwd_Pkt_Len_Mean",
                           "Idle_Max","Idle_Mean","Idle_Min","Pkt_Len_Max","Pkt_Size_Avg","PSH_Flag_Cnt","Subflow_Fwd_Pkts",
                           "SYN_Flag_Cnt","Tot_Fwd_Pkts"]]

    xxx = full_data.loc[full_data.Flow_Duration=='flow_duration']
    full_data = full_data.drop(xxx.index)
    print(full_data)
    print("full_data.Sub_Cat.value_counts():")
    print(full_data.Sub_Cat.value_counts())

    class_full = full_data['Sub_Cat']
    full_data.drop(['Sub_Cat'], axis=1, inplace=True)
    full_data.insert(0, 'Sub_Cat', class_full)
    #return full_data,full_data_o
    return full_data

def graph_balance(full_data,dataset):

    print("full_data.Sub_Cat.value_counts():")
    print(full_data.Sub_Cat.value_counts())

    d = full_data.Sub_Cat.value_counts()
    title = 'Class distribution between attack subcategories on full data in '+ dataset
    fig = px.bar(d, x=d.index, y=d.values,title = title,labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
    fig.update_layout(title_x=0.5,width=1000, height=400)
    fig.show()

def missing(full_data):

    percent_missing = full_data.isnull().sum() * 100 / len(full_data)
    print(percent_missing)
    missing_values = pd.DataFrame({'% missing': percent_missing})
    print(missing_values)
    missing_values.sort_values(by ='% missing' , ascending=False)
    print(full_data.dtypes)
    pd.DataFrame({'Data Type': full_data.dtypes})

def undersampling(full_data):

    shuffled_df = full_data.sample(frac=1,random_state=4)
    #从所选的数据的指定 axis 上返回随机抽样结果，类似于random.sample()函数。意味着frac=1返回所有行
    #shuffled_df 是full_data 抽样后的结果

    #iotid20
    # nonsynack_df = shuffled_df.loc[(shuffled_df['Sub_Cat'] != "synflooding") & (shuffled_df['Sub_Cat'] != "ackflooding")]
    # syn_df = shuffled_df.loc[shuffled_df['Sub_Cat'] == "synflooding"].sample(n=10307,random_state=42)
    # ack_df = shuffled_df.loc[shuffled_df['Sub_Cat'] == "ackflooding"].sample(n=10307,random_state=42)
    # normalized_full_df = pd.concat([nonsynack_df, syn_df, ack_df])
    #pandas.concat()通常用来连接DataFrame对象，默认纵向，axis=1 横向连接
    #mqtt
    nonspbrno_df = shuffled_df.loc[(shuffled_df['Sub_Cat'] != "sparta") & (shuffled_df['Sub_Cat'] != "mqtt_bruteforce")]
    sparta_df = shuffled_df.loc[shuffled_df['Sub_Cat'] == "sparta"].sample(n=88575,random_state=42)
    bruteforce_df = shuffled_df.loc[shuffled_df['Sub_Cat'] == "mqtt_bruteforce"].sample(n=88575,random_state=42)
    normalized_full_df = pd.concat([nonspbrno_df, sparta_df, bruteforce_df])
    return normalized_full_df

def graph_undersample(normalized_full_df,dataset):
    #TRAIN after undersampling
    d = normalized_full_df.Sub_Cat.value_counts()
    title = 'Class Label Distribution in ' + dataset +' (undersampled)'
    fig = px.bar(d, x=d.index, y=d.values,title = title,labels = {'index':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
    fig.update_layout(title_x=0.5,width=1000, height=400)
    fig.show()

def binary_classification(normalized_full_df):

    normalized_full_df['Sub_Cat'] = (normalized_full_df['Sub_Cat'] != 'Normal').astype(int)
    #numpy astype（int）转化为整型数据
    return normalized_full_df

def split_train_test(normalized_full_df):

    X = normalized_full_df.drop(["Sub_Cat"], axis = 1)
    print('X = normalized_full_df.drop(["Sub_Cat"], axis = 1),finish')
    #X = pd.get_dummies(X, prefix_sep='_')
    #print("X = pd.get_dummies(X, prefix_sep='_'),finish")
    #print(X)
    Y = normalized_full_df['Sub_Cat']
    print(Y)
    #x具有输入 ( )的二维数组
    #y具有输出 ( )的一维数组

    X2 = StandardScaler().fit_transform(X)
    print(X2.shape[0])
    print('X2 = StandardScaler().fit_transform(X),finish')
    #fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.2, random_state = 9)
    print('train_test_split,finish')
    #sklearn.model_selection.train_test_split随机划分训练集和测试集

    return X_Train,X_Test,Y_Train,Y_Test

def graph_label_distribution_testing(Y_Train,dataset):

    d = pd.Series(data=Counter(Y_Train).values(), index = Counter(Y_Train).keys())
    title = 'Class Label Distribution in ' + dataset +' - Testing'
    fig = px.bar(d, x=d.index, y=d.values,title = title,labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.3s')
    fig.update_layout(title_x=0.5,width=1000, height=400)
    fig.show()

def oversampling(normalized_full_df,X_Train,Y_Train):

    sm = SMOTE(random_state=42,sampling_strategy='auto') #k_neighbors=3
    #SMOTE模型默认生成一比一的数据
    X_Smote_Train, Y_Smote_Train = sm.fit_resample(X_Train, Y_Train)
    oversampled_train = pd.concat([pd.DataFrame(Y_Smote_Train), pd.DataFrame(X_Smote_Train)], axis=1)
    oversampled_train.columns = normalized_full_df.columns

    # %%
    print(f"Before: {Counter(Y_Train)}, number of records: {sum(Counter(Y_Train).values())}")
    print(f"After: {Counter(Y_Smote_Train)}, number of records: {sum(Counter(Y_Smote_Train).values())}")
    return oversampled_train,X_Smote_Train,Y_Smote_Train

def graph_oversampling(oversampled_train,dataset):
    # %%
    d = oversampled_train.Sub_Cat.value_counts()
    #fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in Bot-IoT (oversampled with SMOTE)',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
    fig = px.bar(d, x=d.index, y=d.values,title = 'Class Label Distribution in MQTT-IoT-IDS2020 (oversampled with SMOTE)',labels = {'x':'Attack','y':'Volume'},color=d.values,text_auto='.2s')
    fig.update_layout(title_x=0.5,width=1000, height=400)
    fig.show()

def ranfom_forest(X_Train, X_Test, Y_Train, Y_Test):

    print("----------------Random Forests----------------")
    startrf = time.time()
    print('program start...\n')
    rfc = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_Train,Y_Train) #undersample
    #rfc = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_Smote_Train,Y_Smote_Train) #SMOTE
    prediction_forest = rfc.predict(X_Test)
    print(classification_report(Y_Test,prediction_forest))
    print(f"Accuracy: {rfc.score(X_Test, Y_Test)}")
    print(f"Model size: {sys.getsizeof(pickle.dumps(rfc))} bytes")

    cm_forest = confusion_matrix(Y_Test, prediction_forest, labels=[1,0])
    print(cm_forest)

    endrf = time.time()

    print('time cost: ')
    print(endrf - startrf, 'seconds')

def k_nearest_neighbour(X_Train, X_Test, Y_Train, Y_Test):

    print("----------------K Nearest Neighbour----------------")
    startknn = time.time()
    print('program start...\n')
    neigh = KNeighborsClassifier().fit(X_Train,Y_Train)  #undersample
    #neigh = KNeighborsClassifier().fit(X_Smote_Train,Y_Smote_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train n_neighbors=3
    prediction_neigh = neigh.predict(X_Test)
    print(classification_report(Y_Test,prediction_neigh))
    print(f"Accuracy: {neigh.score(X_Test, Y_Test)}")
    print(f"Model size: {sys.getsizeof(pickle.dumps(neigh))} bytes")

    cm_knn = confusion_matrix(Y_Test, prediction_neigh, labels=[1,0])
    print(cm_knn)
    endknn = time.time()
    print('time cost: ')
    print(endknn - startknn, 'seconds')

def support_vector_machines(X_Train, X_Test, Y_Train, Y_Test):

    print("----------------Support Vector Machines (SVM)----------------")
    startsvm = time.time()
    print('program start...\n')
    svmclf = svm.SVC(random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
    #svmclf = svm.SVC(random_state=42).fit(X_Smote_Train,Y_Smote_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
    predictions_svm = svmclf.predict(X_Test)
    print(classification_report(Y_Test,predictions_svm))
    print(f"Accuracy: {svmclf.score(X_Test, Y_Test)}")
    print(f"Model size: {sys.getsizeof(pickle.dumps(svmclf))} bytes")

    cm_svm = confusion_matrix(Y_Test, predictions_svm, labels=[1,0])
    print(cm_svm)
    endsvm = time.time()
    print('time cost: ')
    print(endsvm - startsvm, 'seconds')

def artifical_neural_networks(X_Train, X_Test, Y_Train, Y_Test,feature_number):

    print("----------------Artifical Neural Networks----------------")
    ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=42).fit(X_Train,Y_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
    #ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=42).fit(X_Smote_Train,Y_Smote_Train)  #SMOTE: X_Smote_Train,Y_Smote_Train
    prediction_ann = ANN.predict(X_Test)
    print(classification_report(Y_Test,prediction_ann))
    print(f"Accuracy: {ANN.score(X_Test, Y_Test)}")
    print(f"Model size: {sys.getsizeof(pickle.dumps(ANN))} bytes")

    cm_ann = confusion_matrix(Y_Test, prediction_ann, labels=[1,0])
    print(cm_ann)

    # %%
    #%%time



    # Initialising the ANN
    ann_clf = Sequential()
    #模型Sequential适用于简单的层堆栈， 其中每一层恰好有一个输入张量和一个输出张量。
    # Adding the input layer and the first hidden layer
    ann_clf.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = feature_number))

    # Adding the second hidden layer
    # classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    ann_clf.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

    # Compiling the ANN
    ann_clf.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    #summary
    ann_clf.summary()

    startann = time.time()
    print('program start...\n')

    # Fitting the ANN to the Training set
    ann_clf.fit(X_Train,Y_Train , epochs = 10) #SMOTE: X_Smote_Train,Y_Smote_Train

    # Predicting the Test set results
    predict_x=ann_clf.predict(X_Test)
    classes_x=np.argmax(predict_x,axis=1)

    print('classification_report(Y_Test, classes_x):')
    print(classification_report(Y_Test, classes_x))
    print(f"Model size: {sys.getsizeof(pickle.dumps(ann_clf))} bytes")
    endann = time.time()
    print('time cost: ')
    print(endann - startann, 'seconds')

def c_neural_networks(X_Train, X_Test, Y_Train, Y_Test,feature_number):

    print("----------------C Neural Networks----------------")
    cnn_clf = Sequential()
    cnn_clf.add(Dense(2000, activation='relu', input_dim=feature_number))
    cnn_clf.add(Dense(1500, activation='relu'))
    cnn_clf.add(Dropout(0.2))
    cnn_clf.add(Dense(800, activation='relu'))
    cnn_clf.add(Dropout(0.2))
    cnn_clf.add(Dense(400, activation='relu'))
    cnn_clf.add(Dropout(0.2))
    cnn_clf.add(Dense(150, activation='relu'))
    cnn_clf.add(Dropout(0.2))
    cnn_clf.add(Dense(12, activation='softmax'))
    #cnn_clf.add(Dense(12, activation='Sigmoid'))
    cnn_clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_clf.summary())

    startcnn = time.time()
    print('program start...\n')
    print(f"Model size: {sys.getsizeof(pickle.dumps(cnn_clf))} bytes")

    history = cnn_clf.fit(X_Train, Y_Train, epochs = 10, batch_size=256, validation_data=(X_Test, Y_Test), verbose=1)
    #history = cnn_clf.fit(X_Smote_Train, Y_Smote_Train, epochs = 10, batch_size=256, validation_data=(X_Test, Y_Test), verbose=1)
    # Predicting the Test set results
    predict_c=cnn_clf.predict(X_Test)
    y_pred=np.argmax(predict_c,axis=1)
    #y_pred = cnn_clf.fit(X_Train, Y_Train).predict(X_Test)
    print(y_pred)

    endcnn = time.time()
    print('time cost: ')
    print(endcnn - startcnn, 'seconds')
    print('classification_report(Y_Test, classes_x):')
    print(classification_report(Y_Test, y_pred))

def logistic_regression(X_Train, X_Test, Y_Train, Y_Test, feature_number):

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

if __name__ == "__main__":
    # dataset = 'IoTID20'
    # dataset = 'MQTT-IoT-IDS2020'
    #dataset = 'BoT_IoT'
    # dataset = 'IoT_23'
    dataset = 'Combined'
    print('-------------'+ dataset + '-rf_long_fs--------------')
    full_data = read_file()
    #full_data,full_data_o = read_file()
    #graph_balance(full_data_o,dataset)
    #graph_balance(full_data,dataset) #算不出来
    missing(full_data)
    # normalized_full_df = undersampling(full_data)
    normalized_full_df = full_data
    graph_undersample(normalized_full_df,dataset)
    normalized_full_df = binary_classification(normalized_full_df)
    feature_number = 28

    print('normal and abnormal number:')
    print(normalized_full_df.Sub_Cat.value_counts())
    print('top 5 rows:')
    print(normalized_full_df.head())

    X_Train,X_Test,Y_Train,Y_Test = split_train_test(normalized_full_df)
    graph_label_distribution_testing(Y_Test,dataset)
    print('X_Train.shape,X_Test.shape:')
    print(X_Train.shape, X_Test.shape)
    print('Y_Train.shape,Y_Test.shape:')
    print(Y_Train.shape, Y_Test.shape)
    print(Counter(Y_Test))
    print(Counter(Y_Train))
    # 1:attack 0:Normal

    ranfom_forest(X_Train, X_Test, Y_Train, Y_Test)
    k_nearest_neighbour(X_Train, X_Test, Y_Train, Y_Test)
    support_vector_machines(X_Train, X_Test, Y_Train, Y_Test)
    artifical_neural_networks(X_Train, X_Test, Y_Train, Y_Test, feature_number)
    c_neural_networks(X_Train, X_Test, Y_Train, Y_Test, feature_number)
    logistic_regression(X_Train, X_Test, Y_Train, Y_Test, feature_number)
