
import pandas as pd
import time
import numpy as np
import binary
def read_iotid20():
    full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoTID20*/iotid20_reduli.csv"
    print("Dataset:IoTID20")
    full_data = pd.read_csv(full_path)
    full_data.drop([ 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_IAT_Std'],
                     axis=1, inplace=True)
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())

    #remove depulicate
    full_data = full_data.drop_duplicates()
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())

    #undersample
    non_df = full_data.loc[(full_data['Sub_Cat'] != "synflooding")]
    syn_df = full_data.loc[full_data['Sub_Cat'] == "synflooding"].sample(frac=0.3, random_state=4)
    full_data_u = pd.concat([non_df, syn_df])
    print(full_data_u.shape[0])

    #remove nan inf odd value
    print('### remove FLow_Duration')
    xxx = full_data_u.loc[full_data_u.Flow_Duration == 'Flow_Duration']
    full_data_u = full_data_u.drop(xxx.index)
    print(full_data_u.shape[0])
    print('### remove inf')
    full_data_u.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data_u.shape[0])
    print('### remove nan')
    full_data_u = full_data_u.dropna()
    print(full_data_u.shape[0])

    full_data.to_csv('iotid20.csv')

    return full_data_u

def read_mqtt():

    full_path = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/MQTT-IoT-IDS2020*/mqtt_reduli.csv"
    print("Dataset:MQTT")
    full_data = pd.read_csv(full_path)

    full_data.drop([ 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_IAT_Std'],
                     axis=1, inplace=True)
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())

    #remove depulicate
    full_data = full_data.drop_duplicates()
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())

    #undersample
    non_df = full_data.loc[(full_data['Sub_Cat'] != "mqtt_bruteforce")&(full_data['Sub_Cat'] != "sparta")&(full_data['Sub_Cat'] != "Normal")]
    bru_df = full_data.loc[full_data['Sub_Cat'] == "mqtt_bruteforce"].sample(frac=0.04, random_state=4)
    spa_df = full_data.loc[full_data['Sub_Cat'] == "sparta"].sample(frac=0.15, random_state=4)
    normal_df = full_data.loc[full_data['Sub_Cat'] == "Normal"].sample(frac=0.35, random_state=4)
    full_data_u = pd.concat([non_df, bru_df,spa_df,normal_df])
    print(bru_df.shape[0])
    print(spa_df.shape[0])
    print(normal_df.shape[0])
    print(full_data_u.shape[0])


    #remove nan inf odd value
    print('### remove FLow_Duration')
    xxx = full_data_u.loc[full_data_u.Flow_Duration == 'Flow_Duration']
    full_data_u = full_data_u.drop(xxx.index)
    print(full_data_u.shape[0])
    print('### remove inf')
    full_data_u.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data_u.shape[0])
    print('### remove nan')
    full_data_u = full_data_u.dropna()
    print(full_data_u.shape[0])

    full_data.to_csv('mqtt.csv')

    return full_data_u

def read_iot23():
    '''700mb => 1400000 rows 0.05'''

    #botiot
    print("Dataset:IoT-23")
    start = time.time()
    path_attack = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Attack.csv"
    path_CC = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/CC.csv"
    path_DDoS_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/DDoS[1].csv"
    path_DDoS_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/DDoS[2].csv"
    path_FileDownload = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/FileDownload.csv"
    path_HeartBeat = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/HeartBeat.csv"
    path_Mirai = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Mirai.csv"
    path_Normal = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Normal.csv"
    path_Okiru_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Okiru[1].csv"
    path_Okiru_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Okiru[2].csv"
    path_PortScan_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/PortScan[1].csv"
    path_PortScan_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/PortScan[2].csv"
    path_Torii = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Torii.csv"

    attack_data = pd.read_csv(path_attack)
    CC_data = pd.read_csv(path_CC)
    DDoS_1_data = pd.read_csv(path_DDoS_1)
    DDoS_2_data = pd.read_csv(path_DDoS_2)
    FileDownload_data = pd.read_csv(path_FileDownload)
    HeartBeat_data = pd.read_csv(path_HeartBeat)
    Mirai_data = pd.read_csv(path_Mirai)
    Normal_data = pd.read_csv(path_Normal)
    Okiru_1_data = pd.read_csv(path_Okiru_1)
    Okiru_2_data = pd.read_csv(path_Okiru_2)
    PortScan_1_data = pd.read_csv(path_PortScan_1)
    PortScan_2_data = pd.read_csv(path_PortScan_2)
    Torii_data = pd.read_csv(path_Torii)

    Torii_data['Sub_Cat'] = Torii_data['Sub_Cat'].replace('C&C-Torii','Torii')
    PortScan_1_data['Sub_Cat'] = PortScan_1_data['Sub_Cat'].replace('PartOfAHorizontalPortScan','Scan')
    PortScan_2_data['Sub_Cat'] = PortScan_2_data['Sub_Cat'].replace('PartOfAHorizontalPortScan','Scan')
    DDoS_1_data['Sub_Cat'].loc[(DDoS_1_data['Protocol'] == 6) & DDoS_1_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_TCP'
    DDoS_1_data['Sub_Cat'].loc[(DDoS_1_data['Protocol'] == 17) & DDoS_1_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_UDP'
    DDoS_2_data['Sub_Cat'].loc[(DDoS_2_data['Protocol'] == 6) & DDoS_2_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_TCP'
    DDoS_2_data['Sub_Cat'].loc[(DDoS_2_data['Protocol'] == 17) & DDoS_2_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_UDP'

    attack_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    CC_data.drop(['Flow_ID','Bwd_IAT_Mean.1', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    DDoS_1_data.drop(['Flow_ID','Bwd_IAT_Mean.1', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    DDoS_2_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    FileDownload_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    HeartBeat_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    Mirai_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    Normal_data.drop(['Flow_ID','Bwd_IAT_Mean.1', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    Okiru_1_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    Okiru_2_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    PortScan_1_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    PortScan_2_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)
    Torii_data.drop(['Flow_ID', 'Bwd_IAT_Mean.1','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'], axis=1,inplace=True)

    print("original row numbers:")
    print(attack_data.shape[0])
    print(CC_data.shape[0])
    print(DDoS_1_data.shape[0])
    print(DDoS_2_data.shape[0])
    print(FileDownload_data.shape[0])
    print(HeartBeat_data.shape[0])
    print(Mirai_data.shape[0])
    print(Normal_data.shape[0])
    print(Okiru_1_data.shape[0])
    print(Okiru_2_data.shape[0])
    print(PortScan_1_data.shape[0])
    print(PortScan_2_data.shape[0])
    print(Torii_data.shape[0])

    #remove duplicate
    attack_data = attack_data.drop_duplicates()
    CC_data = CC_data.drop_duplicates()
    DDoS_1_data = DDoS_1_data.drop_duplicates()
    DDoS_2_data = DDoS_2_data.drop_duplicates()
    FileDownload_data = FileDownload_data.drop_duplicates()
    HeartBeat_data = HeartBeat_data.drop_duplicates()
    Mirai_data = Mirai_data.drop_duplicates()
    Normal_data = Normal_data.drop_duplicates()
    Okiru_1_data = Okiru_1_data.drop_duplicates()
    Okiru_2_data = Okiru_2_data.drop_duplicates()
    PortScan_1_data = PortScan_1_data.drop_duplicates()
    PortScan_2_data = PortScan_2_data.drop_duplicates()
    Torii_data = Torii_data.drop_duplicates()

    print("after remove duplicate row numbers:")
    print(attack_data.shape[0])
    print(CC_data.shape[0])
    print(DDoS_1_data.shape[0])
    print(DDoS_2_data.shape[0])
    print(FileDownload_data.shape[0])
    print(HeartBeat_data.shape[0])
    print(Mirai_data.shape[0])
    print(Normal_data.shape[0])
    print(Okiru_1_data.shape[0])
    print(Okiru_2_data.shape[0])
    print(PortScan_1_data.shape[0])
    print(PortScan_2_data.shape[0])
    print(Torii_data.shape[0])

    print('### undersample')
    attack_data = attack_data.sample(frac=0.015, random_state=4)
    Normal_data = Normal_data.sample(frac=0.007, random_state=4)
    print(attack_data.shape[0])
    print(Normal_data.shape[0])


    full_data = pd.concat([attack_data,CC_data,DDoS_1_data,DDoS_2_data,FileDownload_data,
                            HeartBeat_data,Mirai_data,Normal_data,Okiru_1_data,
                            Okiru_2_data,PortScan_1_data,PortScan_2_data,Torii_data
                           ],axis=0)

    # remove nan inf odd value
    print('### remove FLow_Duration')
    xxx = full_data.loc[full_data.Flow_Duration == 'Flow_Duration']
    full_data = full_data.drop(xxx.index)
    print(full_data.shape[0])
    print('### remove inf')
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data.shape[0])
    print('### remove nan')
    full_data = full_data.dropna()
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())

    full_data.to_csv('iot23.csv')

    return full_data

def read_botiot():

    #botiot
    print("Dataset:BoT_IoT")
    start = time.time()
    path_DE = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Data_Exfiltration.csv"
    path_DDoS_HTTP = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_HTTP[1].csv"
    path_DDoS_TCP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[1].csv"
    path_DDoS_TCP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[2].csv"
    path_DDoS_TCP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[3].csv"
    path_DDoS_UDP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[1].csv"
    path_DDoS_UDP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[2].csv"
    path_DDoS_UDP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[3].csv"
    path_DoS_HTTP = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_HTTP[1].csv"
    path_DoS_TCP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[1].csv"
    path_DoS_TCP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[2].csv"
    path_DoS_TCP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[3].csv"
    path_DoS_TCP_4 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[4].csv"
    path_DoS_UDP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[1].csv"
    path_DoS_UDP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[2].csv"
    path_DoS_UDP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[3].csv"
    path_DoS_UDP_4 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[4].csv"
    path_KL = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Keylogging.csv"
    path_OS = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/OS.csv"
    path_Service = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Service.csv"

    DE_data = pd.read_csv(path_DE)
    DDoS_HTTP_data = pd.read_csv(path_DDoS_HTTP)
    DDoS_TCP_1_data = pd.read_csv(path_DDoS_TCP_1)
    DDoS_TCP_2_data = pd.read_csv(path_DDoS_TCP_2)
    DDoS_TCP_3_data = pd.read_csv(path_DDoS_TCP_3)
    DDoS_UDP_1_data = pd.read_csv(path_DDoS_UDP_1)
    DDoS_UDP_2_data = pd.read_csv(path_DDoS_UDP_2)
    DDoS_UDP_3_data = pd.read_csv(path_DDoS_UDP_3)
    DoS_HTTP_data = pd.read_csv(path_DoS_HTTP)
    DoS_TCP_1_data = pd.read_csv(path_DoS_TCP_1)
    DoS_TCP_2_data = pd.read_csv(path_DoS_TCP_2)
    DoS_TCP_3_data = pd.read_csv(path_DoS_TCP_3)
    DoS_TCP_4_data = pd.read_csv(path_DoS_TCP_4)
    DoS_UDP_1_data = pd.read_csv(path_DoS_UDP_1)
    DoS_UDP_2_data = pd.read_csv(path_DoS_UDP_2)
    DoS_UDP_3_data = pd.read_csv(path_DoS_UDP_3)
    DoS_UDP_4_data = pd.read_csv(path_DoS_UDP_4)
    KL_data = pd.read_csv(path_KL)
    OS_data = pd.read_csv(path_OS)
    Service_data = pd.read_csv(path_Service)

    OS_data['Sub_Cat'] = OS_data['Sub_Cat'].replace('OS','Scan')
    Service_data['Sub_Cat'] = Service_data['Sub_Cat'].replace('Service','Scan')

    DE_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_HTTP_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_TCP_1_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_TCP_2_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_TCP_3_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_UDP_1_data.drop(['Flow_ID','Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_UDP_2_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DDoS_UDP_3_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_HTTP_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_TCP_1_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_TCP_2_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_TCP_3_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_TCP_4_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_UDP_1_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_UDP_2_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_UDP_3_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    DoS_UDP_4_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    KL_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    OS_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)
    Service_data.drop(['Flow_ID',  'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label",'Fwd_Seg_Size_Min'], axis=1, inplace=True)

    print("original row numbers:")
    print(DE_data.shape[0])#376
    print(DDoS_HTTP_data.shape[0])#34454
    print(DDoS_TCP_1_data.shape[0])#70198
    print(DDoS_TCP_2_data.shape[0])#77784
    print(DDoS_TCP_3_data.shape[0])#83951
    print(DDoS_UDP_1_data.shape[0])#68410
    print(DDoS_UDP_2_data.shape[0])#68410
    print(DDoS_UDP_3_data.shape[0])#68410
    print(DoS_HTTP_data.shape[0]) #52466
    print(DoS_TCP_1_data.shape[0])#86451
    print(DoS_TCP_2_data.shape[0])#86451
    print(DoS_TCP_3_data.shape[0])#86451
    print(DoS_TCP_4_data.shape[0])#86451
    print(DoS_UDP_1_data.shape[0])#77204
    print(DoS_UDP_2_data.shape[0])#77204
    print(DoS_UDP_3_data.shape[0])#77204
    print(DoS_UDP_4_data.shape[0])#77204
    print(KL_data.shape[0])#2046
    print(OS_data.shape[0])#196645
    print(Service_data.shape[0])#273778

    #remove duplicate
    DE_data = DE_data.drop_duplicates()
    DDoS_HTTP_data = DDoS_HTTP_data.drop_duplicates()
    DDoS_TCP_1_data = DDoS_TCP_1_data.drop_duplicates()
    DDoS_TCP_2_data = DDoS_TCP_2_data.drop_duplicates()
    DDoS_TCP_3_data = DDoS_TCP_3_data.drop_duplicates()
    DDoS_UDP_1_data = DDoS_UDP_1_data.drop_duplicates()
    DDoS_UDP_2_data = DDoS_UDP_2_data.drop_duplicates()
    DDoS_UDP_3_data = DDoS_UDP_3_data.drop_duplicates()
    DoS_HTTP_data = DoS_HTTP_data.drop_duplicates()
    DoS_TCP_1_data = DoS_TCP_1_data.drop_duplicates()
    DoS_TCP_2_data = DoS_TCP_2_data.drop_duplicates()
    DoS_TCP_3_data = DoS_TCP_3_data.drop_duplicates()
    DoS_TCP_4_data = DoS_TCP_4_data.drop_duplicates()
    DoS_UDP_1_data = DoS_UDP_1_data.drop_duplicates()
    DoS_UDP_2_data = DoS_UDP_2_data.drop_duplicates()
    DoS_UDP_3_data = DoS_UDP_3_data.drop_duplicates()
    DoS_UDP_4_data = DoS_UDP_4_data.drop_duplicates()
    KL_data = KL_data.drop_duplicates()
    OS_data = OS_data.drop_duplicates()
    Service_data = Service_data.drop_duplicates()

    print("after remove duplicate row numbers:")
    print(DE_data.shape[0])#376
    print(DDoS_HTTP_data.shape[0])#34454
    print(DDoS_TCP_1_data.shape[0])#70198
    print(DDoS_TCP_2_data.shape[0])#77784
    print(DDoS_TCP_3_data.shape[0])#83951
    print(DDoS_UDP_1_data.shape[0])#68410
    print(DDoS_UDP_2_data.shape[0])#68410
    print(DDoS_UDP_3_data.shape[0])#68410
    print(DoS_HTTP_data.shape[0]) #52466
    print(DoS_TCP_1_data.shape[0])#86451
    print(DoS_TCP_2_data.shape[0])#86451
    print(DoS_TCP_3_data.shape[0])#86451
    print(DoS_TCP_4_data.shape[0])#86451
    print(DoS_UDP_1_data.shape[0])#77204
    print(DoS_UDP_2_data.shape[0])#77204
    print(DoS_UDP_3_data.shape[0])#77204
    print(DoS_UDP_4_data.shape[0])#77204
    print(KL_data.shape[0])#2046
    print(OS_data.shape[0])#196645
    print(Service_data.shape[0])#273778
    #
    #
    DDoS_TCP_1_data= DDoS_TCP_1_data.sample(frac=0.006,random_state=4)
    DDoS_TCP_2_data= DDoS_TCP_2_data.sample(frac=0.008,random_state=4)
    DDoS_TCP_3_data = DDoS_TCP_3_data.sample(frac=0.016,random_state=4)
    DDoS_UDP_1_data = DDoS_UDP_1_data.sample(frac=0.003,random_state=4)
    DDoS_UDP_2_data = DDoS_UDP_2_data.sample(frac=0.003,random_state=4)
    DDoS_UDP_3_data = DDoS_UDP_3_data.sample(frac=0.002,random_state=4)
    DoS_HTTP_data = DoS_HTTP_data.sample(frac=0.5,random_state=4)
    DoS_TCP_1_data = DoS_TCP_1_data.sample(frac=0.005, random_state=4)
    DoS_TCP_2_data = DoS_TCP_2_data.sample(frac=0.0025, random_state=4)
    DoS_TCP_3_data = DoS_TCP_3_data.sample(frac=0.0025, random_state=4)
    DoS_TCP_4_data = DoS_TCP_4_data.sample(frac=0.05, random_state=4)
    DoS_UDP_1_data = DoS_UDP_1_data.sample(frac=0.005, random_state=4)
    DoS_UDP_2_data = DoS_UDP_2_data.sample(frac=0.0025, random_state=4)
    DoS_UDP_3_data = DoS_UDP_3_data.sample(frac=0.0025, random_state=4)
    DoS_UDP_4_data = DoS_UDP_4_data.sample(frac=0.0025, random_state=4)
    OS_data = OS_data.sample(frac=0.01, random_state=4)
    Service_data = Service_data.sample(frac=0.13, random_state=4)

    print("undersample:")
    print(DE_data.shape[0])#376
    print(DDoS_HTTP_data.shape[0])#34454
    print(DDoS_TCP_1_data.shape[0])#70198
    print(DDoS_TCP_2_data.shape[0])#77784
    print(DDoS_TCP_3_data.shape[0])#83951
    print(DDoS_UDP_1_data.shape[0])#68410
    print(DDoS_UDP_2_data.shape[0])#68410
    print(DDoS_UDP_3_data.shape[0])#68410
    print(DoS_HTTP_data.shape[0]) #52466
    print(DoS_TCP_1_data.shape[0])#86451
    print(DoS_TCP_2_data.shape[0])#86451
    print(DoS_TCP_3_data.shape[0])#86451
    print(DoS_TCP_4_data.shape[0])#86451
    print(DoS_UDP_1_data.shape[0])#77204
    print(DoS_UDP_2_data.shape[0])#77204
    print(DoS_UDP_3_data.shape[0])#77204
    print(DoS_UDP_4_data.shape[0])#77204
    print(KL_data.shape[0])#2046
    print(OS_data.shape[0])#196645
    print(Service_data.shape[0])#273778
    #
    #
    full_data = pd.concat([DE_data,DDoS_HTTP_data,DDoS_TCP_1_data,DDoS_TCP_2_data,DDoS_TCP_3_data,
                            DDoS_UDP_1_data,DDoS_UDP_2_data,DDoS_UDP_3_data,DoS_HTTP_data,
                            DoS_TCP_1_data,DoS_TCP_2_data,DoS_TCP_3_data,DoS_TCP_4_data,
                            DoS_UDP_1_data,DoS_UDP_2_data,DoS_UDP_3_data,DoS_UDP_4_data,
                            KL_data,OS_data,Service_data
                           ],axis=0)

    # remove nan inf odd value
    print('### remove FLow_Duration')
    xxx = full_data.loc[full_data.Flow_Duration == 'Flow_Duration']
    full_data = full_data.drop(xxx.index)
    print(full_data.shape[0])
    print('### remove inf')
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data.shape[0])
    print('### remove nan')
    full_data = full_data.dropna()
    print(full_data.shape[0])
    print(full_data.Sub_Cat.value_counts())
    full_data.to_csv('botiot.csv')

    return full_data

def combine():
    iot23 = read_iot23()
    mqtt = read_mqtt()
    iotid20 = read_iotid20()
    botiot = read_botiot()
    full_data =  pd.concat([iot23, mqtt, iotid20, botiot], axis=0)
    print(full_data.Sub_Cat.value_counts())
    full_data.to_csv('combine.csv')
    return full_data

#x = read_iot23()
#x = read_iotid20()
# x = read_mqtt()
#read_botiot()
print(combine())