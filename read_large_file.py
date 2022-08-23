#iotid20
import pandas as pd
import time
import numpy as np
import binary
def read_botiot():
    '''700mb => 1400000 rows 0.05'''

    #botiot
    print("Dataset:BoT_IoT")
    start = time.time()
    path_DE = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Data_Exfiltration.csv"
    DE_data = pd.read_csv(path_DE)
    #print(DE_data.shape[0])#376
    path_DDoS_HTTP = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_HTTP[1].csv"
    DDoS_HTTP_data = pd.read_csv(path_DDoS_HTTP)
    #print(DDoS_HTTP_data.shape[0])#34454
    path_DDoS_TCP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[1].csv"
    DDoS_TCP_1_data = pd.read_csv(path_DDoS_TCP_1)
    DDoS_TCP_1_data_u = DDoS_TCP_1_data.sample(frac=0.05,random_state=4)
    #print(DDoS_TCP_1_data.shape[0])#70198
    path_DDoS_TCP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[2].csv"
    DDoS_TCP_2_data = pd.read_csv(path_DDoS_TCP_2)
    DDoS_TCP_2_data_u = DDoS_TCP_2_data.sample(frac=0.05,random_state=4)
    #print(DDoS_TCP_2_data.shape[0])#77784
    path_DDoS_TCP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_TCP[3].csv"
    DDoS_TCP_3_data = pd.read_csv(path_DDoS_TCP_3)
    DDoS_TCP_3_data_u = DDoS_TCP_3_data.sample(frac=0.05,random_state=4)
    #print(DDoS_TCP_3_data.shape[0])#83951
    path_DDoS_UDP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[1].csv"
    DDoS_UDP_1_data = pd.read_csv(path_DDoS_UDP_1)
    DDoS_UDP_1_data_u = DDoS_UDP_1_data.sample(frac=0.03,random_state=4)
    #print(DDoS_UDP_1_data.shape[0])#68410
    path_DDoS_UDP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[2].csv"
    DDoS_UDP_2_data = pd.read_csv(path_DDoS_UDP_2)
    DDoS_UDP_2_data_u = DDoS_UDP_2_data.sample(frac=0.03,random_state=4)
    #print(DDoS_UDP_2_data.shape[0])#93470
    path_DDoS_UDP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DDoS_UDP[3].csv"
    DDoS_UDP_3_data = pd.read_csv(path_DDoS_UDP_3)
    DDoS_UDP_3_data_u = DDoS_UDP_3_data.sample(frac=0.025,random_state=4)
    #print(DDoS_UDP_3_data.shape[0])#100901
    path_DoS_HTTP = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_HTTP[1].csv"
    DoS_HTTP_data = pd.read_csv(path_DoS_HTTP)
    #print(DoS_HTTP_data.shape[0]) #52466
    path_DoS_TCP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[1].csv"
    DoS_TCP_1_data = pd.read_csv(path_DoS_TCP_1)
    DoS_TCP_1_data_u = DoS_TCP_1_data.sample(frac=0.05, random_state=4)
    #print(DoS_TCP_1_data.shape[0])#86451
    path_DoS_TCP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[2].csv"
    DoS_TCP_2_data = pd.read_csv(path_DoS_TCP_2)
    DoS_TCP_2_data_u = DoS_TCP_2_data.sample(frac=0.025, random_state=4)
    #print(DoS_TCP_2_data.shape[0])#66684
    path_DoS_TCP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[3].csv"
    DoS_TCP_3_data = pd.read_csv(path_DoS_TCP_3)
    DoS_TCP_3_data_u = DoS_TCP_3_data.sample(frac=0.03, random_state=4)
    #print(DoS_TCP_3_data.shape[0])#76245
    path_DoS_TCP_4 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_TCP[4].csv"
    DoS_TCP_4_data = pd.read_csv(path_DoS_TCP_4)
    DoS_TCP_4_data_u = DoS_TCP_4_data.sample(frac=0.05, random_state=4)
    #print(DoS_TCP_4_data.shape[0])#86451
    path_DoS_UDP_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[1].csv"
    DoS_UDP_1_data = pd.read_csv(path_DoS_UDP_1)
    DoS_UDP_1_data_u = DoS_UDP_1_data.sample(frac=0.05, random_state=4)
    #print(DoS_UDP_1_data.shape[0])#77204
    path_DoS_UDP_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[2].csv"
    DoS_UDP_2_data = pd.read_csv(path_DoS_UDP_2)
    DoS_UDP_2_data_u = DoS_UDP_2_data.sample(frac=0.025, random_state=4)
    #print(DoS_UDP_2_data.shape[0])#67175
    path_DoS_UDP_3 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[3].csv"
    DoS_UDP_3_data = pd.read_csv(path_DoS_UDP_3)
    DoS_UDP_3_data_u = DoS_UDP_3_data.sample(frac=0.025, random_state=4)
    #print(DoS_UDP_3_data.shape[0])#66505
    path_DoS_UDP_4 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/DoS_UDP[4].csv"
    DoS_UDP_4_data = pd.read_csv(path_DoS_UDP_4)
    DoS_UDP_4_data_u = DoS_UDP_4_data.sample(frac=0.025, random_state=4)
    #print(DoS_UDP_4_data.shape[0])#66505
    path_KL = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Keylogging.csv"
    KL_data = pd.read_csv(path_KL)
    #print(KL_data.shape[0])#2046
    path_OS = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/OS.csv"
    OS_data = pd.read_csv(path_OS)
    #print(OS_data.shape[0])#196645
    path_S = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/BoT-IoT*/csv/Service.csv"
    S_data = pd.read_csv(path_S)
    S_data_u = S_data.sample(frac=0.4, random_state=4)
    #print(S_data.shape[0])#273778

    full_data = pd.concat([DE_data,DDoS_HTTP_data,DDoS_TCP_1_data,DDoS_TCP_2_data,DDoS_TCP_3_data,
                            DDoS_UDP_1_data,DDoS_UDP_2_data,DDoS_UDP_3_data,DoS_HTTP_data,
                            DoS_TCP_1_data,DoS_TCP_2_data,DoS_TCP_3_data,DoS_TCP_4_data,
                            DoS_UDP_1_data,DoS_UDP_2_data,DoS_UDP_3_data,DoS_UDP_4_data,
                            KL_data,OS_data,S_data_u
                           ],axis=0)
    full_data_u = pd.concat([DDoS_TCP_1_data_u,DDoS_TCP_2_data_u,DDoS_TCP_3_data_u,
                            DDoS_UDP_1_data_u,DDoS_UDP_2_data_u,DDoS_UDP_3_data_u,DoS_HTTP_data,
                            DoS_TCP_1_data_u,DoS_TCP_2_data_u,DoS_TCP_3_data_u,DoS_TCP_4_data_u,
                            DoS_UDP_1_data_u,DoS_UDP_2_data_u,DoS_UDP_3_data_u,DoS_UDP_4_data_u,S_data_u
                           ],axis=0)
    full_data_u = full_data_u.sample(frac=0.3, random_state=4)
    full_data_f_u = pd.concat([DE_data,DDoS_HTTP_data,DoS_HTTP_data,KL_data,full_data_u],axis=0)
    full_data_f_u.drop([ 'Flow_ID','Timestamp', 'Dst_IP', 'Src_IP','Src_Port',  'Dst_Port',"Label","Cat",'Fwd_Seg_Size_Min'],axis=1,inplace=True)
    print(full_data_f_u.shape[0])
    binary.graph_balance(full_data, "BoT_IoT")
    binary.graph_undersample(full_data_f_u,'BoT_IoT')

    full_data_f_u['Sub_Cat'] = full_data_f_u['Sub_Cat'].replace('OS','Scan')
    full_data_f_u['Sub_Cat'] = full_data_f_u['Sub_Cat'].replace('Service','Scan')
    print(full_data_f_u.shape[0])
    end = time.time()

    print('time cost: ')
    print(end - start, 'seconds')
    # full_data.to_csv('/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/botiot.csv')


    full_data_f_u.drop(['Fwd_IAT_Std'], axis=1, inplace=True)
    #full_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(full_data_f_u)
    print('drop unnamed')
    xxx = full_data_f_u.loc[full_data_f_u.Flow_Duration == 'Flow_Duration']
    full_data_f_u = full_data_f_u.drop(xxx.index)
    print('remove Flow_Duration')
    print(full_data_f_u.shape[0])
    print('remove inf')
    full_data_f_u.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data_f_u.shape[0])
    full_data_f_u = full_data_f_u.dropna()
    print('remove nan')
    print(full_data_f_u.shape[0])
    full_data_f_u = full_data_f_u.drop_duplicates()
    print(full_data_f_u.shape[0])
    print(full_data_f_u)

    return full_data_f_u

def read_iot23():
    '''700mb => 1400000 rows 0.05'''

    #botiot
    print("Dataset:IoT-23")
    start = time.time()
    path_attack = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Attack.csv"
    attack_data = pd.read_csv(path_attack)
    attack_data_u = attack_data.sample(frac=0.04, random_state=4)
    print(attack_data.shape[0])#68671
    path_CC = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/CC.csv"
    CC_data = pd.read_csv(path_CC)
    print(CC_data.shape[0])#23984
    path_DDoS_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/DDoS[1].csv"
    DDoS_1_data = pd.read_csv(path_DDoS_1)
    DDoS_1_data_u = DDoS_1_data.sample(frac=0.015,random_state=4)
    print(DDoS_1_data.shape[0])#80906
    path_DDoS_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/DDoS[2].csv"
    DDoS_2_data = pd.read_csv(path_DDoS_2)
    DDoS_2_data_u = DDoS_2_data.sample(frac=0.015,random_state=4)
    print(DDoS_2_data.shape[0])#79386
    path_FileDownload = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/FileDownload.csv"
    FileDownload_data = pd.read_csv(path_FileDownload)
    print(FileDownload_data.shape[0])#8035
    path_HeartBeat = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/HeartBeat.csv"
    HeartBeat_data = pd.read_csv(path_HeartBeat)
    print(HeartBeat_data.shape[0])#12895
    path_Mirai = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Mirai.csv"
    Mirai_data = pd.read_csv(path_Mirai)
    print(Mirai_data.shape[0])#756
    path_Normal = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Normal.csv"
    Normal_data = pd.read_csv(path_Normal)
    Normal_data_u = Normal_data.sample(frac=0.02,random_state=4)
    print(Normal_data.shape[0])#86276
    path_Okiru_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Okiru[1].csv"
    Okiru_1_data = pd.read_csv(path_Okiru_1)
    Okiru_1_data.drop([ 'Flow_ID','Timestamp', 'Dst_IP', 'Src_IP','Src_Port',  'Dst_Port',"Label",'Fwd_Seg_Size_Min'],axis=1,inplace=True)
    Okiru_1_data = Okiru_1_data.drop_duplicates()
    path_Okiru_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Okiru[1].csv"
    Okiru_2_data = pd.read_csv(path_Okiru_2)
    Okiru_2_data.drop(['Flow_ID', 'Timestamp', 'Dst_IP', 'Src_IP', 'Src_Port', 'Dst_Port', "Label", 'Fwd_Seg_Size_Min'],axis=1, inplace=True)
    Okiru_2_data = Okiru_2_data.drop_duplicates()
    print(Okiru_2_data.shape[0])#86512
    path_PortScan_1 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/PortScan[1].csv"
    PortScan_1_data = pd.read_csv(path_PortScan_1)
    PortScan_1_data_u = PortScan_1_data.sample(frac=0.015, random_state=4)
    print(PortScan_1_data.shape[0])#88366
    path_PortScan_2 = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/PortScan[1].csv"
    PortScan_2_data = pd.read_csv(path_PortScan_2)
    PortScan_2_data_u = PortScan_2_data.sample(frac=0.015, random_state=4)
    print(PortScan_2_data.shape[0])#88366
    path_Torii = "/Users/jiangkaiwen/Desktop/Dissertation/Dataset/IoT-23*/csv/Torii.csv"
    Torii_data = pd.read_csv(path_Torii)
    print(Torii_data.shape[0])#33858

    print('iot23 read file finished')

    full_data = pd.concat([attack_data,CC_data,DDoS_1_data,DDoS_2_data,FileDownload_data,
                            HeartBeat_data,Mirai_data,Normal_data,Okiru_1_data,
                            Okiru_2_data,PortScan_1_data,PortScan_2_data,Torii_data
                           ],axis=0)
    full_data_u = pd.concat([attack_data_u,DDoS_1_data_u,DDoS_2_data_u,Normal_data_u,PortScan_1_data_u,PortScan_2_data_u,],axis=0)
    full_data_u = full_data_u.sample(frac=0.3, random_state=4)
    full_data_f_u = pd.concat([HeartBeat_data,Mirai_data,CC_data,FileDownload_data,Torii_data,full_data_u,Okiru_1_data,Okiru_2_data],axis=0)
    full_data_f_u.drop([ 'Flow_ID','Timestamp', 'Dst_IP', 'Src_IP','Src_Port',  'Dst_Port',"Label",'Fwd_Seg_Size_Min'],axis=1,inplace=True)
    full_data_f_u['Sub_Cat'] = full_data_f_u['Sub_Cat'].replace('C&C-Torii','Torii')
    print(full_data_f_u.shape[0])

    full_data_f_u['Sub_Cat'].loc[(full_data_f_u['Protocol'] == 6) & full_data_f_u['Sub_Cat'].isin(['DDoS'])] = 'DDoS_TCP'
    full_data_f_u['Sub_Cat'].loc[(full_data_f_u['Protocol'] == 17) & full_data_f_u['Sub_Cat'].isin(['DDoS'])] = 'DDoS_UDP'
    full_data_f_u['Sub_Cat'] = full_data_f_u['Sub_Cat'].replace('PartOfAHorizontalPortScan','Scan')
    full_data['Sub_Cat'].loc[(full_data['Protocol'] == 6) & full_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_TCP'
    full_data['Sub_Cat'].loc[(full_data['Protocol'] == 17) & full_data['Sub_Cat'].isin(['DDoS'])] = 'DDoS_UDP'
    full_data['Sub_Cat'] = full_data['Sub_Cat'].replace('PartOfAHorizontalPortScan','Scan')
    binary.graph_balance(full_data, "IoT_23")
    binary.graph_undersample(full_data_f_u,'IoT_23')
    print(full_data_f_u.shape[0])
    end = time.time()

    print('time cost: ')
    print(end - start, 'seconds')
    #full_data.to_csv('/Users/jiangkaiwen/PycharmProjects/conda_Dissertation/IDS-with-ML-using-Bot-IoT-dataset/iot23.csv')

    #full_data.drop(['Fwd_IAT_Std'], axis=1, inplace=True)
    #full_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(full_data_f_u)
    print('drop unnamed')

    xxx = full_data_f_u.loc[full_data_f_u.Flow_Duration == 'Flow_Duration']
    full_data_f_u = full_data_f_u.drop(xxx.index)
    print('remove Flow_Duration')
    print(full_data_f_u.shape[0])
    print('remove inf')
    full_data_f_u.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(full_data_f_u.shape[0])
    full_data_f_u = full_data_f_u.dropna()
    print('remove nan')
    print(full_data_f_u.shape[0])
    full_data_f_u = full_data_f_u.drop_duplicates()
    print(full_data_f_u.shape[0])
    print(full_data_f_u)

    print('iot23_data.Sub_Cat.value_counts() in read file')
    print(full_data_f_u.Sub_Cat.value_counts())

    return full_data_f_u

#botiot = read_botiot()
#iot23 = read_iot23()