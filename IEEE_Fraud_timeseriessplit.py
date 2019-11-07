#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:15:26 2019

@author: ruiqianyang
"""
#
#files=['test_identity.csv', 
#         'test_transaction.csv',
#         'train_identity.csv',
#         'train_transaction.csv']
#
#import pandas as pd
#def load_data(file):
#    return pd.read_csv(file,index_col='TransactionID')
##%%
#import multiprocessing
#import time
#with multiprocessing.Pool() as pool:
#    test_id,test_trans,train_id,train_trans=pool.map(load_data,files)
#
##%%
#train=train_trans.merge(train_id,how='left',left_index=True,right_index=True)
#test=test_trans.merge(test_id,how='left',left_index=True,right_index=True)
#
#
#import datetime
#startDT = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
#startDT
##add month,day of month,day of week,hour of day
#train['TransactionDT1']=train['TransactionDT'].apply(lambda x:(startDT+datetime.timedelta(seconds=x)))
#train['month']=train['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%m')))
#train['day']=train['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%d')))
#train['dayofweek']=train['TransactionDT1'].dt.date.apply(lambda x: x.weekday())
#train['hour']=train['TransactionDT1'].apply(lambda x: int(x.strftime('%H')))
#train['weekofyear']=train['TransactionDT1'].apply(lambda x: int(x.strftime('%U')))
#
##engineer Month to start from 12 and monotonous increasing;
##seems redcuce score...
##train['month'] = (train['TransactionDT1'].dt.year-2017)*12 + train['TransactionDT1'].dt.month 
##train.iloc[299000:299100]
#
#test['TransactionDT1']=test['TransactionDT'].apply(lambda x:(startDT+datetime.timedelta(seconds=x)))
#test['month']=test['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%m')))
#test['day']=test['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%d')))
#test['dayofweek']=test['TransactionDT1'].dt.date.apply(lambda x: x.weekday())
#test['hour']=test['TransactionDT1'].apply(lambda x: int(x.strftime('%H')))
#test['weekofyear']=test['TransactionDT1'].apply(lambda x: int(x.strftime('%U')))
#
##engineer Month to start from 12 and monotonous increasing;
##test['month'] = (test['TransactionDT1'].dt.year-2017)*12 + test['TransactionDT1'].dt.month 
#
#test.head(3)
##%%
#train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
#test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
##plot_numerical1('TransactionAmt_decimal')
#
##creat new feature: Number of NaN's
#train['nulls']=train.isnull().sum(axis=1)
#test['nulls']=test.isnull().sum(axis=1)
#for df in [train, test, X]:
#    ########################### Device info
#    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

#to be continued for filling missing data ...........................................
################################################################################
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc

train = pd.read_pickle('train_Basic.pkl')
test = pd.read_pickle('test_Basic.pkl')

y = train['isFraud']
X = pd.DataFrame()
#%%
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 
          'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google',
          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 
          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 
          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 
          'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

#%%
def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() 
                                                                    < 200].index), 'device_name'] = "Others"
    #dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe


train = id_split(train)
test= id_split(test)


#%%



#%%
col=['TransactionDT','ProductCD','P_emaildomain','R_emaildomain','nulls','month','hour',
     'TransactionAmt', 'TransactionAmt_decimal','D5','D2','D8','D9','D11','D15','C1','C4','C8','C10','C13','C2',
     'V256','V13','card1','card2','card3','card4', 'card5', 'card6','addr1','M4','M5','M6','M1',
     'M2','M3','M7','M8','M9','DeviceType',
     'DeviceInfo', 'addr2','D6','D13','C5','C9','D7','C14',
    'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09','id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 
     'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33','id_36', 'id_37', 'id_38', 
   'V145',
    'V3','V4','V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V17',
    'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
    'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
    'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
    'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
    'V143', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
    'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
    'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
    'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
    'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
    'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V257', 'V258', 'V259', 'V261',
    'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
    'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
    'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 
    'V323','V324', 'V326','V329','V331','V332', 'V333', 'V335', 'V336',
     'isFraud']


#'device_name','device_version','OS_id_30','version_id_30','browser_id_31','version_id_31','screen_width','screen_height',
#     'id_34','id_23',

# 'V145',
#     'V3','V4','V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V17',
#     'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
#     'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
#     'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
#     'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
#     'V143', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
#     'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
#     'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
#     'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
#     'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
#     'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V257', 'V258', 'V259', 'V261',
#     'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
#     'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
#     'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 
#     'V323','V324', 'V326','V329','V331','V332', 'V333', 'V335', 'V336',


#'V48','V53','V96','V338'
#month is important
# Save to pickle 
New_DatasetName = "Basic"
#train[col].to_pickle("train_{}.pkl".format(New_DatasetName))
#test[col[:-1]].to_pickle("test_{}.pkl".format(New_DatasetName))
col.pop()
X[col] = train[col]

X['M4_count'] =X['M4'].map(pd.concat([train['M4'], test['M4']], ignore_index=True).value_counts(dropna=False))
X['M6_count'] =X['M6'].map(pd.concat([train['M6'], test['M6']], ignore_index=True).value_counts(dropna=False))
X['card1_count'] = X['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
X['card6_count'] =X['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

# X['card2_count'] =X['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))
# X['card3_count'] =X['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))
# X['card4_count'] =X['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))
# X['card5_count'] =X['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))
# X['id_36_count'] =X['id_36'].map(pd.concat([train['id_36'], test['id_36']], ignore_index=True).value_counts(dropna=False))


# # Encoding - count encoding for both train and test
# for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
#     train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
#     test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

# # Encoding - count encoding separately for train and test
# for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
#     train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
#     test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


for feature in ['id_30','id_06']:
    X[feature + '_count'] = X[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

X['DeviceType_count'] =X['DeviceType'].map(pd.concat([train['DeviceType'], test['DeviceType']], ignore_index=True).value_counts(dropna=False))
#X['DeviceInfo_count'] =X['DeviceInfo'].map(pd.concat([train['DeviceInfo'], test['DeviceInfo']], ignore_index=True).value_counts(dropna=False))
X['ProductCD_count'] =X['ProductCD'].map(pd.concat([train['ProductCD'], test['ProductCD']], ignore_index=True).value_counts(dropna=False))
X['P_emaildomain_count'] =X['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))
#X['R_emaildomain_count'] =X['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))
X['addr1_count'] = X['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

#X['id_02_count'] = X['id_02'].map(pd.concatD.value_counts(dropna=False))
#X['id_17_count'] = X['id_17'].map(pd.concat([train['id_17'], test['id_17']], ignore_index=True).value_counts(dropna=False))
###X['id_02'].fillna(0, inplace=True)

# Risk mapping transformation
card4_dic = {'american express':287,'discover':773,'mastercard':343,'visa':348}
X['card4_dic']=X['card4'].map(card4_dic)
#train=train.replace({'card4':card4_dic})
#test=test.replace({'card4':card4_dic})
ProductCD_dic = {'C':117,'H':48,'R':38,'S':59,'W':20}
X['ProductCD_dic']=X['ProductCD'].map(ProductCD_dic)

#New feature:id risk mapping:id_12,id_15,id_16,id_27,id_28,id_29
#df['id_12'] = df['id_12'].fillna('Missing').str.lower()
# id_12_dic = {'NotFound':85,'Found':60}
# X['id_12_dic']=X['id_12'].map(id_12_dic)

#df['id_15'] = df['id_15'].fillna('Missing').str.lower()
# id_15_dic = {'Found':105,'Unknown':92,'New':49}
# X['id_15_dic']=X['id_15'].map(id_15_dic)

# df['id_16'] = df['id_16'].fillna('Missing').str.lower()
# id_16_dic = {'NotFound':48,'Found':107}
# X['id_16_dic']=X['id_16'].map(id_16_dic)

# df['id_28'] = df['id_28'].fillna('Missing').str.lower()
# id_28_dic = {'Found':103,'New':52}
# X['id_28_dic']=X['id_28'].map(id_28_dic)
#New feature:addr2 risk mapping
#addr2_dic={46:100,51:100,10:100,65:54,96:14,60:9,32:7,87:2}
#X['addr2_dic']=X['addr2'].map(addr2_dic)
#38:67,73:20,54:33,68:10,29:9,

card6_dic = {'charge card':0,'credit':668,'debit':243,'debit or credit':0}
X['card6_dic']=X['card6'].map(card6_dic)
#train=train.replace({'card6':card6_dic})
#test=test.replace({'card6':card6_dic})

M1_dic={'F':0,'T':2}
X['M1_dic']=X['M1'].map(M1_dic)

M4_dic={'M0':4,'M1':3,'M2':13}
X['M4_dic']=X['M4'].map(M4_dic)

M5_dic={'F':2,'T':3}
X['M5_dic']=X['M5'].map(M5_dic)

M6_dic={'F':4,'T':3}
X['M6_dic']=X['M6'].map(M6_dic)

#DeviceInfo has many category levels
#DeviceType_dic={'desktop':3,'mobile':5}
#X['DeviceType_dic']=X['DeviceType'].map(DeviceType_dic)

P_emaildomain_dic={'protonmail.com':40,'mail.com':19,'outlook.es':13,'aim.com':12,
                    'outlook.com':9,'hotmail.es':7,'live.com.mx':5,'hotmail.com':5,'gmail.com':4}
X['P_emaildomain_dic']=X['P_emaildomain'].map(P_emaildomain_dic)


R_emaildomain_dic={'protonmail.com':95,'mail.com':38,'netzero.net':22,'outlook.com':17,
                    'outlook.es':13,'icloud.com':13,'gmail.com':12,'hotmail.com':8,
                    'earthlink.net':8,'earthlink.net':7,'hotmail.es':7,'live.com.mx':6,
                   'yahoo.com':5,'live.com':5}
X['R_emaildomain_dic']=X['R_emaildomain'].map(R_emaildomain_dic)

#New feature:C2 vs TransactionDT
def func_C2_Tran(a,b):
        if a<400:
            return 344
        elif a>=400 and a<=651 and b<=4000000:
            return 3846
        elif a>=400 and a<=651 and b>10000000:
            return 10000
        else:
            return 1082
X['C2_TransactionDT']=X.apply(lambda x:func_C2_Tran(x['C2'],x['TransactionDT']),axis=1)


###sep 06,2019 add fill na:
X['D9'] = np.where(X['D9'].isna(),0,1)

########################### Reset values for "noise" card1
# i_cols = ['card1']

# for col in i_cols: 
#     valid_card = pd.concat([train[[col]], test[[col]]])
#     valid_card = valid_card[col].value_counts()
#     valid_card = valid_card[valid_card>2]
#     valid_card = list(valid_card.index)

#     X[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
#     X[col] = np.where(X[col].isin(valid_card), X[col], np.nan)
# #     test_x[col]  = np.where(test_x[col].isin(train[col]), test_x[col], np.nan)
# #     test_x[col]  = np.where(test_x[col].isin(valid_card), test_x[col], np.nan)
  
    
########################## M columns (except M4)
#All these columns are binary encoded 1/0
#We can have some features from it
# i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

# X['M_sum'] = X[i_cols].sum(axis=1).astype(np.int8)
# X['M_na'] = X[i_cols].isna().sum(axis=1).astype(np.int8)


# Check if the Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with a model we are telling to trust or not to these values   
X['TransactionAmt_check'] = np.where(X['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
####################################Interaction term
X['uid'] = X['card1'].astype(str)+'_'+X['card2'].astype(str)
test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)

X['uid2'] = X['uid'].astype(str)+'_'+X['card3'].astype(str)+'_'+X['card5'].astype(str)
test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)

X['uid3'] = X['uid2'].astype(str)+'_'+X['addr1'].astype(str)+'_'+X['addr2'].astype(str)
test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

# For our model current TransactionAmt is a noise
# https://www.kaggle.com/kyakovlev/ieee-check-noise
# (even if features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations

i_cols = ['card1','card2','card3','card5']
#'uid','uid2','uid3'
for col2 in i_cols:
    for agg_type in ['mean','std']:
        new_col_name = col2+'_TransactionAmt_'+agg_type
        temp_df = pd.concat([train[[col2, 'TransactionAmt']], test[[col2,'TransactionAmt']]])
        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
        temp_df = temp_df.groupby([col2])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col2])
        temp_df = temp_df[new_col_name].to_dict()   
    
        X[new_col_name] = X[col2].map(temp_df)
        
# i_cols2 = ['uid','uid2','uid3']

# for col3 in i_cols2:
#     for agg_type in ['mean','std']:
#         new_col_name = col3+'_TransactionAmt_'+agg_type
#         temp_df = pd.concat([X[[col3, 'TransactionAmt']], test[[col3,'TransactionAmt']]])
#         #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
#         temp_df = temp_df.groupby([col3])['TransactionAmt'].agg([agg_type]).reset_index().rename(
#                                                 columns={agg_type: new_col_name})
        
#         temp_df.index = list(temp_df[col3])
#         temp_df = temp_df[new_col_name].to_dict()   
    
#         X[new_col_name] = X[col3].map(temp_df)        
           
# Small "hack" to transform distribution 
# (doesn't affect auc much, but I like it more)
# please see how distribution transformation can boost your score 
# (not our case but related)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html

#X['TransactionAmt'] = np.log1p(X['TransactionAmt'])


# ########################### Device info
# X['DeviceInfo'] = X['DeviceInfo'].fillna('unknown_device').str.lower()
# X['DeviceInfo_device'] = X['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
# X['DeviceInfo_version'] = X['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

########################### Device info 2
#X['id_30'] = X['id_30'].fillna('unknown_device').str.lower()
#X['id_30_device'] = X['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
#X['id_30_version'] = X['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

########################### Browser
#X['id_31'] = X['id_31'].fillna('unknown_device').str.lower()
#X['id_31_device'] = X['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

###########
for feature in ['uid','uid2','uid3']:
        # Count encoded separately for train and test,
    X[feature + '_count'] = X[feature].map(X[feature].value_counts(dropna=False))

    
X=X.drop(columns=['ProductCD','P_emaildomain','R_emaildomain','M4','M5','M6','M1', 'M2','M3','M7','M8','M9','card6', 'DeviceType',
                  'card4','id_02','id_19','id_20','id_17','C2','TransactionDT','addr2','D6','D13','C5','C9','D7','C14',
                  'id_12', 'id_15', 'id_30', 'id_31', 'id_33', 'id_36', 'id_37', 'id_38','uid','uid2','uid3'
                 ])
#%%
X['addr1'].fillna(0, inplace=True)
#X['addr2_dic'].fillna(3, inplace=True)
#X['addr2'].fillna(96, inplace=True)
#X['D5'].fillna(150, inplace=True)
X['D2'].fillna(10, inplace=True)
X['D11'].fillna(10, inplace=True)
X['D15'].fillna(376, inplace=True)

#X['V1'].fillna(X['V1'].mode()[0], inplace=True)          
#X['V2'].fillna(3, inplace=True)

X['V13'].fillna(X['V13'].mode()[0], inplace=True)
 #fillna reduce score
#X['V263'].fillna(X['V263'].mode()[0], inplace=True)
#X['V256'].fillna(X['V256'].mode()[0], inplace=True)

X['card2'].fillna(502, inplace=True)
###X['card3'].fillna(150, inplace=True)
X['card4_dic'].fillna(X['card4_dic'].mode()[0], inplace=True)
                   
X['M4_dic'].fillna(2, inplace=True)
X['M5_dic'].fillna(3, inplace=True)  #24:18:78
X['M6_dic'].fillna(13, inplace=True)  #24:18:78

#X['cross'].fillna(2.6, inplace=True)

#X['id_19'].fillna(444, inplace=True)
#X['id_20'].fillna(266, inplace=True)
#X['id_17'].fillna(133, inplace=True)
X['P_emaildomain_dic'].fillna(0, inplace=True)
X['R_emaildomain_dic'].fillna(0, inplace=True)

#sep 04,2019 add fill na:
#X['V145'].fillna(0, inplace=True)  #V145 has more than 80% missing
X['V19'].fillna(0, inplace=True)
X['V36'].fillna(0, inplace=True)
X['V64'].fillna(0, inplace=True)
X['V70'].fillna(0, inplace=True)
X['V80'].fillna(0, inplace=True)
X['V94'].fillna(0, inplace=True)
# X['V143'].fillna(0, inplace=True)
# X['V150'].fillna(0, inplace=True)
# X['V152'].fillna(0, inplace=True)
# X['V158'].fillna(0, inplace=True)##
# X['V163'].fillna(0, inplace=True)
# X['V165'].fillna(0, inplace=True)
# X['V177'].fillna(0, inplace=True)
# X['V204'].fillna(0, inplace=True)
# X['V207'].fillna(0, inplace=True)
# X['V209'].fillna(0, inplace=True) ##
# X['V221'].fillna(0, inplace=True)
# X['V222'].fillna(0, inplace=True)
# X['V266'].fillna(0, inplace=True)
# X['V267'].fillna(0, inplace=True)
# X['V274'].fillna(0, inplace=True)
# X['V275'].fillna(0, inplace=True)
X['V279'].fillna(0, inplace=True)
X['V283'].fillna(0, inplace=True)
#X['id_30_count'].fillna(0, inplace=True)


#%%
#train test split
#X_train, X_test, y_train, y_test = train_test_split(X, 
#                             y, test_size=0.33, random_state=47, shuffle=False)
#

#%%
#train with lgb
#params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':0.7,
#        'importance_type':'split', 'learning_rate':0.05, 'max_depth':25,
#        'min_child_samples':20, 'min_child_weight':0.00298, 'min_split_gain':0.0,
#        'n_estimators':300, 'n_jobs':-1, 'num_leaves':2**8, 'silent':False, 'subsample':0.7,
#          'reg_alpha':0.38999, 'reg_lambda':2.0,'subsample_for_bin':200000, 'subsample_freq':1,
#        'objective': 'binary', "bagging_seed": 8, 'metric': 'auc', 'random_state': 47}

#                
#clf = lgb.LGBMClassifier(**params)
#clf.fit(X_train, y_train)
#print("basic score is "+str(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])))


#%%
# clf.fit(upsampled.drop(columns=['isFraud']),upsampled['isFraud'])
# X_test=X_test.drop(columns=['TransactionDT'])
# roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

#%%
#def addfeature(var):
#    
#    y = train['isFraud']
#    X = pd.DataFrame()
#    
#    col=['TransactionDT','ProductCD','P_emaildomain','R_emaildomain','nulls','month','hour',
#         'TransactionAmt', 'TransactionAmt_decimal','D5','D2','D8','D9','D11','D15','C1','C4','C8','C10','C13','C2',
#         'V256','V13','card1','card2','card3','card4', 'card5', 'card6','addr1','M4','M5','M6','M1','DeviceType',
#         'DeviceInfo','addr2','D6','D13','C5','C9','D7','C14','V145',
#         'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
#    'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
#    'id_36', 'id_37', 'id_38'
#        ]+[var]
#    
#    X[col] = train[col]
#    
#    print('preprocessing including '+ var)
#    
#    X['M4_count'] =X['M4'].map(pd.concat([train['M4'], test['M4']], ignore_index=True).value_counts(dropna=False))
#    X['M6_count'] =X['M6'].map(pd.concat([train['M6'], test['M6']], ignore_index=True).value_counts(dropna=False))
#    X['card1_count'] = X['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
#    X['card6_count'] =X['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))
#    
#    X['DeviceType_count'] =X['DeviceType'].map(pd.concat([train['DeviceType'], test['DeviceType']], ignore_index=True).value_counts(dropna=False))
#    #X['DeviceInfo_count'] =X['DeviceInfo'].map(pd.concat([train['DeviceInfo'], test['DeviceInfo']], ignore_index=True).value_counts(dropna=False))
#    X['ProductCD_count'] =X['ProductCD'].map(pd.concat([train['ProductCD'], test['ProductCD']], ignore_index=True).value_counts(dropna=False))
#    X['P_emaildomain_count'] =X['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))
#    #X['R_emaildomain_count'] =X['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))
#    X['addr1_count'] = X['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
#       
#    # Risk mapping transformation
#    card4_dic = {'american express':287,'discover':773,'mastercard':343,'visa':348}
#    X['card4_dic']=X['card4'].map(card4_dic)
#    #train=train.replace({'card4':card4_dic})
#    #test=test.replace({'card4':card4_dic})
#    ProductCD_dic = {'C':117,'H':48,'R':38,'S':59,'W':20}
#    X['ProductCD_dic']=X['ProductCD'].map(ProductCD_dic)
#    
#    
#    card6_dic = {'charge card':0,'credit':668,'debit':243,'debit or credit':0}
#    X['card6_dic']=X['card6'].map(card6_dic)
#    #train=train.replace({'card6':card6_dic})
#    #test=test.replace({'card6':card6_dic})
#    
#    M1_dic={'F':0,'T':2}
#    X['M1_dic']=X['M1'].map(M1_dic)
#    
#    M4_dic={'M0':4,'M1':3,'M2':13}
#    X['M4_dic']=X['M4'].map(M4_dic)
#    
#    M5_dic={'F':2,'T':3}
#    X['M5_dic']=X['M5'].map(M5_dic)
#    
#    M6_dic={'F':4,'T':3}
#    X['M6_dic']=X['M6'].map(M6_dic)
#      
#    P_emaildomain_dic={'protonmail.com':40,'mail.com':19,'outlook.es':13,'aim.com':12,
#                        'outlook.com':9,'hotmail.es':7,'live.com.mx':5,'hotmail.com':5,'gmail.com':4}
#    X['P_emaildomain_dic']=X['P_emaildomain'].map(P_emaildomain_dic)
#    
#    
#    R_emaildomain_dic={'protonmail.com':95,'mail.com':38,'netzero.net':22,'outlook.com':17,
#                        'outlook.es':13,'icloud.com':13,'gmail.com':12,'hotmail.com':8,
#                        'earthlink.net':8,'earthlink.net':7,'hotmail.es':7,'live.com.mx':6,
#                       'yahoo.com':5,'live.com':5}
#    X['R_emaildomain_dic']=X['R_emaildomain'].map(R_emaildomain_dic)
#    
#    #New feature:C2 vs TransactionDT
##    def func_C2_Tran(a,b):
##            if a<400:
##                return 344
##            elif a>=400 and a<=651 and b<=4000000:
##                return 3846
##            elif a>=400 and a<=651 and b>10000000:
##                return 10000
##            else:
##                return 1082
##    X['C2_TransactionDT']=X.apply(lambda x:func_C2_Tran(x['C2'],x['TransactionDT']),axis=1)
##   
#
#    X[var + '_count'] = X[var].map(pd.concat([train[var], test[var]], ignore_index=True).value_counts(dropna=False))
#     
#    X=X.drop(columns=['ProductCD','P_emaildomain','R_emaildomain','M4','M5','M6','M1','card6', 'DeviceType','DeviceInfo',
#                  'card4','id_02','id_19','id_20','id_17','C2','TransactionDT','addr2','D6','D13','C5','C9','D7','C14',
#                  'id_12', 'id_15', 'id_30', 'id_31', 'id_33', 'id_36', 'id_37', 'id_38'
#                 ])


#X['addr1'].fillna(0, inplace=True)
##X['addr2_dic'].fillna(3, inplace=True)
##X['addr2'].fillna(96, inplace=True)
##X['D5'].fillna(150, inplace=True)
#X['D2'].fillna(10, inplace=True)
#X['D11'].fillna(10, inplace=True)
#X['D15'].fillna(376, inplace=True)
#
##X['V1'].fillna(X['V1'].mode()[0], inplace=True)          
##X['V2'].fillna(3, inplace=True)
#
#X['V13'].fillna(X['V13'].mode()[0], inplace=True)
##X['V145'].fillna(X['V145'].mode()[0], inplace=True) #fillna reduce score
##X['V263'].fillna(X['V263'].mode()[0], inplace=True)
##X['V256'].fillna(X['V256'].mode()[0], inplace=True)
#
#X['card2'].fillna(502, inplace=True)
####X['card3'].fillna(150, inplace=True)
#X['card4_dic'].fillna(X['card4_dic'].mode()[0], inplace=True)
#                   
#X['M4_dic'].fillna(2, inplace=True)
#X['M5_dic'].fillna(3, inplace=True)  #24:18:78
#X['M6_dic'].fillna(13, inplace=True)  #24:18:78
#
##X['cross'].fillna(2.6, inplace=True)
#
##X['id_19'].fillna(444, inplace=True)
##X['id_20'].fillna(266, inplace=True)
##X['id_17'].fillna(133, inplace=True)
#X['P_emaildomain_dic'].fillna(0, inplace=True)
#X['R_emaildomain_dic'].fillna(0, inplace=True)


#    print('start training with '+var)
#    X_train, X_test, y_train, y_test = train_test_split(X, 
#                             y, test_size=0.33, random_state=47, shuffle=False)
#
#    clf = lgb.LGBMClassifier(**params)
#    clf.fit(X_train, y_train)
#    score=roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
#    print(score)
#    return score
#    
#res=[]
##ls=['V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V17',
##     'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
##     'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
##     'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
##     'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
##     'V143', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
##     'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
##     'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
##     'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
##     'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
##     'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V257', 'V258', 'V259', 'V261',
##     'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
##     'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
##     'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322']
#ls=[ 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09','id_11', 'id_13', 'id_14',  'id_17', 'id_19', 'id_20',  'id_32']
#for i in ls:
#    res.append(addfeature(i))
#%%
#ls1=['V323','V324', 'V326','V329','V331','V332', 'V333', 'V335', 'V336']
#for i in ls1:
#    res.append(addfeature(i))
#  
#%%
#def nmaxelement(ls,n):
#    result=[]
#    tmp=sorted(ls)[-n:]
#    for i in tmp:
#        result.append(ls.index(i))
#    return result
#nmaxelement(res,10)
    
#%%    
#from sklearn.model_selection import GridSearchCV
#lg = lgb.LGBMClassifier(silent=False)
#param_dist = {"max_depth": [25],             #<=0 means no limit
#              "learning_rate" : [0.05],
#              "num_leaves": [256],
#              "n_estimators": [300],
#              'min_child_samples':[20],
#              'bagging_fraction': [0.42],
#              'feature_fraction': [0.38],
#               'min_child_weight': [0.00298],
#               'min_data_in_leaf': [106],
#               'reg_alpha': [0.38999],
#               'reg_lambda': [2.0]  
#
##               "max_depth": [5,25,-1,50],
##               "learning_rate" : [0.05,0.00688,0.01],
##               "num_leaves": [300, 491,382,2**8],
##               "n_estimators": [300,800],
##               'min_child_samples':[20],
##               'bagging_fraction': [0.42,0.9],
##               'feature_fraction': [0.38,0.9],
##               'min_child_weight': [0.00298,0.03454],
##               'min_data_in_leaf': [20,106],
##               'reg_alpha': [1.0,0.38999],
##               'reg_lambda': [2.0,0.6485],
#                      
##                     'colsample_bytree': 0.7,
##                     'subsample_freq':1,
##                     'subsample':0.7,
##                     'max_bin':255,
##                     'verbose':-1,
##                     'seed': SEED,
##                     'early_stopping_rounds':100, 
#             }
#grid_search = GridSearchCV(lg, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
#grid_search.fit(X_train,y_train)
#grid_search.best_estimator_
#%%
# LGB_BO.max['params']

# {'bagging_fraction': 0.8999999999997461,
#  'feature_fraction': 0.8999999999999121,
#  'max_depth': 50.0,
#  'min_child_weight': 0.0029805017044362268,
#  'min_data_in_leaf': 20.0,
#  'num_leaves': 381.85354295079446,
#  'reg_alpha': 1.0,
#  'reg_lambda': 2.0}

# params = {'num_leaves': 491,
#           'min_child_weight': 0.03454472573214212,
#           'feature_fraction': 0.3797454081646243,
#           'bagging_fraction': 0.4181193142567742,
#           'min_data_in_leaf': 106,
#           'objective': 'binary',
#           'max_depth': -1,
#           'learning_rate': 0.006883242363721497,
#           "boosting_type": "gbdt",
#           "bagging_seed": 11,
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 0.3899927210061127,
#           'reg_lambda': 0.6485237330340494,
#           'random_state': 47
#          }


#%%
#preprocessing in test data
#for prediction on test dataset
#for prediction on test dataset
test_x=pd.DataFrame()
test_x[col] = test[col]

test_x['M4_count'] =test_x['M4'].map(pd.concat([train['M4'], test['M4']], ignore_index=True).value_counts(dropna=False))
test_x['M6_count'] =test_x['M6'].map(pd.concat([train['M6'], test['M6']], ignore_index=True).value_counts(dropna=False))
test_x['card1_count'] = test_x['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
test_x['card6_count'] = test_x['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

test_x['addr1_count'] = test_x['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
test_x['ProductCD_count'] =test_x['ProductCD'].map(pd.concat([train['ProductCD'], test['ProductCD']], ignore_index=True).value_counts(dropna=False))
test_x['P_emaildomain_count'] =test_x['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))
#test_x['R_emaildomain_count'] =test_x['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))
test_x['DeviceType_count'] =test_x['DeviceType'].map(pd.concat([train['DeviceType'], test['DeviceType']], ignore_index=True).value_counts(dropna=False))

# Risk mapping transformation
test_x['card4_dic']=test_x['card4'].map(card4_dic)
test_x['card6_dic']=test_x['card6'].map(card6_dic)
test_x['ProductCD_dic']=test_x['ProductCD'].map(ProductCD_dic)
#test_x['cross'] =  test_x.apply(lambda x:func(x['card4'],x['ProductCD']),axis=1)
M1_dic={'F':0,'T':2}
test_x['M1_dic']=test_x['M1'].map(M1_dic)
M4_dic={'M0':4,'M1':3,'M2':13}
test_x['M4_dic']=test_x['M4'].map(M4_dic)
M5_dic={'F':2,'T':3}
test_x['M5_dic']=test_x['M5'].map(M5_dic)
M6_dic={'F':4,'T':3}
test_x['M6_dic']=test_x['M6'].map(M6_dic)
test_x['R_emaildomain_dic']=test_x['R_emaildomain'].map(R_emaildomain_dic)
test_x['P_emaildomain_dic']=test_x['P_emaildomain'].map(P_emaildomain_dic)
test_x['C2_TransactionDT']=test_x.apply(lambda x:func_C2_Tran(x['C2'],x['TransactionDT']),axis=1)

for feature in [ 'id_30','id_06']:
       # Count encoded separately for train and test 'id_12', 'id_15','id_31', 'id_33', 'id_36', 'id_37', 'id_38'
    test_x[feature + '_count'] = test_x[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


###sep 06,2019 add fill na:
test_x['D9'] = np.where(test_x['D9'].isna(),0,1)

########################### Reset values for "noise" card1
# i_cols = ['card1']

# for col in i_cols: 
#     valid_card = pd.concat([train[[col]], test[[col]]])
#     valid_card = valid_card[col].value_counts()
#     valid_card = valid_card[valid_card>2]
#     valid_card = list(valid_card.index)

#     test_x[col]  = np.where(test_x[col].isin(train[col]), test_x[col], np.nan)
#     test_x[col]  = np.where(test_x[col].isin(valid_card), test_x[col], np.nan)
  
    
########################## M columns (except M4)
#All these columns are binary encoded 1/0
#We can have some features from it
# i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

# test_x['M_sum'] = test_x[i_cols].sum(axis=1).astype(np.int8)
# test_x['M_na'] = test_x[i_cols].isna().sum(axis=1).astype(np.int8)


####################################Interaction term

test_x['uid'] = test['uid']
test_x['uid2'] = test['uid2']
test_x['uid3'] = test['uid3']
# Check if the Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with a model we are telling to trust or not to these values   

test_x['TransactionAmt_check']  = np.where(test_x['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)

# For our model current TransactionAmt is a noise
# https://www.kaggle.com/kyakovlev/ieee-check-noise
# (even if features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations

i_cols = ['card1','card2','card3','card5']

for col2 in i_cols:
    for agg_type in ['mean','std']:
        new_col_name = col2+'_TransactionAmt_'+agg_type
        temp_df = pd.concat([train[[col2, 'TransactionAmt']], test[[col2,'TransactionAmt']]])
        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
        temp_df = temp_df.groupby([col2])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col2])
        temp_df = temp_df[new_col_name].to_dict()   
    
        test_x[new_col_name]  = test_x[col2].map(temp_df)
        
i_cols2 = ['uid','uid2','uid3']

for col3 in i_cols2:
    for agg_type in ['mean','std']:
        new_col_name = col3+'_TransactionAmt_'+agg_type
        temp_df = pd.concat([X[[col3, 'TransactionAmt']], test[[col3,'TransactionAmt']]])
        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
        temp_df = temp_df.groupby([col3])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col3])
        temp_df = temp_df[new_col_name].to_dict()   
    
        test_x[new_col_name] = test_x[col3].map(temp_df) 
                   

# Small "hack" to transform distribution 
# (doesn't affect auc much, but I like it more)
# please see how distribution transformation can boost your score 
# (not our case but related)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html

#test_x['TransactionAmt'] = np.log1p(test_x['TransactionAmt']) 

########################### Device info
test_x['DeviceInfo'] = test_x['DeviceInfo'].fillna('unknown_device').str.lower()
test_x['DeviceInfo_device'] = test_x['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
test_x['DeviceInfo_version'] = test_x['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

########################### Device info 2
test_x['id_30'] = test_x['id_30'].fillna('unknown_device').str.lower()
test_x['id_30_device'] = test_x['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
test_x['id_30_version'] = test_x['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

########################### Browser
test_x['id_31'] = test_x['id_31'].fillna('unknown_device').str.lower()
test_x['id_31_device'] = test_x['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))


###########
for feature in ['uid','uid2','uid3']:
        # Count encoded separately for train and test,
    test_x[feature + '_count'] = test_x[feature].map(X[feature].value_counts(dropna=False))
   
    
test_x=test_x.drop(columns=['ProductCD','P_emaildomain','R_emaildomain','M4','M5','M6','M1', 'M2','M3','M7','M8','M9','card6', 'DeviceType',
                  'card4','id_02','id_19','id_20','id_17','C2','TransactionDT','addr2','D6','D13','C5','C9','D7','C14',
                  'id_12', 'id_15', 'id_30', 'id_31', 'id_33', 'id_36', 'id_37', 'id_38','uid','uid2','uid3'])

#%%
#filling missing in test data
test_x['addr1'].fillna(0, inplace=True)
##test_x['D5'].fillna(150, inplace=True)
test_x['D2'].fillna(10, inplace=True)
test_x['D11'].fillna(10, inplace=True)
test_x['D15'].fillna(376, inplace=True)
#test_x['V1'].fillna(test_x['V1'].mode()[0], inplace=True)
#test_x['V2'].fillna(3, inplace=True)
test_x['V13'].fillna(test_x['V13'].mode()[0], inplace=True)
##test_x['V256'].fillna(test_x['V256'].mode()[0], inplace=True)

test_x['card2'].fillna(502, inplace=True)
test_x['card4_dic'].fillna(test_x['card4_dic'].mode()[0], inplace=True)

test_x['M4_dic'].fillna(2, inplace=True)
test_x['M5_dic'].fillna(3, inplace=True)  #24:18:78
test_x['M6_dic'].fillna(13, inplace=True)  #24:18:78
test_x['P_emaildomain_dic'].fillna(0, inplace=True)
test_x['R_emaildomain_dic'].fillna(0, inplace=True)

# test_x['V145'].fillna(0, inplace=True)
test_x['V19'].fillna(0, inplace=True)
test_x['V36'].fillna(0, inplace=True)
test_x['V64'].fillna(0, inplace=True)
test_x['V70'].fillna(0, inplace=True)
test_x['V80'].fillna(0, inplace=True)
test_x['V94'].fillna(0, inplace=True)
# test_x['V143'].fillna(0, inplace=True)
# test_x['V150'].fillna(0, inplace=True)
# test_x['V152'].fillna(0, inplace=True)
# test_x['V158'].fillna(0, inplace=True)##
# test_x['V163'].fillna(0, inplace=True)
# test_x['V165'].fillna(0, inplace=True)
# test_x['V177'].fillna(0, inplace=True)
# test_x['V204'].fillna(0, inplace=True)
# test_x['V207'].fillna(0, inplace=True)
# test_x['V209'].fillna(0, inplace=True) ##
# test_x['V221'].fillna(0, inplace=True)
# test_x['V222'].fillna(0, inplace=True)
# test_x['V266'].fillna(0, inplace=True)
# test_x['V267'].fillna(0, inplace=True)
# test_x['V274'].fillna(0, inplace=True)
# test_x['V275'].fillna(0, inplace=True)
test_x['V279'].fillna(0, inplace=True)
test_x['V283'].fillna(0, inplace=True)
#test_x['id_30_count'].fillna(0, inplace=True)
#%%
###KFOLD lightGBM
#from sklearn.model_selection import KFold
#
#import numpy as np
#import gc
#
#
##params for kfold CV 
#params={'objective':'binary',
#        'boosting_type':'gbdt',
#        'metric':'auc',
#        'n_jobs':-1,
#        'learning_rate':0.01,
#        'num_leaves': 2**8,
#        'max_depth':-1,
#        'tree_learner':'serial',
#        'colsample_bytree': 0.7,
#        'subsample_freq':1,
#        'subsample':0.7,
#        'n_estimators':800,
#        'max_bin':255,
#        'verbose':-1,
#        'early_stopping_rounds':100 }  
#
#
#def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=5):
#    folds = KFold(n_splits=NFOLDS, shuffle=False, random_state=42)
#
#    X,y = tr_df[features_columns], tr_df[target]    
#    P,P_y = tt_df[features_columns], tt_df[target]  
#
#    tt_df = tt_df[['TransactionID',target]]    
#    predictions = np.zeros(len(tt_df))
#    
#    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#        print('Fold:',fold_)
#        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
#        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
#            
#        print(len(tr_x),len(vl_x))
#        tr_data = lgb.Dataset(tr_x, label=tr_y)
#
#        if LOCAL_TEST:
#            vl_data = lgb.Dataset(P, label=P_y) 
#        else:
#            vl_data = lgb.Dataset(vl_x, label=vl_y)  
#
#        estimator = lgb.train(
#            params,
#            tr_data,
#            valid_sets = [tr_data, vl_data],
#            verbose_eval = 200,
#        )   
#        
#        pp_p = estimator.predict(P)
#        predictions += pp_p/NFOLDS     #consider weighting predictions??
#
#        if LOCAL_TEST:
#            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), 
#                                       columns=['Value','Feature'])
#            print(feature_imp)
#        
#        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
#        gc.collect()
#    
#    print('Best iteration in KFOLD CV is: '+estimator.best_iteration)
#    tt_df['prediction'] = predictions
#    
#    return tt_df
#
############################ Model Train
#LOCAL_TEST = False
#TARGET = 'isFraud'
#
##assign train_df and test_df:
#if LOCAL_TEST:
#    
#    train_df=pd.concat([X,y],axis=1)
#    test_df = train_df[train_df['month']==train_df['month'].max()].reset_index(drop=True) #last month data used as test
#    train_df = train_df[train_df['month']<(train_df['month'].max()-1)].reset_index(drop=True)
#
#else:
#    train_df=pd.concat([X,y],axis=1)
#    test_df=test_x.iloc[:]
#    test_df['isFraud']=0
#    
#features_columns=X.columns
#
#if LOCAL_TEST:
#    params['learning_rate'] = 0.01
#    params['n_estimators'] = 20000
#    params['early_stopping_rounds'] = 100
#    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, params)
#    print(roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
#else:
#    params['learning_rate'] = 0.05
#    params['n_estimators'] = 1800
#    params['early_stopping_rounds'] = 100    
#    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, params, NFOLDS=10)
#    #should NFOLDS=5 or 10?
#%%
test_predictions

#%%
###time series split
from sklearn.model_selection import TimeSeriesSplit
from time import time
folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns


#params for time series split and lgb train
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }


training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, 
                    early_stopping_rounds=500)
    
    pp_p = clf.predict(test_x)
    #ensemble
    predictions_tssplit += pp_p/5     #n_splits=5, consider weighting predictions??
    
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    #consider weighting aucs??
    aucs.append(clf.best_score['valid_1']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)

test_x['prediction'] = predictions_tssplit
#%%

feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


#%%
#time series split best iteration
best_iter = clf.best_iteration
clf.best_score
#%%
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, y)
#%%
#important to reset index
test_x=test_x.reset_index(drop=False)

#No cross validation prediction:
#test_x['isFraud']=clf.predict_proba(test_x.drop(columns=['TransactionID']))[:,1]

#for kfold
test_x['isFraud']=test_predictions['prediction']

#for time series split
#test_x['isFraud'] = clf.predict_proba(test_x.drop(columns=['TransactionID']))[:, 1]



#%%
output=test_x[['TransactionID','isFraud']]
#%%

#test.shape
output.to_csv("output_submission_ts.csv",index=False)


