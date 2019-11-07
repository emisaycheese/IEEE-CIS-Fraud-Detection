#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:22:29 2019

@author: ruiqianyang
"""
files=['test_identity.csv', 
         'test_transaction.csv',
         'train_identity.csv',
         'train_transaction.csv']

import pandas as pd
def load_data(file):
    return pd.read_csv(file,index_col='TransactionID')
#%%
import multiprocessing
import time
with multiprocessing.Pool() as pool:
    test_id,test_trans,train_id,train_trans=pool.map(load_data,files)

#%%
train=train_trans.merge(train_id,how='left',left_index=True,right_index=True)
del(train_trans)
del(train_id)
test=test_trans.merge(test_id,how='left',left_index=True,right_index=True)
del test_id, test_trans

#train=pd.merge(train_trans,train_id,on='TransactionID',how='left')
#test=pd.merge(test_trans,test_id,on='TransactionID',how='left')

#%%
import datetime
startDT = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
startDT
#add month,day of month,day of week,hour of day
train['TransactionDT1']=train['TransactionDT'].apply(lambda x:(startDT+datetime.timedelta(seconds=x)))
train['month']=train['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%m')))
train['day']=train['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%d')))
train['dayofweek']=train['TransactionDT1'].dt.date.apply(lambda x: x.weekday())
train['hour']=train['TransactionDT1'].apply(lambda x: int(x.strftime('%H')))
train['weekofyear']=train['TransactionDT1'].apply(lambda x: int(x.strftime('%U')))

#engineer Month to start from 12 and monotonous increasing;
#seems redcuce score...
#train['month'] = (train['TransactionDT1'].dt.year-2017)*12 + train['TransactionDT1'].dt.month 

test['TransactionDT1']=test['TransactionDT'].apply(lambda x:(startDT+datetime.timedelta(seconds=x)))
test['month']=test['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%m')))
test['day']=test['TransactionDT1'].dt.date.apply(lambda x: int(x.strftime('%d')))
test['dayofweek']=test['TransactionDT1'].dt.date.apply(lambda x: x.weekday())
test['hour']=test['TransactionDT1'].apply(lambda x: int(x.strftime('%H')))
test['weekofyear']=test['TransactionDT1'].apply(lambda x: int(x.strftime('%U')))

#engineer Month to start from 12 and monotonous increasing;
#test['month'] = (test['TransactionDT1'].dt.year-2017)*12 + test['TransactionDT1'].dt.month 

#test.head(3)
#%%
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
#plot_numerical1('TransactionAmt_decimal')

#creat new feature: Number of NaN's
train['nulls']=train.isnull().sum(axis=1)
test['nulls']=test.isnull().sum(axis=1)


################################################################################
#%%
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
y = train['isFraud']
X = pd.DataFrame()

col=['TransactionDT','ProductCD','P_emaildomain','R_emaildomain','nulls','month','hour',
     'TransactionAmt', 'TransactionAmt_decimal','D5','D2','D8','D9','D11','D15','C1','C4','C8','C10','C13','C2',
     'V256','V13','card1','card2','card3','card4', 'card5', 'card6','addr1','M4','M5','M6','M1','DeviceType',
     'DeviceInfo', 'addr2','D6','D13','C5','C9','D7','C14','V145',
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
    'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
    'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
    'id_36', 'id_37', 'id_38','isFraud']

# Save to pickle 
#New_DatasetName = "Basic"
#train[col].to_pickle("train_{}.pkl".format(New_DatasetName))
#test[col[:-1]].to_pickle("test_{}.pkl".format(New_DatasetName))
col.pop()
X[col] = train[col]

X['M4_count'] =X['M4'].map(pd.concat([train['M4'], test['M4']], ignore_index=True).value_counts(dropna=False))
X['M6_count'] =X['M6'].map(pd.concat([train['M6'], test['M6']], ignore_index=True).value_counts(dropna=False))
X['card1_count'] = X['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
X['card6_count'] =X['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

X['DeviceType_count'] =X['DeviceType'].map(pd.concat([train['DeviceType'], test['DeviceType']], ignore_index=True).value_counts(dropna=False))
#X['DeviceInfo_count'] =X['DeviceInfo'].map(pd.concat([train['DeviceInfo'], test['DeviceInfo']], ignore_index=True).value_counts(dropna=False))
X['ProductCD_count'] =X['ProductCD'].map(pd.concat([train['ProductCD'], test['ProductCD']], ignore_index=True).value_counts(dropna=False))
X['P_emaildomain_count'] =X['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))
#X['R_emaildomain_count'] =X['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))
X['addr1_count'] = X['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))


# Risk mapping transformation
card4_dic = {'american express':287,'discover':773,'mastercard':343,'visa':348}
X['card4_dic']=X['card4'].map(card4_dic)
ProductCD_dic = {'C':117,'H':48,'R':38,'S':59,'W':20}
X['ProductCD_dic']=X['ProductCD'].map(ProductCD_dic)


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



for feature in ['id_30','id_02', 'id_03',  'id_06', 'id_09','id_11', 'id_14', 
                'id_17', 'id_19', 'id_20', 'id_32']:
        # Count encoded separately for train and test,
    X[feature + '_count'] = X[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


X=X.drop(columns=['ProductCD','P_emaildomain','R_emaildomain','M4','M5','M6','M1','card6', 'DeviceType','DeviceInfo','card4','id_02','id_19','id_20','id_17','C2','TransactionDT','addr2','D6','D13','C5','C9','D7','C14','id_12', 'id_15', 'id_30', 'id_31', 'id_33', 'id_36', 'id_37', 'id_38'
                 ])              
                 
#%%
print('start fillna')
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
#X['V145'].fillna(X['V145'].mode()[0], inplace=True) #fillna reduce score
#X['V263'].fillna(X['V263'].mode()[0], inplace=True)
#X['V256'].fillna(X['V256'].mode()[0], inplace=True)

X['card2'].fillna(502, inplace=True)
###X['card3'].fillna(150, inplace=True)
X['card4_dic'].fillna(X['card4_dic'].mode()[0], inplace=True)
                   
X['M4_dic'].fillna(2, inplace=True)
X['M5_dic'].fillna(3, inplace=True)  #24:18:78
X['M6_dic'].fillna(13, inplace=True)  #24:18:78

X['P_emaildomain_dic'].fillna(0, inplace=True)
X['R_emaildomain_dic'].fillna(0, inplace=True)

####start GridSearchCV
#%%
#train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=47, shuffle=False)


#%%  
#add grid of paramters  
print('start GridSearchCV')
from sklearn.model_selection import GridSearchCV
lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [25],             #<=0 means no limit
              "learning_rate" : [0.05],
              "num_leaves": [256],
              "n_estimators": [300,1800,3000],
              'min_child_samples':[20],
              'bagging_fraction': [0.42],
              'feature_fraction': [0.38],
               'min_child_weight': [0.00298],
               'min_data_in_leaf': [106],
               'reg_alpha': [0.38999],
               'reg_lambda': [2.0]  
             }

#candidates for param
#               "max_depth": [5,25,-1,50],
#               "learning_rate" : [0.05,0.00688,0.01],
#               "num_leaves": [300, 491,382,2**8],
#               "n_estimators": [300,800],
#               'min_child_samples':[20],
#               'bagging_fraction': [0.42,0.9],
#               'feature_fraction': [0.38,0.9],
#               'min_child_weight': [0.00298,0.03454],
#               'min_data_in_leaf': [20,106],
#               'reg_alpha': [1.0,0.38999],
#               'reg_lambda': [2.0,0.6485],
                      
#               'colsample_bytree': 0.7,
#               'subsample_freq':1,
#               'subsample':0.7,
#               'max_bin':255,
#               'verbose':-1,
#               'seed': SEED,
#               'early_stopping_rounds':100, 
             
grid_search = GridSearchCV(lg, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
grid_search.fit(X_train,y_train)
print(grid_search.best_estimator_)
