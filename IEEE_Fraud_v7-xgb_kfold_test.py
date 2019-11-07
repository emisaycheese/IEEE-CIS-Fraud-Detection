#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:54 2019

@author: ruiqianyang
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc

#%%
####data from r
train_r=pd.read_csv('train_r.csv')
test_r=pd.read_csv('test_r.csv')

#del X,test_x
X=train_r.drop(train_r.columns[0],axis=1)
test_x=test_r.drop(test_r.columns[0],axis=1)

label = pd.read_pickle('train_Basic.pkl')
y = label['isFraud']

train=pd.read_pickle('train_Basic.pkl')
test=pd.read_pickle('test_Basic.pkl')
y=y.reset_index(drop=False)
train=train.reset_index(drop=False)
test=test.reset_index(drop=False)

X['TransactionID']=train['TransactionID']
test_x['TransactionID']=test['TransactionID']

X['month']=train['month']
test_x['month']=test['month']

import datetime
startDT = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
train['TransactionDT1']=train['TransactionDT'].apply(lambda x:(startDT+datetime.timedelta(seconds=x)))
X['DT_M'] = ((train['TransactionDT1'].dt.year-2017)*12 + train['TransactionDT1'].dt.month).astype(np.int8)
X['DT_M']


for df in [X, test_x]:
    for col_1 in df.columns:
        if df[col_1].dtype == 'object':
            print(col_1)
            le = LabelEncoder()
            le.fit(list(train_r[col_1].astype(str).values) + 
                   list(test_r[col_1].astype(str).values))
            df[col_1] = le.transform(list(df[col_1].astype(str).values))
    df.fillna(-999,inplace=True)
    df[np.isinf(df.values)==True] = 999
    
#%%
 ###minify data
X_12=X[X['month']==12]
y_12=label[label['month']==12]['isFraud']

#%%

X_train, X_test, y_train, y_test = train_test_split(X_xgb.drop(columns=['TransactionID','DT_M']), 
                             y_xgb, test_size=0.33, random_state=47, shuffle=False)


import xgboost as xgb
import time
start=time.time()
clf = xgb.XGBClassifier(
            n_jobs=4,
            n_estimators=500,
            max_depth=20,
            learning_rate=0.035,        #0.04,
            subsample=0.7,
            colsample_bytree= 0.7463058454739352,
            tree_method='gpu_hist',  # THE MAGICAL PARAMETER
#            gamma=1,
 #              min_child_weight=1,
#                max_delta_step=,
            random_state=2019,
#            max_delta_step= 4.762550705407337,
            reg_alpha= 0.39871702770778467,
            reg_lamdba=0.24309304355829786)
#             min_child_samples= 170)

# {'target': 0.9175568588299362, 'params': 
#  {'colsample_bytree': 0.9285591089934457, 'gamma': 1.0, 'learning_rate': 0.026245848007381067,
#   'max_delta_step': 4.762550705407337, 'max_depth': 19.868689007456137, 'min_child_weight': 2.1576248698291813, 
#   'reg_alpha': 0.2866349425015622, 'reg_lambda': 0.40562804952710174, 'subsample': 0.9301734656335603}}

clf.fit(X_train, y_train,eval_set = [(X_test, y_test)], eval_metric="auc",
                early_stopping_rounds = 10)
print(time.time()-start)

roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])






