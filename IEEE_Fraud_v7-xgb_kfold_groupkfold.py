#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:52 2019

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
#############################XGBOOST kfold##################################
import xgboost as xgb
from sklearn.model_selection import KFold,GroupKFold
import time
start=time.time()
       
def make_predictions(tr_df, tt_df, features_columns, target, 
                      NFOLDS=10):
    
    #folds = GroupKFold(n_splits=NFOLDS)  #GroupKFold by XX
    folds = KFold(n_splits=NFOLDS, shuffle=False, random_state=42)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    #split_groups = tr_df['DT_M']      #GroupKFold by XX

    tt_df = tt_df[['TransactionID',target]] 
    #tt_df = tt_df[target]   
    predictions = np.zeros(len(tt_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
   # for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)): #GroupKFold by XX
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
#         tr_data = xgb.DMatrix(tr_x, label=tr_y)
#         vl_data = xgb.DMatrix(vl_x, label=vl_y)
#         P_D = xgb.DMatrix(P)
        
 #       print('train month is '+str(tr_x['month'].unique()))
 #       print('test month is '+str(vl_x['month'].unique()))
#        tr_data = lgb.Dataset(tr_x, label=tr_y)

#         if LOCAL_TEST:
#             vl_data = lgb.Dataset(P, label=P_y) 
#         else:
#             vl_data = lgb.Dataset(vl_x, label=vl_y)  

#         estimator = xgb.XGBClassifier.train(
#             xgb_params,
#             tr_data,
#             evals = [(tr_data, 'train'),(vl_data, 'valid')], 
#             verbose_eval = 50
#         ) 

        clf = xgb.XGBClassifier(
             n_jobs=4,
            n_estimators=500,
            max_depth=20,
            learning_rate=0.035,
            subsample=0.7,
            colsample_bytree=0.7463058454739352,
            #tree_method='gpu_hist',  # THE MAGICAL PARAMETER
            reg_alpha=0.39871702770778467,
            reg_lamdba=0.24309304355829786,
             random_state=2019,
#                gamma=0.6665437467229817,
#                min_child_weight=,
#                max_delta_step=,
#           min_child_samples= 170
        )
    
        clf.fit(tr_x, tr_y,eval_set = [(tr_x, tr_y),(vl_x, vl_y)], eval_metric="auc",
                early_stopping_rounds = 10)
        print(time.time()-start)
        print('ROC accuracy: {}'.format(roc_auc_score(vl_y, clf.predict_proba(vl_x)[:, 1])))
        #pp_p+= clf.predict_proba(P)[:,1] 
     
        pp_p = clf.predict_proba(P)[:,1]
        
        print(pp_p)
        tt_df['fold '+str(fold_)]=pp_p
        predictions += pp_p/NFOLDS     #consider weighting predictions??
       # print('weight is '+str(w[fold_]))
       # predictions += pp_p*w[fold_]
        
        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), 
                                       columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y
        gc.collect()
    
    tt_df['prediction'] = predictions
    
    return tt_df   

########################### Model Train
LOCAL_TEST = False
TARGET = 'isFraud'

#assign train_df and test_df:
train_df=X.iloc[:]
train_df['isFraud']=y['isFraud']
test_df=test_x.iloc[:]
test_df['isFraud']=0

#!!!if transactionID is index: 
train_df=train_df.reset_index(drop=False)
test_df=test_df.reset_index(drop=False)
    
#make sure 'DT_M' and'TransactionID' are not in feature list
features_columns=list(set(X)-{'DT_M','TransactionID'})     
print(len(features_columns))
print(len(X.columns))

test_predictions = make_predictions(train_df, test_df, features_columns, TARGET,NFOLDS=10)
####End XGB KFOLD    
#%%
#XGB GROUP kfold##################################
import xgboost as xgb
from sklearn.model_selection import KFold,GroupKFold
import time
start=time.time()
       
def make_predictions(tr_df, tt_df, features_columns, target, 
                      NFOLDS=7):
    
    folds = GroupKFold(n_splits=NFOLDS)  #GroupKFold by DT_M

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    split_groups = tr_df['DT_M']      #GroupKFold by XX

    tt_df = tt_df[['TransactionID',target]] 
    predictions = np.zeros(len(tt_df))
    
 
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)): #GroupKFold by XX
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
#         tr_data = xgb.DMatrix(tr_x, label=tr_y)
#         vl_data = xgb.DMatrix(vl_x, label=vl_y)
#         P_D = xgb.DMatrix(P)
        
        print('train month is '+str(tr_x['month'].unique()))
        print('test month is '+str(vl_x['month'].unique()))
#        tr_data = lgb.Dataset(tr_x, label=tr_y)

#         if LOCAL_TEST:
#             vl_data = lgb.Dataset(P, label=P_y) 
#         else:
#             vl_data = lgb.Dataset(vl_x, label=vl_y)  

        clf = xgb.XGBClassifier(
             n_jobs=4,
            n_estimators=500,
            max_depth=20,
            learning_rate=0.035,
            subsample=0.7,
            colsample_bytree=0.7463058454739352,
            #tree_method='gpu_hist',  # THE MAGICAL PARAMETER
            reg_alpha=0.39871702770778467,
            reg_lamdba=0.24309304355829786,
             random_state=2019,
#                gamma=0.6665437467229817,
#                min_child_weight=,
#                max_delta_step=,
#           min_child_samples= 170
        )
    
        clf.fit(tr_x, tr_y,eval_set = [(tr_x, tr_y),(vl_x, vl_y)], eval_metric="auc",
                early_stopping_rounds = 10)
        print(time.time()-start)
        print('ROC accuracy: {}'.format(roc_auc_score(vl_y, clf.predict_proba(vl_x)[:, 1])))
        #pp_p+= clf.predict_proba(P)[:,1] 
     
        pp_p = clf.predict_proba(P)[:,1]
        
        print(pp_p)
        tt_df['fold '+str(fold_)]=pp_p
        predictions += pp_p/NFOLDS     #consider weighting predictions??
       # print('weight is '+str(w[fold_]))
       # predictions += pp_p*w[fold_]
        
        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), 
                                       columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y
        gc.collect()
    
    tt_df['prediction'] = predictions
    
    return tt_df   

########################### Model Train
LOCAL_TEST = False
TARGET = 'isFraud'

#assign train_df and test_df:
train_df=X.iloc[:]
train_df['isFraud']=y['isFraud']
test_df=test_x.iloc[:]
test_df['isFraud']=0
train_df=train_df.reset_index(drop=False)
test_df=test_df.reset_index(drop=False)
    
#make sure 'DT_M' and'TransactionID' are not in feature list
features_columns=list(set(X)-{'DT_M','TransactionID'})     
print(len(features_columns))
print(len(X.columns))

test_predictions = make_predictions(train_df, test_df, features_columns, TARGET,NFOLDS=7)
####End GROUPKFOLD    
    
    
    
#%%    
 #important to reset index
test_x=test_x.reset_index(drop=False)

test_predictions.to_csv("XGB_Kfold_predictions_every_fold.csv",index=False)

#for kfold/groupkfold/stratifiedkfold
test_x['isFraud']=test_predictions['prediction']
   
    
output=test_x[['TransactionID','isFraud']]
    
output.to_csv("output_submission_XGB_Kfold.csv",index=False)
  
    
    
    
    
    
    
    
    
    
    
    
    
