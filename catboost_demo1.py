# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:47:04 2019

@author: 16703
"""

import numpy as np
import pandas as pd
from catboost import  CatBoostClassifier
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import time
from sklearn.model_selection import train_test_split
gc.enable()

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_table(file)
    df = reduce_mem_usage(df)
    return df

def unique_count(index_col, feature, df_data):
    if isinstance(index_col, list):
        name = "{0}_{1}_nq".format('_'.join(index_col), feature)
    else:
        name = "{0}_{1}_nq".format(index_col, feature)
    print(name)
    gp1 = df_data.groupby(index_col)[feature].nunique().reset_index().rename(
        columns={feature: name})
    df_data = pd.merge(df_data, gp1, how='left', on=[index_col])
    return df_data.fillna(0)
    #return df_data

train = pd.read_table("./input/round2_iflyad_anticheat_traindata.txt")
test = pd.read_table("./input/round2_iflyad_anticheat_testdata_feature_A.txt")
train2 = pd.read_table("./input/round1_iflyad_anticheat_traindata.txt")

#train = import_data("./input/round2_iflyad_anticheat_traindata.txt")
#test = import_data("./input/round2_iflyad_anticheat_testdata_feature_A.txt")
data = train.append(train2).reset_index(drop=True)
data = data.append(test).reset_index(drop=True)
print(train.shape,test.shape)

#data cleaning
for fea in ['model','make','lan']:
    data[fea] = data[fea].astype('str')
    data[fea] = data[fea].map(lambda x:x.upper())

    from urllib.parse import unquote

    def url_clean(x):
        x = unquote(x,'utf-8').replace('%2B',' ').replace('%20',' ').replace('%2F','/').replace('%3F','?').replace('%25','%').replace('%23','#').replace(".",' ').replace('??',' ').\
                               replace('%26',' ').replace("%3D",'=').replace('%22','').replace('_',' ').replace('+',' ').replace('-',' ').replace('__',' ').replace('  ',' ').replace(',',' ')
        
        if (x[0]=='V') & (x[-1]=='A'):
            return "VIVO {}".format(x)
        elif (x[0]=='P') & (x[-1]=='0'):
            return "OPPO {}".format(x)
        elif (len(x)==5) & (x[0]=='O'):
            return "Smartisan {}".format(x)
        elif ('AL00' in x):
            return "HW {}".format(x)
        elif 'IPHONE' in x or 'APPLE' in x:
            return 'APPLE'
        elif '华为' in x or 'HUAWEI' in x:
            return 'HUAWEI'
        elif "魅族" in x:
            return 'MEIZU'
        elif "金立" in x:
            return 'GIONEE'
        elif "三星" in x:
            return 'SAMSUNG'
        elif 'XIAOMI' in x or 'REDMI' in x:
            return 'XIAOMI'
        elif 'OPPO' in x:
            return 'OPPO'
        elif('荣耀' in x):
            return 'HONOR'
        elif('联想' in x):
            return 'LENOVO'
        else:
            return x

    data[fea] = data[fea].map(url_clean)
#data extension
data['big_model'] = data['model'].map(lambda x:x.split(' ')[0])
data['model_equal_make'] = (data['big_model']==data['make']).astype(int)
data['time'] = pd.to_datetime(data['nginxtime']*1e+6) + timedelta(hours=8)
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data.orientation[data.orientation == 90] = 0
data.orientation[data.orientation == 2] = 1 

data['size'] = (np.sqrt(data['h']**2 + data['w'] ** 2) / 2.54) / 1000
data['ratio'] = data['h'] / data['w']
data['px'] = data['ppi'] * data['size']
data['mj'] = data['h'] * data['w']

data['screen_area'] = (data['w'] * data['h']).astype('category')
data['creative_dpi'] = data['w'].astype(str) + "_" + data['h'].astype(str)

data['ip_0'] = data['ip'].map(lambda x:'.'.join(x.split('.')[:1]))
data['ip_1'] = data['ip'].map(lambda x:'.'.join(x.split('.')[0:2]))
data['ip_2'] = data['ip'].map(lambda x:'.'.join(x.split('.')[0:3]))
data['reqrealip'] = data['reqrealip'].astype(str)
data['reqrealip_0'] = data['reqrealip'].map(lambda x:'.'.join(x.split('.')[:1]))
data['reqrealip_1'] = data['reqrealip'].map(lambda x:'.'.join(x.split('.')[0:2]))
data['reqrealip_2'] = data['reqrealip'].map(lambda x:'.'.join(x.split('.')[0:3]))

data['ip_equal'] = (data['ip'] == data['reqrealip']).astype(int)
data['ip_equal0'] = (data['ip_0'] == data['reqrealip_0']).astype(int)
data['ip_equal1'] = (data['ip_1'] == data['reqrealip_1']).astype(int)

ip_feat = ['ip_0','ip_1','reqrealip_0','reqrealip_1','ip_equal','ip_equal1','ip_equal0','ip_2','reqrealip_2']

#data = unique_count('ip', 'city', data) 没用
data = unique_count('ip_0', 'city', data)
data = unique_count('reqrealip', 'city', data) 
data = unique_count('day', 'ip', data)

object_col = [i for i in data.select_dtypes(object).columns if i not in ['sid','label']]
for i in tqdm(object_col):
    lbl = LabelEncoder()
    data[i] = lbl.fit_transform(data[i].astype(str))
print(data.nunique())

g_list = ['province','province_count' ,'idfamd5','idfamd5_count','openudidmd5','openudidmd5_count','os','os_count','time' ,'macmd5','adidmd5']
cat_list = [i for i in train.columns if i not in ['sid','label','nginxtime']+g_list] + ['hour']  +  ip_feat 
for i in tqdm(cat_list):
    data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['sid'].transform('count')

feature_name = [i for i in data.columns if i not in ['sid', 'label','time','day']+g_list ]
X_train = data[0:6000000][list(set(feature_name))].reset_index(drop=True)
y = data[0:6000000]['label'].reset_index(drop=True).astype(int)
X_test = data[6000000:][list(set(feature_name))].reset_index(drop=True)
print(X_train.shape,X_test.shape)
#oof = np.zeros(X_train.shape[0])
prediction = np.zeros(X_test.shape[0])
seeds = [19941227, 2019 * 2 + 1024, 4096, 2048, 1024]
num_model_seed = 1
gc.collect() 


# oof_cat = np.zeros(X_train.shape[0])
prediction_cat=np.zeros(X_test.shape[0])
skf = StratifiedKFold(n_splits=5, random_state=seeds[0], shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index)
    train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    cbt_model = CatBoostClassifier(iterations=3000,learning_rate=0.1,max_depth=7,verbose=100,
                                   early_stopping_rounds=500,task_type='GPU',eval_metric='F1',cat_features=cat_list,max_ctr_complexity=4)
    cbt_model.fit(train_x[feature_name], train_y,eval_set=(test_x[feature_name],test_y))
    del train_x, test_x, train_y, test_y
    gc.collect()
    
    prediction_cat += cbt_model.predict_proba(X_test[feature_name])[:,1]/5

    del cbt_model
    gc.collect() 
 #   print('F1',f1_score(y, np.round(oof_cat)))    
   # oof += oof_cat / num_model_seed
   # prediction += prediction_cat / num_model_seed
'''
prediction_cat=np.zeros(X_test.shape[0])
for model_seed in range(num_model_seed):
    
    train_x, test_x, train_y, test_y =train_test_split(X_train,y,test_size=0.1, random_state=44)
    cbt_model = CatBoostClassifier(iterations=3000,learning_rate=0.1,max_depth=7,verbose=100,
                                   early_stopping_rounds=500,task_type='GPU',eval_metric='F1',
                                   cat_features=cat_list,max_ctr_complexity=2,gpu_cat_features_storage = 'CpuPinnedMemory')
    cbt_model.fit(train_x[feature_name], train_y,eval_set=(test_x[feature_name],test_y))
    del train_x, test_x, train_y, test_y
    gc.collect()    
    prediction_cat += cbt_model.predict_proba(X_test[feature_name])[:,1]
    print('finish')
'''
#prediction += prediction_cat / num_model_seed
# write to csv
submit = test[['sid']]
#submit['label'] = (prediction_cat>=0.501).astype(int)
submit['label'] = prediction_cat
#print(submit['label'].value_counts())
submit.to_csv("submission_no_addition_A.csv",index=False)
