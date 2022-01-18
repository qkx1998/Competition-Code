'''
https://challenge.datacastle.cn/v3/cmptDetail.html?id=604
'''
import pandas as pd
import numpy as np
import os 
import random 
import gc
from sklearn.metrics import *
from tqdm import tqdm 
import warnings 
warnings.filterwarnings('ignore')

def seed_everything(seed=2020):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

data_path_1 = os.getcwd().replace('code', '主表数据\\')
data_path_2 = os.getcwd().replace('code', '其他数据表\\')

seed_everything(seed=1998)

train = pd.read_csv(data_path_1 + 'x_train.csv')
test = pd.read_csv(data_path_1 + 'x_test.csv')
train_label = pd.read_csv(data_path_1 + 'y_train.csv')
train = train.merge(train_label, on='id', how='left')

test.rename(columns={'c2':'a2','c3':'a3'}, inplace=True)
df = pd.concat([train, test]).reset_index(drop=True)
print(df.shape)
df['rank'] = [i for i in range(df.shape[0])]

# 客户信息表
d = pd.read_csv(data_path_2 + 'd.csv')
d.fillna(5, inplace=True)
df = df.merge(d, on='core_cust_id', how='left')

# 客户信息表
d = pd.read_csv(data_path_2 + 'd.csv')
d.fillna(5, inplace=True)
df = df.merge(d, on='core_cust_id', how='left')

# 产品表信息
g = pd.read_csv(data_path_2 + 'g.csv')
h = pd.read_csv(data_path_2 + 'h.csv')
i = pd.read_csv(data_path_2 + 'i.csv')
j = pd.read_csv(data_path_2 + 'j.csv')
k = pd.read_csv(data_path_2 + 'k.csv')
l = pd.read_csv(data_path_2 + 'l.csv')

for data in [g, h, i, j, k, l]:
    com_pid = set(data['prod_code'].unique()).intersection(set(df['prod_code'].unique()))
    tmp = data[data['prod_code'].isin(com_pid)].describe().T
    
    useful_cols = list(tmp[(tmp['std'] != 0) & (tmp['std'] != np.nan)].index)
    useful_cols = [c for c in useful_cols if c not in ['g9', 'h8', 'i9', 'j13', 'k11', 'l7']]
    print(useful_cols)
    
    if len(useful_cols) > 0:
        df = df.merge(data[['prod_code']+useful_cols], on='prod_code', how='left')
        
    #产品交易流水
n = pd.read_csv(data_path_2 + 'n.csv')
o = pd.read_csv(data_path_2 + 'o.csv')
q = pd.read_csv(data_path_2 + 'q.csv')
p = pd.read_csv(data_path_2 + 'p.csv')

n['date'] = n['n11'].apply(lambda x: str(x)[:6])
o['date'] = o['o12'].apply(lambda x: str(x)[:6])
q['date'] = q['q10'].apply(lambda x: str(x)[:6])
p['date'] = p['p12'].apply(lambda x: str(x)[:6])

n['apply_amt'] = n['n7'].apply(lambda x: str(x).replace(',','')).astype('float') 
o['apply_amt'] = o['o7'].apply(lambda x: str(x).replace(',','')).astype('float') 
q['apply_amt'] = q['q7'].apply(lambda x: str(x).replace(',','')).astype('float') 
p['apply_amt'] = p['p7'].apply(lambda x: str(x).replace(',','')).astype('float') 

dict_ = {'2021-07-01':'202106', '2021-08-01':'202107', '2021-09-01':'202108', '2021-10-01':'202109'}

for k, data in enumerate([n, q, o, p]):
    print('p == ', k+1)
    dfs = []
    
    for month in sorted(df['a3'].unique()):
        print(month)
        tmp_df = df[df['a3'] == month]
        
        stat_1 = data[data['date'] < dict_[month]].groupby('core_cust_id')['prod_code'].count().reset_index()
        stat_1.columns = ['core_cust_id',f'uid_cnt_in_p{k+1}_all_2m_ago']
        stat_1[f'pid_cnt_grp_uid_in_p{k+1}_all_2m_ago'] = data[data['date'] < dict_[month]].groupby('core_cust_id')['prod_code'].agg('nunique').values
        stat_1[f'apply_amt_mean_grp_uid_in_p{k+1}_all_2m_ago'] = data[data['date'] < dict_[month]].groupby('core_cust_id')['apply_amt'].agg('mean').values
        stat_1[f'apply_amt_sum_grp_uid_in_p{k+1}_all_2m_ago'] = data[data['date'] < dict_[month]].groupby('core_cust_id')['apply_amt'].agg('sum').values
       
        stat_2 = data[data['date'] == dict_[month]].groupby('core_cust_id')['prod_code'].count().reset_index()
        stat_2.columns = ['core_cust_id',f'uid_cnt_in_p{k+1}_1m_ago']
        stat_2[f'pid_cnt_grp_uid_in_p{k+1}_1m_ago'] = data[data['date'] == dict_[month]].groupby('core_cust_id')['prod_code'].agg('nunique').values
        stat_2[f'apply_amt_mean_grp_uid_in_p{k+1}_1m_ago'] = data[data['date'] == dict_[month]].groupby('core_cust_id')['apply_amt'].agg('mean').values
        stat_2[f'apply_amt_sum_grp_uid_in_p{k+1}_1m_ago'] = data[data['date'] == dict_[month]].groupby('core_cust_id')['apply_amt'].agg('sum').values
        
        tmp_df = tmp_df.merge(stat_1, on='core_cust_id', how='left')
        tmp_df = tmp_df.merge(stat_2, on='core_cust_id', how='left')

        dfs.append(tmp_df)

    df = pd.concat(dfs).reset_index(drop=True)
    
    # APP点击行为表
r = pd.read_csv(data_path_2 + 'r.csv')
r['date'] = r['r5'].apply(lambda x: x[:7])

dict_ = {'2021-07-01':'2021-06', '2021-08-01':'2021-07', '2021-09-01':'2021-08', '2021-10-01':'2021-09'}
dfs = []

for month in sorted(df['a3'].unique()):
    print(month)
    tmp_df = df[df['a3'] == month]

    stat_1 = r[(r['date'] <= dict_[month])].groupby('core_cust_id')['prod_code'].count().reset_index()
    stat_1.columns = ['core_cust_id','uid_cnt_in_click_action_all_1m_ago']
    stat_1['pid_cnt_grp_uid_in_click_action_all_1m_ago'] = r[(r['date'] <= dict_[month])].groupby('core_cust_id')['prod_code'].agg('nunique').values
    
    tmp_df = tmp_df.merge(stat_1, on='core_cust_id', how='left')

    dfs.append(tmp_df)

df = pd.concat(dfs).reset_index(drop=True)

# 账户交易流水表
s = pd.read_csv(data_path_2 + 's.csv')
s['date'] = s['s7'].apply(lambda x: x[:7])
s['s4'] = s['s4'].apply(lambda x: str(x).replace(',','')).astype('float') 

dict_ = {'2021-07-01':'2021-06', '2021-08-01':'2021-07', '2021-09-01':'2021-08', '2021-10-01':'2021-09'}
dfs = []

for month in sorted(df['a3'].unique()):
    print(month)
    tmp_df = df[df['a3'] == month]
    
    stat_1 = s[s['date'] == dict_[month]].groupby('s3')['s1'].count().reset_index()
    stat_1.columns = ['core_cust_id','borrow_cnt_1m_ago']
    stat_1['borrow_mean_1m_ago'] = s[s['date'] == dict_[month]].groupby('s3')['s4'].agg('mean').values
    stat_1['borrow_sum_1m_ago'] = s[s['date'] == dict_[month]].groupby('s3')['s4'].agg('sum').values
    
    stat_2 = s[s['date'] == dict_[month]].groupby('s6')['s1'].count().reset_index()
    stat_2.columns = ['core_cust_id','loan_cnt_1m_ago']
    stat_2['loan_mean_1m_ago'] = s[s['date'] == dict_[month]].groupby('s6')['s4'].agg('mean').values
    stat_2['loan_sum_1m_ago'] = s[s['date'] == dict_[month]].groupby('s6')['s4'].agg('sum').values

    tmp_df = tmp_df.merge(stat_1, on='core_cust_id', how='left')
    tmp_df = tmp_df.merge(stat_2, on='core_cust_id', how='left')
    
    dfs.append(tmp_df)

df = pd.concat(dfs).reset_index(drop=True)

# 资产信息表
f = pd.read_csv(data_path_2 + 'f.csv')

f['date'] = f['f22'].apply(lambda x: str(x)[:6])
used_cols = [c for c in f.columns if c not in ['core_cust_id','f1','f22','date']]
for c in used_cols:
    f[c] = f[c].apply(lambda x: str(x).replace(',','')).astype('float')
    
dict_ = {'2021-07-01':'202106', '2021-08-01':'202107', '2021-09-01':'202108', '2021-10-01':'202109'}
dfs = []

for month in sorted(df['a3'].unique()):
    print(month)

    tmp_df = df[df['a3'] == month]
    
    stat_1 = f[f['date'] == dict_[month]].groupby('core_cust_id')[used_cols].mean().reset_index()
    stat_1.columns = ['core_cust_id'] + [f'{c}_mean_1m_ago' for c in used_cols]
    
    stat_2 = f[f['date'] < dict_[month]].groupby('core_cust_id')[used_cols].mean().reset_index()
    stat_2.columns = ['core_cust_id'] + [f'{c}_mean_all_2m_ago' for c in used_cols]
 
    tmp_df = tmp_df.merge(stat_1, on='core_cust_id', how='left')
    tmp_df = tmp_df.merge(stat_2, on='core_cust_id', how='left')

    dfs.append(tmp_df)

df = pd.concat(dfs).reset_index(drop=True)

df = df.sort_values('rank').reset_index(drop=True)

features = [c for c in df.columns if c != 'y']
for c in features:
    df[c].fillna(-1, inplace=True)
    
drop_cols = [c for c in features if df[c].dtype != 'object' and df[c].std() == 0]
df.drop(drop_cols, axis=1, inplace=True)
print(drop_cols)

drop_fea = ['id','core_cust_id','a3','y','prod_code','rank','j7'] 
      
feature= [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)

df_0 = df[(df['a3'] < '2021-10-01') & (df['y'] == 0)].drop_duplicates(feature)
df_1 = df[(df['a3'] < '2021-10-01') & (df['y'] == 1)]
df_ = pd.concat([df_0, df_1]).sample(frac=1, random_state=2).reset_index(drop=True)

X_train = df_[df_["a3"] < '2021-09-01'][feature].reset_index(drop=True)
y_train = df_[df_["a3"] < '2021-09-01']["y"]
X_valid = df_[df_["a3"] == '2021-09-01'][feature].reset_index(drop=True)
y_valid = df_[df_["a3"] == '2021-09-01']["y"]
X_test = df[df["a3"] == '2021-10-01'][feature].reset_index(drop=True)

del df_, df_0, df_1; gc.collect()
print(len(X_train), len(y_train), len(X_valid), len(y_valid))

from lightgbm import LGBMClassifier

clf = LGBMClassifier(num_leaves=128,
                     n_estimators=10000,
                     learning_rate=0.01,
                     verbose=-1,
                     metric='auc',
                     lambda_l1=0.1,
                     lambda_l2=0.1, 
                     min_child_weight=30,
                     n_jobs=20)

clf.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,verbose=20)

gc.collect()

oof_prob = clf.predict_proba(X_valid)[:, 1]  


def find_best_threshold(y_valid, oof_prob):
    best_f2 = 0
    
    for th in tqdm([i/1000 for i in range(50, 200)]):
        oof_prob_copy = oof_prob.copy()
        oof_prob_copy[oof_prob_copy >= th] = 1
        oof_prob_copy[oof_prob_copy < th] = 0

        recall = recall_score(y_valid, oof_prob_copy)
        precision = precision_score(y_valid, oof_prob_copy)
        f2 = 5*recall*precision / (4*precision+recall)
        
        if f2 > best_f2:
            best_th = th
            best_f2 = f2
  
        gc.collect()
        
    return best_th, best_f2

best_th, best_f2 = find_best_threshold(y_valid, oof_prob)
print(best_th, best_f2)

X = pd.concat([X_train, X_valid]).reset_index(drop=True)
y = pd.concat([y_train, y_valid]).reset_index(drop=True)

clf1 = LGBMClassifier(num_leaves=128,
                     n_estimators= int(clf.best_iteration_*1.2),
                     learning_rate=0.01,
                     verbose=-1,
                     metric='auc',
                     lambda_l1=0.1,
                     lambda_l2=0.1, 
                     min_child_weight=30,
                     n_jobs=20)

clf1.fit(X, y)

gc.collect
y_pre = clf1.predict_proba(X_test)[:, 1]  

res = test[['id']]
res['y'] = y_pre
res.loc[res['y'] >= best_th, 'y'] = 1
res.loc[res['y'] < best_th, 'y'] = 0
res.to_csv('submission.csv',index = False) 
