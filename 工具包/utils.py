import os
import random
import numpy as np

# 固定随机种子
def seed_everything(seed=2020):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 缩减数据内存
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# N折目标编码
def n_fold_target_encoding(train_df,test_df,label='label',n=5,enc_list=[],functions=['mean']):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=1998)
    for f in tqdm(enc_list):
        for func in functions:
            train_df[f + f'_target_enc_{func}'] = 0
            test_df[f + f'_target_enc_{func}'] = 0
            for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df[label])):
                trn_x = train_df[[f, label]].iloc[trn_idx].reset_index(drop=True)
                val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
                enc_df = trn_x.groupby(f, as_index=False)[label].agg({f + f'_target_enc_{func}': func})
                val_x = val_x.merge(enc_df, on=f, how='left')
                test_x = test_df[[f]].merge(enc_df, on=f, how='left')
                val_x[f + f'_target_enc_{func}'] = val_x[f + f'_target_enc_{func}'].fillna(train_df[label].agg(func))
                test_x[f + f'_target_enc_{func}'] = test_x[f + f'_target_enc_{func}'].fillna(train_df[label].agg(func))
                train_df.loc[val_idx, f + f'_target_enc_{func}'] = val_x[f + f'_target_enc_{func}'].values
                test_df[f + f'_target_enc_{func}'] += test_x[f + f'_target_enc_{func}'].values / skf.n_splits
    return train_df,test_df
  
  

  
