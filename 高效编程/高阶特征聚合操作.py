import pandas as pd

df = pd.read_csv('recruit_folder.csv')

'''
RECRUIT_ID	PERSON_ID	LABEL
	825081	  6256839	  0
	772899	  5413605	  0
	795668	  5219796	  0
	769754	  5700693	  0
	773645	  6208645	  1
'''

# 直接对多个统计函数进行计算
stat_list = ['mean', 'max', 'min']
grp_df = df.groupby('RECRUIT_ID')['LABEL'].agg(stat_list).reset_index()
grp_df.columns = ['RECRUIT_ID'] + [f'LABEL_{i}_grp_RECRUIT_ID' for i in stat_list]8

# 使用itertools同时对三个列表进行遍历，减少代码量
import itertools

cat_cols = ['RECRUIT_ID','PERSON_ID']
num_cols = ['LABEL']
stat_list = ['max','min']

for c, n, f in itertools.product(cat_cols, num_cols, stat_list):
    df[f'{n}_{f}_grp_{c}'] = df.groupby(c)[n].transform(f)
    
# 在构造特征的过程中进行特征的命名。注意，这里因为python版本的问题，不再支持之前的采取（{'LABEL_MEAN'：'mean'}）的形式
df_grp = df.groupby('PERSON_ID')['LABEL'].agg([
                                              ('LABEL_MEAN', 'mean'),
                                              ('LABEL_STD', 'std'),
                                              ]).reset_index()

# 巧用字典和map的结合，省去merge的时间
dict_ = df.groupby('RECRUIT_ID')['LABEL'].agg('mean').to_dict()
df['LABEL_mean_grp_RECRUIT_ID'] = df['RECRUIT_ID'].map(dict_)
