import pandas as pd

# 用 np.where来做寻找。示例：如果label为0的位置，则新特征label_reverse为1。反之。
df['label_reverse'] = np.where(df['LABEL'] == 0, 1, 0)
