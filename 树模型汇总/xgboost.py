from sklearn import datasets
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train_pred = np.zeros(train.shape[0])
test_pred = np.zeros(test.shape[0])
label = 'label'

kf = StratifiedKFold(n_splits=3, shuffle=True,random_state=2)
for i, (ta_idx , val_idx) in enumerate(kf.split(train,train[label])):
    X_ta, X_val, y_ta, y_val = train[used_cols].iloc[ta_idx], train[used_cols].iloc[val_idx],np.array(train[label])[ta_idx], np.array(train[label])[val_idx]
    
    model_xgb = XGBClassifier(learning_rate=0.1,
                      n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      objective='multi:softmax', # 指定损失函数
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27            # 随机数
                      )
    
    model_xgb.fit(x_train, y_train, eval_set = [(x_test,y_test)], eval_metric = "mlogloss", early_stopping_rounds = 10, verbose = True)
   
    train_pred[val_idx] =  model_xgb.predict_proba(X_val)[:,1]   
    test_pred +=  model_xgb.predict_proba(test[used_cols])[:,1] / kf.n_splits
    imp = pd.Series(model_xgb.feature_importances_, used_cols).sort_values(ascending=False)
    gc.collect()
### make prediction for test data
y_pred = model.predict(x_test)
 
### model evaluate
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))
"""
95.74%

