from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import *

base_cols = ['uid', 'label']
used_cols = [c for c in train.columns if train[c].dtype != 'object' and c not in base_cols]
label = 'label'
print(len(used_cols), used_cols)

train_pred = np.zeros(train.shape[0])
test_pred = np.zeros(test.shape[0])
imps = []

kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=5)
for i, (ta_idx , val_idx) in enumerate(kf.split(train,train[label])):
    X_ta, X_val, y_ta, y_val = train[used_cols].iloc[ta_idx], train[used_cols].iloc[val_idx],np.array(train[label])[ta_idx], np.array(train[label])[val_idx]
    
    model_lgb = LGBMClassifier(num_leaves=16,
                               objective='binary', # 还可以是 multiclass，regression，regression_l1，mape
                               n_estimators=10000,
                               learning_rate=0.01,
                               lambda_l1=0.1,
                               lambda_l2=0.1,
                               verbose=-1,
                               metric='auc',
                               n_jobs=16)
    
    model_lgb.fit(X_ta, y_ta, eval_set = [(X_val,y_val)], early_stopping_rounds=200, verbose=200)
    
    train_pred[val_idx] = model_lgb.predict_proba(X_val)[:,1]
    test_pred += model_lgb.predict_proba(test[used_cols])[:,1] / kf.n_splits 
   
    imps.append(pd.DataFrame(model_lgb.feature_importances_))
    gc.collect()
 
roc_auc_score(train[label], train_pred)

imp = pd.concat(imps, axis=1)
imp.columns = [f'fold_{i}_score' for i in range(1,6)]
imp['col'] = used_cols
