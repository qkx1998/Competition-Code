from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score,auc

not_used_cols = [c for c in train.columns if train[c].dtype in ['datetime64[ns]','object']]
used_cols = [x for x in train.columns if x not in ['label','ID']+not_used_cols]

train_pred = np.zeros(train.shape[0])
test_pred = np.zeros(test.shape[0])
label = 'label'

kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=2)
for i, (ta_idx , val_idx) in enumerate(kf.split(train,train[label])):
    X_ta, X_val, y_ta, y_val = train[used_cols].iloc[ta_idx], train[used_cols].iloc[val_idx],np.array(train[label])[ta_idx], np.array(train[label])[val_idx]
    
    model_cat = CatBoostClassifier(learning_rate=0.07, 
                                   n_estimators = 10000,
                                   depth = 8,
                                   subsample=0.8,
                                   l2_leaf_reg=3,
                                   random_seed = 2021,
                                   colsample_bylevel = 0.6,
                                   eval_metric='AUC',
                                   loss_function='Logloss', 
                                   early_stopping_rounds=200,
                                   verbose=200)
    model_cat.fit(X_ta, y_ta.astype('int32'), eval_set = [(X_val,y_val.astype('int32'))], early_stopping_rounds=200, verbose=200)
   
    train_pred[val_idx] =  model_cat.predict_proba(X_val)[:,1]   
    test_pred +=  model_cat.predict_proba(test[used_cols])[:,1] / kf.n_splits
    imp = pd.Series(model_cat.feature_importances_, used_cols).sort_values(ascending=False)
    gc.collect()
