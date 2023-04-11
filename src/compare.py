# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import pickle
import xgboost as xgb
import lightgbm as lgb
import catboost as cbst
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import shap


def model_training(model_type='lr'):
    churn_df = pd.read_csv('data/nsh_churn_no_roleid.csv')
    retain_df = pd.read_csv('data/nsh_retain_no_roleid.csv')
    print(churn_df.info())
    print(retain_df.info())
    all_data = pd.concat([churn_df, retain_df])
    all_data_inf = np.isinf(all_data)
    all_data[all_data_inf] = 0
    all_label = np.append(np.ones(churn_df.shape[0]), np.zeros(retain_df.shape[0]))
    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2, random_state=42)
    if model_type == 'lr':
        ss = StandardScaler()
        train_data = ss.fit_transform(train_data)
        test_data = ss.fit_transform(test_data)
        lr = LogisticRegression()
        lr.fit(train_data, train_label)
        predictions = lr.predict_proba(test_data)[:, 1]
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        with open('model/lr_auc{:.4f}.pickle'.format(auc), 'wb') as f:
            pickle.dump(lr, f)
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'tree':
        #
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(train_data, train_label)
        predictions = clf.predict_proba(test_data)[:, 1]
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        with open('model/tree_auc{:.4f}.pickle'.format(auc), 'wb') as f:
            pickle.dump(clf, f)
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'rf':
        clf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=500, max_depth=7)
        clf.fit(train_data, train_label)
        predictions = clf.predict_proba(test_data)[:, 1]
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        with open('model/rf_auc{:.4f}.pickle'.format(auc), 'wb') as f:
            pickle.dump(clf, f)
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'cbst':
        model = cbst.CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.5, loss_function='Logloss', eval_metric='AUC', random_seed=696, reg_lambda=3,
                                        verbose=True)
        model.fit(train_data, train_label, eval_set=(test_data, test_label), early_stopping_rounds=20)
        predictions = model.predict_proba(test_data)[:, 1]
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model/cbst_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'lgb':
        lgb_train = lgb.Dataset(train_data, train_label)
        lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test, early_stopping_rounds=20)
        predictions = model.predict(test_data, num_iteration=model.best_iteration)
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model/lgb_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'xgb':
        dtrain = xgb.DMatrix(train_data, label=train_label)
        dtest = xgb.DMatrix(test_data, label=test_label)
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'lambda': 1,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 2,
            'eta': 0.025,
            'seed': 0,
            'nthread': 8,
            'silent': 1
        }
        watchlist = [(dtest, 'validation')]
        model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
        predictions = model.predict(dtest)
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model/xgb_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'mlp':
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        inputs = Input(shape=(train_data.shape[1],))
        dense1 = Dense(64, activation='tanh')(inputs)
        dense2 = Dense(64, activation='tanh')(dense1)
        outputs = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        weight_path = 'model_nsh/mlp.hdf5'
        check_point = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [check_point]
        model.fit(train_data, train_label, epochs=15, batch_size=32, validation_data=(test_data, test_label),
                  callbacks=callbacks_list)
        predictions = model.predict(test_data)
        predictions = [score[0] for score in predictions]
        print(predictions)
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))


def feature_selection(model_type):
    churn_df = pd.read_csv('data/nsh_churn_no_roleid.csv')
    retain_df = pd.read_csv('data/nsh_retain_no_roleid.csv')
    all_data = pd.concat([churn_df, retain_df])
    all_data_inf = np.isinf(all_data)
    all_data[all_data_inf] = 0
    all_label = np.append(np.ones(churn_df.shape[0]), np.zeros(retain_df.shape[0]))
    if model_type == 'lr':
        with open('model/lr.pickle', 'rb') as f:
            model = pickle.load(f)
        weight = np.abs(model.coef_[0])
        col_list = np.array(all_data.columns)
        p = np.argsort(-weight)
        col_rank = col_list[p]
        print(col_rank[:5])

    if model_type == 'tree':
        with open('model/tree.pickle', 'rb') as f:
            model = pickle.load(f)
        feat_importance = model.tree_.compute_feature_importances(normalize=False)
        # print('feat importance = ' + str(feat_importance))
        col_list = np.array(all_data.columns)
        p = np.argsort(-feat_importance)
        col_rank = col_list[p]
        print(col_rank[:5])

    if model_type == 'lgb':
        model = lgb.Booster(model_file='model/lgb.model')
        preds = model.predict(all_data)
        feat_imp = model.get_score(importance_type='total_gain')
        col_rank = []
        col_imp = []
        for key, value in sorted(feat_imp.items(), key=lambda x:x[1], reverse=True):
            col_rank.append(key)
            col_imp.append(value)
        col_rank = np.array(col_rank)
        print(col_rank[:5])
        print(col_imp)

    if model_type == 'lgb_shap':
        model = lgb.Booster(model_file='model/lgb.model')
        bg_data = all_data.sample(n=200, random_state=42)
        preds = model.predict(all_data)
        shap.initjs()
        explainer = shap.TreeExplainer(model, data=bg_data)
        shap_values = explainer.shap_values(all_data)
        shap_values = np.sum(np.abs(shap_values), axis=0)
        col_list = np.array(all_data.columns)
        p = np.argsort(-shap_values)
        col_rank = col_list[p]
        print(col_rank[:5])

    n = [10, 20, 30, 40, 50]
    for x in n:
        col_list_tmp = col_rank[:x]
        all_data_tmp = all_data[col_list_tmp]
        train_data, test_data, train_label, test_label = train_test_split(all_data_tmp, all_label, test_size=0.2,
                                                                          random_state=42)
        lgb_train = lgb.Dataset(train_data, train_label)
        lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test, early_stopping_rounds=20)
        predictions = model.predict(test_data, num_iteration=model.best_iteration)
        auc = roc_auc_score(test_label, predictions)
        print(x, ' features')
        print('The roc of prediction is {}'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))


if __name__ == '__main__':
    model_training(model_type='tree')
    # feature_selection(model_type='lgb')





