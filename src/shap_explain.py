# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import shap
import time
from plot_helper import data_anomal_filter, data_extremum, dependence_plot, force_plot, group_plot


def model_explain(churn_path, retain_path):
    churn_data = pd.read_csv(churn_path)
    retain_data = pd.read_csv(retain_path)
    all_data = pd.concat([churn_data, retain_data])
    all_labels = np.append(np.ones(churn_data.shape[0]), np.zeros(retain_data.shape[0]))

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_labels, test_size=0.1, random_state=42)
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
    preds = model.predict(test_data, num_iteration=model.best_iteration)

    preds_y = [round(pred) for pred in preds]
    print('The roc of prediction is {}'.format(roc_auc_score(test_label, preds)))
    print('The acc of prediction is {}'.format(accuracy_score(test_label, preds_y)))
    print('The precision of prediction is {}'.format(precision_score(test_label, preds_y)))
    print('The recall of prediction is {}'.format(recall_score(test_label, preds_y)))
    bg_data = train_data.sample(n=200, random_state=42)
    explainer = shap.TreeExplainer(model, bg_data)
    start = time.clock()
    shap_values = explainer.shap_values(all_data)
    end = time.clock()
    print('times:', end - start)

    data_filtered, shap_values_filtered = data_anomal_filter(all_data, shap_values, 99.95)

    feature_inds = shap.summary_plot(shap_values_filtered, data_filtered, feature_names=all_data.columns, max_display=20)
    extremum_dict = data_extremum(data_filtered, shap_values_filtered, all_data.columns)

    dependence_plot(data_filtered, shap_values_filtered, feature_inds, all_data.columns, extremum_dict, save_path='dependence/')

    force_plot(explainer.expected_value, shap_values_filtered, data_filtered, save_path='force/')

    group_plot(explainer.expected_value, shap_values_filtered, data_filtered, save_path='group/')


if __name__ == '__main__':
    model_explain(churn_path='data/churn.csv', retain_path='data/retain.csv')


