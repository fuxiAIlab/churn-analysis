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

    if model_type == 'cforest':
        dic = {'2w_bh_task_ratio': 6.475675e-04,
        '2w_bind_yuanbao_get_ratio': 3.311456e-04,
        '2w_bl_task_ratio': 1.395837e-03,
        '2w_bt_task_ratio': 0.000000e+00,
        '2w_chujia_cnt': -8.057787e-10,
        '2w_cjg_die_cnt': 1.598502e-05,
        '2w_create_team_ratio': 3.131210e-04,
        '2w_d_avg_f_bl_task_cnt': 9.522499e-04,
        '2w_d_avg_f_ml_task_cnt': 1.352373e-03,
        '2w_d_avg_merge_equip_cnt': 2.659905e-04,
        '2w_d_avg_xilian_equip_cnt': 1.562314e-04,
        '2w_divorce_cnt': 0.000000e+00,
        '2w_equip_play_time_ratio': 6.381736e-04,
        '2w_exp_play_time_ratio': 2.256818e-03,
        '2w_get_exp_ratio': 1.397950e-03,
        '2w_get_purple_equip_amt': 1.786711e-03,
        '2w_get_skillexp_ratio': 1.637939e-03,
        '2w_get_yl_ratio': 5.936452e-04,
        '2w_get_yp_ratio': 6.943923e-04,
        '2w_guild_chuangong_cnt': 8.666633e-04,
        '2w_hjh_fail_ratio': 7.729079e-06,
        '2w_jj_task_ratio': 1.822978e-04,
        '2w_kickout_team_cnt': 4.384896e-03,
        '2w_kickout_team_ratio': 3.881583e-04,
        '2w_killed_ratio': 1.169986e-04,
        '2w_leisure_task_die_cnt': 2.017631e-03,
        '2w_leisure_yabiao_die_cnt': 3.907752e-05,
        '2w_lw_time_ratio': 5.990696e-05,
        '2w_mh_time_ratio': 7.087550e-04,
        '2w_ml_task_ratio': 1.581162e-03,
        '2w_not_bind_yuanbao_get_ratio': 8.711981e-03,
        '2w_passionfight_die_cnt': 1.202641e-04,
        '2w_pet_play_time_ratio': 5.532133e-06,
        '2w_qmlf_yx_time_ratio': 1.299488e-03,
        '2w_qmlfsxyqt_time_ratio': 2.081809e-03,
        '2w_qy_ratio': 4.202498e-04,
        '2w_shitu_play_cnt': 1.432651e-05,
        '2w_shl_time_ratio': 2.751435e-03,
        '2w_shop_money_get_amt': 1.996184e-04,
        '2w_sjtx_die_cnt': 1.280946e-03,
        '2w_sjtx_time_ratio': 8.352574e-04,
        '2w_szqx_time_ratio': 1.605355e-03,
        '2w_task_giveup_ratio': 1.048538e-03,
        '2w_team_match_ratio': 1.796025e-04,
        '2w_tushi_play_cnt': 2.279415e-05,
        '2w_weiwang_play_time_ratio': 9.311807e-04,
        '2w_wl_time_ratio': 8.671280e-04,
        '2w_wxl_time_ratio': 2.181825e-04,
        '2w_wyc_time_ratio': 5.137514e-04,
        '2w_zjx_time_ratio': 1.223151e-03,
        '2w_zl_time_ratio': 3.514253e-07,
        '2w_zy_time_ratio': 1.767752e-04,
        'acm_get_hero_card_amt': 1.102374e-02,
        'acm_qy_num': 2.317205e-03,
        'acm_sjbl_cnt': 4.145173e-03,
        'acm_up_9_level_skill_amt': 2.351233e-02,
        'addtitle_cnt': 1.438766e-02,
        'baishifail_cnt': 1.389351e-05,
        'churn_friends_ratio': 8.048648e-04,
        'couple_latest_log_time': 2.665799e-05,
        'del_shitu_acm_cnt': 1.763285e-03,
        'deled_shitu_acm_cnt': 1.606866e-03,
        'equip_score_upgrade': 9.207476e-04,
        'f_bl_task_acm_num': 1.791325e-03,
        'f_ml_task_acm_num': 4.229288e-03,
        'friends_chat_num': 7.863741e-04,
        'friends_num': 7.459957e-03,
        'fund': 3.575097e-03,
        'guild_level': 2.860874e-03,
        'kickout_guild_acm_cnt': 3.818467e-04,
        'latest_rare_word_num_sjwq': 2.639797e-06,
        'level_upgrade': 7.882802e-04,
        'low_maintain': 8.830684e-06,
        'nie_lian_time': 1.646907e-04,
        'practice_score_upgrade': 5.874507e-04,
        'shifu_latest_log_time': 5.733820e-05,
        'shop_bankrupt_num': 5.973757e-05,
        'shop_num': 1.481417e-04,
        'skill_avg_level_upgrade': 1.139780e-03,
        'total_score_upgrade': 1.601566e-03,
        'watch_movie_acm_pct_avg': 1.200939e-03,
        'wuxue_score_upgrade': 9.913760e-04}

        feature_importance = [6.475675e-04,3.311456e-04,1.395837e-03,0.000000e+00,-8.057787e-10,1.598502e-05,
        3.131210e-04,9.522499e-04,1.352373e-03,2.659905e-04,1.562314e-04,0.000000e+00,6.381736e-04,2.256818e-03,
        1.397950e-03,1.786711e-03,1.637939e-03,5.936452e-04,6.943923e-04,8.666633e-04,7.729079e-06,1.822978e-04,
        4.384896e-03,3.881583e-04,1.169986e-04,2.017631e-03,3.907752e-05,5.990696e-05,7.087550e-04,1.581162e-03,
        8.711981e-03,1.202641e-04,5.532133e-06,1.299488e-03,2.081809e-03,4.202498e-04,1.432651e-05,2.751435e-03,
        1.996184e-04,1.280946e-03,8.352574e-04,1.605355e-03,1.048538e-03,1.796025e-04,2.279415e-05,9.311807e-04,
        8.671280e-04,2.181825e-04,5.137514e-04,1.223151e-03,3.514253e-07,1.767752e-04,1.102374e-02,2.317205e-03,
        4.145173e-03,2.351233e-02,1.438766e-02,1.389351e-05,8.048648e-04,2.665799e-05,1.763285e-03,1.606866e-03,
        9.207476e-04,1.791325e-03,4.229288e-03,7.863741e-04,7.459957e-03,3.575097e-03,2.860874e-03,3.818467e-04,
        2.639797e-06,7.882802e-04,8.830684e-06,1.646907e-04,5.874507e-04,5.733820e-05,5.973757e-05,1.481417e-04,
        1.139780e-03,1.601566e-03,1.200939e-03,9.913760e-04]
        col_list = np.array(all_data.columns)
        p = np.argsort(-np.array(feature_importance))
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





