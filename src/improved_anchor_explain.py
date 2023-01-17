# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/8 14:22
@Auth ： Yu Xiong
@File ：explain_dataset2.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

from anchors import anchor_tabular
# from anchor import anchor_tabular
from mdlps.ent_mdlp import mdlpx
import numpy as np
import pandas as pd
import pickle
import time
import lightgbm
import joblib
import configparser
import dill
import os
import pathos
import multiprocessing
from multiprocessing import Manager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, classification_report


def data_preprocess(train_path):
    train_df = pd.read_csv(train_path)
    train_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train_df = train_df.fillna(value=-1)
    print(train_df.info())
    return train_df


def data_discretize(data, data_disc_path, data_disc_dict_path):
    y = np.array(data['Survived'])
    X = data.drop(['Survived'], axis=1)
    continuous_features = ['Age', 'SibSp', 'Parch', 'Fare']
    feature_names = list(X.columns)
    continuous_features_idx = [feature_names.index(feature) for feature in continuous_features]
    for feature in continuous_features:
        if X[feature].dtypes == object:
            X[feature] = X[feature].astype(float)
    X_disc, dic, dic_all = mdlpx(X, y, continuous_features_idx, quantile=100)
    print(dic_all)
    X_disc['label'] = y
    X_disc.to_csv(data_disc_path)
    with open(data_disc_dict_path, 'wb') as f:
        pickle.dump(dic_all, f, pickle.HIGHEST_PROTOCOL)


def data_load(path):
    df = pd.read_csv(path, index_col=0)
    print(df['label'].value_counts())
    return df


def build_dataset(df, cut_path):
    y = np.array(df['label'])
    X = df.drop(['label'], axis=1)
    feature_names = list(X.columns)
    with open(cut_path, 'rb') as f:
        dic_all = pickle.load(f)
        print(dic_all)
    continuous_features = []
    continuous_names = {}
    for feature in dic_all:
        continuous_features.append(feature)
        if len(dic_all[feature][1]) == 1 and dic_all[feature][1][0] == dic_all[feature][2][0]:
            fname = '%.2f <= %s <= %.2f' % (dic_all[feature][0][0], feature, dic_all[feature][2][0])
            continuous_names[feature_names.index(feature)] = [fname]
        else:
            fname_list = ['%.2f <= %s <= %.2f' % (dic_all[feature][0][0], feature, dic_all[feature][1][0])]
            for i in range(1, len(dic_all[feature][1])):
                fname_list.append('%.2f < %s <= %.2f' % (dic_all[feature][1][i-1], feature, dic_all[feature][1][i]))
            fname_list.append('%.2f < %s <= %.2f' % (dic_all[feature][1][-1], feature, dic_all[feature][2][0]))
            continuous_names[feature_names.index(feature)] = fname_list

    # for feature in continuous_features:
    #     if X[feature].dtypes == object:
    #         X[feature] = X[feature].astype(float)

    categorical_features = [feature for feature in feature_names if feature not in continuous_features]
    for feature in categorical_features:
        # if X[feature].dtypes != object:
        X[feature] = X[feature].astype(str)
    categorical_names = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(X[feature])
        X[feature] = encoder.transform(X[feature])
        categorical_names[feature_names.index(feature)] = list(encoder.classes_)
    return X, y, categorical_features, categorical_names, continuous_features, continuous_names


def model_train(X, y, categorical_features, model_path, train=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    if train:
        model = lightgbm.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=1000, n_jobs=1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_logloss', early_stopping_rounds=20, categorical_feature=categorical_features)
        #list(X.columns)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
    preds = model.predict(X, num_iteration=model.best_iteration_)
    print('The accuracy is {}'.format(accuracy_score(y, preds)))
    print('The precision is {}'.format(precision_score(y, preds)))
    print('The recall is {}'.format(recall_score(y, preds)))
    return model


def model_train_overfit(X, y, categorical_features, model_path, train=True):
    if train:
        model = lightgbm.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=150, n_jobs=1)
        model.fit(X, y, eval_set=[(X, y)], eval_metric='binary_logloss', early_stopping_rounds=20, categorical_feature=categorical_features)
        # list(X.columns)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
    preds = model.predict(X, num_iteration=model.best_iteration_)
    print('The accuracy is {}'.format(accuracy_score(y, preds)))
    print('The precision is {}'.format(precision_score(y, preds)))
    print('The recall is {}'.format(recall_score(y, preds)))
    return model



def model_load(path):
    model = joblib.load(path)
    return model


def build_anchor_explainer(X, y, categorical_names, continuous_names):
    categorical_names.update(continuous_names)
    label_columns = ['Unsurvival', 'Survival']
    feature_names = X.columns
    explainer = anchor_tabular.AnchorTabularExplainer(label_columns, feature_names, X.values, categorical_names)
    return explainer, label_columns


def build_explanations_plus(target_X, explainer, model, anchor_path):
    target_X = target_X.values
    explanations = []
    state_global = {}
    # state_global=state_global,
    with open(anchor_path + 'explanation.txt', 'w', encoding='utf-8') as f:
        for i in range(target_X.shape[0]):
            start = time.time()
            explanation = explainer.explain_instance(target_X[i], model.predict, max_anchor_size=5, threshold=0.95, counterfactual=True)
            end = time.time()
            print('index:{}, times:{}'.format(i, end - start))
            explanations.append(explanation)
            condition = ' AND '.join(explanation.names())
            print('rule:', condition, explanation.label(), explanation.precision())
            print('counterfactual_rule:', explanation.counterfactual())
            f.write(condition + '\n')
    with open(anchor_path, 'wb') as f:
        dill.dump(explanations, f)
    return explanations


def parallel_build_explanations_plus(target_X, explainer, model, node_nums, anchor_path):
    def batch_build_explanations(batch, state_global):
        output_file = anchor_path + 'explanation-{}'.format(multiprocessing.current_process().pid)
        print('start multiprocessing explanation:' + output_file)
        batch_explanation = {}
        with open(output_file + '.txt', 'a+') as f:
            for i in batch:
                start = time.time()
                explanation = explainer.explain_instance(target_X[i], model.predict, max_anchor_size=5, threshold=0.95, state_global=state_global, counterfactual=False)
                end = time.time()
                print('index:{}, times:{}'.format(i, end - start))
                batch_explanation[i] = explanation
                condition = ' AND '.join(explanation.names())
                print('rule:', condition, explanation.label(), explanation.precision())
                # print('counterfactual_rule:', explanation.counterfactual())
                # f.write(str(i)+','+condition+','+str(explanation.label())+','+str(explanation.precision()) + '@@@' + str(explanation.counterfactual()) + '\n')
        # with open(output_file, 'wb') as f:
        #     dill.dump(batch_explanation, f)
        return batch_explanation
    target_X = target_X.values
    explanations = {}
    pool = pathos.multiprocessing.ProcessPool(nodes=node_nums)
    batch_num = node_nums
    batches = [[] for _ in range(batch_num)]
    for i in range(target_X.shape[0]):
        batches[i % batch_num].append(i)
    manager = Manager()
    state_global = manager.dict()
    batch_explanations = pool.amap(batch_build_explanations, batches, [state_global for _ in range(batch_num)]).get()
    pool.close()
    pool.join()
    for batch_explanation in batch_explanations:
        explanations.update(batch_explanation)
    explanations = [explanations[key] for key in sorted(explanations.keys())]
    return explanations


def build_explanations(target_X, explainer, model, anchor_path):
    target_X = target_X.values
    explanations = []
    with open(anchor_path + 'test.txt', 'w', encoding='utf-8') as f:
        for i in range(target_X.shape[0]):
            start = time.time()
            explanation = explainer.explain_instance(target_X[i], model.predict, max_anchor_size=5, threshold=0.95)
            end = time.time()
            print('index:{}, times:{}'.format(i, end - start))
            explanations.append(explanation)
            condition = ' AND '.join(explanation.names())
            print('rule:', condition)
            f.write(condition + '\n')
    return explanations


def explanations_load(path):
    explanation_list = os.listdir(path)
    indexs, rules, predictions, precisions, counterfactuals = [], [], [], [], []
    rule_set = set()
    for explanation in explanation_list:
        explanation_file = os.path.join(path, explanation)
        with open(explanation_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                # print(line)
                rules_info = line.split('@@@')[0]
                rule = rules_info.split(',')[1].split('AND')
                rule_s = tuple(set([predicate.strip() for predicate in rule]))
                if rule_s not in rule_set:
                    rule_set.add(rule_s)
                    indexs.append(rules_info.split(',')[0])
                    rules.append(rules_info.split(',')[1])
                    predictions.append(int(rules_info.split(',')[2]))
                    precisions.append(float(rules_info.split(',')[3]))
                    counterfactuals.append(eval(line.split('@@@')[1]))
    return indexs, rules, predictions, precisions, counterfactuals


def build_global_explanations(explanations_path, explainer, X, y, label_columns, result_path):
    def metric_explanation(X, prediction, rule, feature_names):
        fit_anchor = range(X.shape[0])
        use = 1
        precision_increases = []
        explanation = rule.split('AND')
        for predicate in explanation:
            predicate = predicate.strip()
            fit_anchor_tmp = fit_anchor
            if '=' in predicate and '<=' not in predicate:
                predicate_list = predicate.split('=')
                feature_name, token, feature_value = predicate_list[0].strip(), '=', predicate_list[1].strip()
                feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(feature_value)
            else:
                predicate_list = predicate.split()
                s_feature_value, s_token, feature_name, b_token, b_feature_value = float(predicate_list[0]), \
                                                                                   predicate_list[1], predicate_list[2], \
                                                                                   predicate_list[3], float(
                    predicate_list[4])
                # print(explainer.categorical_names[feature_names.index(feature_name)])
                feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(predicate)
            fit_anchor = np.intersect1d(np.where(X[:, feature_names.index(feature_name)] == feature_value), fit_anchor_tmp)
            precision_previous = np.mean(y[fit_anchor_tmp] == prediction)
            precision_now = np.mean(y[fit_anchor] == prediction)
            precision_increases.append(precision_now - precision_previous)
            if precision_now <= precision_previous:
                use = 0
        length = len(explanation)
        dataset_precision = np.mean(y[fit_anchor] == prediction)
        dataset_coverage = fit_anchor.shape[0] / float(X.shape[0])
        number = fit_anchor.shape[0]
        pred = explainer.class_names[list(np.unique(y)).index(prediction)]
        return length, dataset_precision, dataset_coverage, number, pred, use, precision_increases

    def metric_counterfactual(X, rules, feature_names):
        counterfactual_info = []
        for rule in rules:
            explanation = rule[0]
            prediction = rule[1]
            precision = rule[2]
            fit_anchor = range(X.shape[0])
            for predicate in explanation:
                fit_anchor_tmp = fit_anchor
                if '=' in predicate and '<=' not in predicate:
                    predicate_list = predicate.split('=')
                    feature_name, token, feature_value = predicate_list[0].strip(), '=', predicate_list[1].strip()
                    feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(
                        feature_value)
                else:
                    predicate_list = predicate.split()
                    s_feature_value, s_token, feature_name, b_token, b_feature_value = float(predicate_list[0]), \
                                                                                       predicate_list[1], \
                                                                                       predicate_list[2], \
                                                                                       predicate_list[3], float(
                        predicate_list[4])
                    feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(
                        predicate)
                fit_anchor = np.intersect1d(np.where(X[:, feature_names.index(feature_name)] == feature_value),
                                            fit_anchor_tmp)
            dataset_precision = np.mean(y[fit_anchor] == prediction)
            dataset_coverage = fit_anchor.shape[0] / float(X.shape[0])
            number = fit_anchor.shape[0]
            counterfactual_info.append(str([explanation, dataset_precision, number, precision]))
        return counterfactual_info
    print(explanations_path)
    indexs, rules, predictions, precisions, counterfactuals = explanations_load(explanations_path)

    df = pd.DataFrame()
    feature_names = list(X.columns)
    X = X.values
    lengths, dataset_precisions, dataset_coverages, conditions, preds, numbers, uses, precision_increases, counterfactual_infos = [], [], [], [], [], [], [], [], []
    for i in range(len(indexs)):
        print('index:{}'.format(i))
        length, dataset_precision, dataset_coverage, number, pred, use, precision_increase = metric_explanation(X, predictions[i], rules[i], feature_names)
        lengths.append(length)
        dataset_precisions.append(dataset_precision)
        dataset_coverages.append(dataset_coverage)
        preds.append(pred)
        numbers.append(number)
        uses.append(use)
        precision_increases.append(precision_increase)
        counterfactual_infos.append(metric_counterfactual(X, counterfactuals[i], feature_names))

    df['rule'] = rules
    df['prediction'] = preds
    df['length'] = lengths
    df['dataset_precision'] = dataset_precisions
    df['dataset_numbers'] = numbers
    df['dataset_coverage'] = dataset_coverages
    df['precision_increase'] = precision_increases
    df['disturb_space_precision'] = precisions
    df['counterfactual_rules'] = counterfactual_infos
    df['uses'] = uses
    df = df.drop(['uses'], axis=1)
    df = df.drop_duplicates(['rule'])
    df.sort_values(by=['dataset_precision', 'dataset_coverage', 'length'], ascending=(False, False, True), inplace=True)
    for label in label_columns:
        df[df['prediction'] == label].to_csv(result_path+'{}.csv'.format(label), encoding='gbk')


if __name__ == '__main__':
    train_path = 'data/titanic/train.csv'
    test_path = 'data/titanic/test.csv'
    # df = data_preprocess(train_path)

    data1_disc_path = 'data/titanic/titanic_disc.csv'
    data1_disc_dict_path = 'data/titanic/titanic_disc_dict.pkl'
    # data_discretize(df, data1_disc_path, data1_disc_dict_path)

    model_path = 'model/lgb_titanic.pkl'
    # 读入离散化数据，构建数据集，模型训练
    df = data_load(data1_disc_path)
    X, y, categorical_features, categorical_names, continuous_features, continuous_names = build_dataset(df, data1_disc_dict_path)
    model = model_train(X, y, categorical_features, model_path)

    anchor_path = 'data/titanic/anchor/rule/'
    model = model_load(model_path)
    explainer, label_columns = build_anchor_explainer(X, y, categorical_names, continuous_names)

    # explanations = build_explanations_plus(X, explainer, model, anchor_path)
    start = time.time()
    explanations = parallel_build_explanations_plus(X, explainer, model, 1, anchor_path)
    # explanations = build_explanations(X, explainer, model, anchor_path)
    end = time.time()
    print('all time consume:{}'.format(end-start))
    result_path = 'data/titanic/anchor/result/'
    # build_global_explanations(anchor_path, explainer, X, y, label_columns, result_path)

