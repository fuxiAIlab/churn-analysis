#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import pathos
import sklearn
import joblib
import xgboost as xgb
import lightgbm as lgb
from anchor import anchor_tabular


def data_load(path):
    df = pd.read_csv(path)
    return df


def data_processing(df):
    df['label'] = df.label.apply(lambda x: 1 if x == 'churn' else 0)
    columns = df.drop(['label'], axis=1).columns
    X = df.drop(['label'], axis=1).values
    y = df.label.values
    return X, y, columns


def model_load(path):
    model = joblib.load(path)
    return model


def build_anchor_explainer(X_train, y_train, X_test, y_test, columns, discretizer='quartile'):
    label_columns = ['retain', 'churn']
    feature_columns = columns
    data = X_train
    categorical_columns = {}
    ordinal_features = columns
    explainer = anchor_tabular.AnchorTabularExplainer(label_columns, feature_columns, data, categorical_columns,
                                                      ordinal_features)
    explainer.fit(X_train, y_train, X_test, y_test, discretizer=discretizer)
    return explainer


def parallel_build_explanations(target_X, explainer, model, cpus=8):
    def metric_explanation(X, x, explanation, explainer, model):
        d_target_X = explainer.disc.discretize(target_X)
        fit_anchor = np.where(
            np.all(d_target_X[X][:, explanation.features()] == d_target_X[x][explanation.features()], axis=1))[0]
        length = len(explanation.names())
        precision = np.mean(
            model.predict(d_target_X[X][fit_anchor]) == model.predict(d_target_X[x].reshape(1, -1)))
        coverage = fit_anchor.shape[0] / float(len(X))
        condition = 'AND'.join(explanation.names())
        pred = explainer.class_names[model.predict(d_target_X[x].reshape(1, -1))[0]]
        return length, precision, coverage, condition, pred

    def batch_build_explanations(batch):
        batch_explanations = {}
        newline = ''
        for x in batch:
            explanation = explainer.explain_instance(target_X[x],
                                                     model.predict,
                                                     max_anchor_size=6,
                                                     threshold=0.95)
            batch_explanations[x] = explanation
            length, precision, coverage, condition, pred = metric_explanation(range(target_X.shape[0]),
                                                                              x,
                                                                              explanation,
                                                                              explainer,
                                                                              model)

            newline = newline + ','.join(list(map(str, [x,
                                                        length,
                                                        explanation.precision(),
                                                        explanation.coverage(),
                                                        precision,
                                                        coverage,
                                                        condition]))) + '\n'

        output = '/data/tmp/output/output-%d.txt' % (mp.current_process().pid)
        with open(output, 'a+') as f:
            f.write(newline)
        return batch_explanations

    explanations = {}
    batch_explanations = []
    pool = pathos.multiprocessing.ProcessPool(nodes=cpus)
    batch_num = cpus
    batches = [[] for _ in range(batch_num)]
    for i in range(target_X.shape[0]):
        if isinstance(batches[i % batch_num], list):
            batches[i % batch_num].append(i)
        else:
            batches[i % batch_num] = []
            batches[i % batch_num].append(i)
    print(batches)
    for batch in batches:
        batch_explanations.append(pool.apipe(batch_build_explanations, batch))
    pool.close()
    pool.join()
    for batch_explanation in batch_explanations:
        explanations.update(batch_explanation.get())
    explanations = [explanations[key] for key in sorted(explanations.keys())]
    return explanations


# X：原始样本的index列表
def build_learner(explanations, explainer, model, X):
    def metric_explanation(X, x, explanation, explainer, model):
        d_target_X = explainer.disc.discretize(target_X)
        fit_anchor = np.where(
            np.all(d_target_X[X][:, explanation.features()] == d_target_X[x][explanation.features()], axis=1))[0]
        length = len(explanation.names())
        precision = np.mean(
            model.predict(d_target_X[X][fit_anchor]) == model.predict(d_target_X[x].reshape(1, -1)))
        coverage = fit_anchor.shape[0] / float(len(X))
        condition = 'AND'.join(explanation.names())
        pred = explainer.class_names[model.predict(d_target_X[x].reshape(1, -1))[0]]
        return length, precision, coverage, condition, pred

    # X：原始样本的index列表
    def metric(explanations, explainer, model, X):
        df = pd.DataFrame()
        lengths = []
        precisions = []
        coverages = []
        conditions = []
        preds = []
        for x in X:
            length, precision, coverage, condition, pred = metric_explanation(X=X,
                                                                              x=x,
                                                                              explanation=explanations[x],
                                                                              explainer=explainer,
                                                                              model=model)
            lengths.append(length)
            precisions.append(precision)
            coverages.append(coverage)
            conditions.append(condition)
            preds.append(pred)
        df['rule'] = X
        df['length'] = lengths
        df['precision'] = precisions
        df['coverage'] = coverages
        df['condition'] = conditions
        df['pred'] = preds
        return df

    learner = []
    default_pred = 0
    while True:
        # 在当前样本x中计算规则集合rules中每个规则的覆盖率和准确率
        # 选择最优的rule(i)加入候选模型，rules移除rule
        # X中移除正确分类的样本
        # 样本全部分类完成 或者 当前top 1规则准确率小于默认类别准确率
        if len(X) == 0:
            break
        X_pred = model.predict(explainer.validation[X])
        default_pred = np.argmax(np.bincount(X_pred))
        default_precision = np.mean(X_pred == default_pred)
        df = metric(explanations, explainer, model, X)
        # 初筛条件 覆盖率大于等于0.01
        X = np.setdiff1d(X, df[df.coverage < 0.01].rule.values)
        df = df[df.coverage >= 0.01]
        df.sort_values(by=['precision', 'coverage', 'length'], ascending=(False, False, True), inplace=True)
        if df.head(1).precision.values[0] <= default_precision:
            learner.append(-1)
            break
        best_explanation = df.head(1).rule.values[0]
        learner.append(best_explanation)
        fit_anchor = np.where(
            np.all(
                explainer.d_validation[:, explanations[best_explanation].features()] ==
                explainer.d_validation[best_explanation][explanations[best_explanation].features()],
                axis=1))[0]
        print(best_explanation)
        print(fit_anchor)
        X = np.setdiff1d(X, fit_anchor)
        print(X)
        del df
    return learner, default_pred


def metric_learner(learner, X, explainer, explanations, model, default_pred):
    explainer_pred = np.zeros(len(X))
    model_pred = model.predict(explainer.validation[X])
    for rule in learner:
        if rule == -1:
            explainer_pred[X] = default_pred
            break
        else:
            fit_anchor = np.where(np.all(
                explainer.d_validation[:, explanations[rule].features()] == explainer.d_validation[rule][
                    explanations[rule].features()],
                axis=1))[0]
            explainer_pred[np.intersect1d(fit_anchor, X)] = model.predict(explainer.validation[rule].reshape(1, -1))
            X = np.setdiff1d(X, fit_anchor)
    precision = np.mean(explainer_pred == model_pred)
    print(precision)
    return precision


def parse_learner(learner, explanations, default_pred, path):
    df_feature = pd.read_csv(path, sep='=', names=['name_en', 'name_zh'])
    dict_feature = df_feature.set_index('name_en').T.to_dict('list')
    for rule in learner:
        if rule == -1:
            print('Else', default_pred)
        else:
            print(' AND '.join(
                [str(dict_feature[x.split(' ')[0]][0]) + ' ' + str(' '.join(x.split(' ')[1:])) for x in
                 explanations[rule].names()]))


if __name__ == '__main__':
    # 数据路径
    data_path = 'data/dataset.csv'
    model_path = 'data/lgb.model'
    parse_path = 'data/feature.txt'
    # 获取数据集
    data = data_load(data_path)
    # 数据预处理
    X, y, columns = data_processing(data)
    # 创建预测模型
    model = model_load(model_path)
    # 模型准确率
    print('prediction model accuracy:', sklearn.metrics.accuracy_score(y, model.predict(X)))
    # 目标数据集
    target_df = data[(data['is_high_pay'] == 1) & (data['label'] == 'churn')]
    target_X, target_y, target_columns = data_processing(target_df)
    print('target dataset shape:', target_X.shape)
    # 创建anchor解释器
    anchor_explainer = build_anchor_explainer(X, y, X, y, columns, discretizer='quartile')
    # 多进程创建规则解释(个体)
    start = time.time()
    explanations = parallel_build_explanations(target_X, anchor_explainer, model, cpus=5)
    end = time.time()
    print('times:', end - start)
    # 创建规则解释(全局)
    learner, default_pred = build_learner(explanations, anchor_explainer, model, anchor_explainer.validation.shape[0])
    metric_learner(learner, anchor_explainer.validation.shape[0], anchor_explainer, explanations, model, default_pred)
    parse_learner(learner, explanations, default_pred, parse_path)
