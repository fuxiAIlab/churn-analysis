# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/19 14:13
@Auth ： Yu Xiong
@File ：anchor_tabular.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from anchors import anchor_explanation
from anchors import anchor_base
from collections import Counter
import copy
import collections
import numpy as np
import json


class AnchorTabularExplainer(object):
    """
    Args:
        class_names:
        feature_names:
        train_data:
        categorical_names:
    """
    def __init__(self, class_names, feature_names, train_data, categorical_names={}, encoder_fn=None):
        self.class_names = class_names
        self.min = {}
        self.max = {}
        self.train = train_data
        self.feature_names = feature_names
        self.encoder_fn = lambda x: x
        if encoder_fn is not None:
            self.encoder_fn = encoder_fn

        self.categorical_names = copy.deepcopy(categorical_names)
        self.categorical_features = []
        if categorical_names:
            self.categorical_features = sorted(categorical_names.keys())

        for f in range(train_data.shape[1]):
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])

    def sample_from_train(self, conditions_eq, num_samples):
        train = self.train
        idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)
        sample = train[idx]

        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)
        return sample

    def get_sample_fn(self, data_row, classifier_fn, desired_label=None):
        def predict_fn(x):
            return classifier_fn(self.encoder_fn(x))
        true_label = desired_label
        if true_label is None:
            true_label = predict_fn(data_row.reshape(1, -1))[0]

        mapping = {}
        for f in self.categorical_features:
            idx = len(mapping)
            mapping[idx] = (f, 'eq', data_row[f])

        def sample_fn(present, num_samples, compute_labels=True):
            conditions_eq = {}
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v

            raw_data = self.sample_from_train(conditions_eq, num_samples)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (raw_data[:, f] == data_row[f]).astype(int)

            labels = []
            raw_data_labels = predict_fn(raw_data)
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, raw_data_labels, labels
        return sample_fn, mapping, true_label

    def explain_instance(self, data_row, classifier_fn, threshold=0.95,
                         delta=0.1, tau=0.15, batch_size=200,
                         max_anchor_size=None, desired_label=None,
                         beam_size=4, state_global=None, counterfactual=False, **kwargs):
        sample_fn, mapping, true_label = self.get_sample_fn(data_row, classifier_fn, desired_label=desired_label)

        exp = anchor_base.AnchorBaseBeam.anchor_beam(data_row, true_label, sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
                                                     desired_confidence=threshold, max_anchor_size=max_anchor_size, state_global=state_global, **kwargs)
        self.add_names_to_exp(exp, mapping)
        exp['instance'] = data_row
        exp['prediction'] = classifier_fn(self.encoder_fn(data_row.reshape(1, -1)))[0]
        counterfactual_anchors = []
        counterfactual_features = ''
        if counterfactual:
            counterfactual_anchors, counterfactual_features = self.get_counterfactual_anchor(exp, mapping, data_row, classifier_fn, true_label)
        explanation = anchor_explanation.AnchorExplanation('tabular', exp, true_label, counterfactual_anchors, counterfactual_features)
        return explanation

    def add_names_to_exp(self, hoeffding_exp, mapping):
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v] or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            hoeffding_exp['names'].append(fname)

    def add_names_to_counterfactual(self, feature_idxs, mapping):
        counterfactual_names = []
        for idx in feature_idxs:
            f, op, v = mapping[idx]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v] or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            counterfactual_names.append(fname)
        return counterfactual_names

    def get_counterfactual_anchor(self, exp, mapping, data_row, classifier_fn, true_label):
        idxs = exp['feature']
        precision = exp['precision'][-1]
        counterfactual_anchors = []
        counterfactual_features = []
        for idx in idxs:
            f, op, v = mapping[idx]
            for i in range(len(self.categorical_names[f])):
                if i == v:
                    continue
                data_row_copy = data_row.copy()
                data_row_copy[idx] = i
                sample_fn_copy, mapping_copy, true_label_copy = self.get_sample_fn(data_row_copy, classifier_fn)
                raw_data, data, raw_data_labels, _ = sample_fn_copy(idxs, 1000, compute_labels=False)
                most_count = Counter(raw_data_labels).most_common(1)[0]
                # print('label:', true_label, most_count)
                precision_copy = most_count[1] / 1000
                if most_count[0] == true_label:
                    #  or precision_copy < precision - 0.05
                    continue
                counterfactual_names = self.add_names_to_counterfactual(idxs, mapping_copy)
                counterfactual_anchors.append((counterfactual_names, most_count[0], precision_copy))
                counterfactual_feat = self.add_names_to_counterfactual([idx], mapping_copy)
                counterfactual_features.append((counterfactual_feat[0], most_count[0], precision_copy))
        return counterfactual_anchors, counterfactual_features





