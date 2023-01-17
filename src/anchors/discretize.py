# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/19 15:50
@Auth ： Yu Xiong
@File ：discretize.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from abc import ABCMeta, abstractmethod
import numpy as np

class BaseDiscretizer():
    __metaclass__ = ABCMeta

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None, data_stats=None):
        """
        Args:
            data:
            categorical_features:
            categorical_names:
            feature_names:
            data_stats:
        """
        self.to_discretize = ([x for x in range(data.shape[1]) if x not in categorical_features])
        self.data_stats = data_stats
        self.names = {}
        self.lambdas = {}

        bins = self.bins(data, labels)

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]
            boundaries = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]

            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <=%.2f' % (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](data[:, feature])


    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """
        Discretizes the data.
        """
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](ret[:, feature]).astype(int)
        return ret


class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_feature, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_feature, feature_names, labels=labels, random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
            bins.append(qts)
        return bins


class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_feature, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_feature, feature_names, labels=labels,
                                 random_state=random_state)
