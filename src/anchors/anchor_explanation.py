# -*- coding: utf-8 -*-


class AnchorExplanation:
    def __init__(self, type_, exp_map, true_label, counterfactual_rules, counterfactual_features):
        self.type = type_
        self.exp_map = exp_map
        self.true_label = true_label
        self.counterfactual_rules = counterfactual_rules
        self.counterfactual_features = counterfactual_features

    def names(self, partial_index=None):
        names = self.exp_map['names']
        if partial_index is not None:
            names = names[:partial_index + 1]
        return names

    def features(self, partial_index=None):
        features = self.exp_map['feature']
        if partial_index is not None:
            features = features[:partial_index + 1]
        return features

    def precision(self, partial_index=None):
        precision = self.exp_map['precision']
        if len(precision) == 0:
            return self.exp_map['all_precision']
        if partial_index is not None:
            return precision[partial_index]
        else:
            return precision[-1]

    def coverage(self, partial_index=None):
        coverage = self.exp_map['coverage']
        if len(coverage) == 0:
            return 1
        if partial_index is not None:
            return coverage[partial_index]
        else:
            return coverage[-1]

    def label(self):
        return self.true_label

    def counterfactual(self):
        return self.counterfactual_rules

    def counterfactual_feature(self):
        return self.counterfactual_features