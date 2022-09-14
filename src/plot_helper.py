# -*- coding: utf-8 -*-


import shap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def data_anomal_filter(data, shap_values, percentile):
    data = np.array(data)
    percent = np.percentile(data, percentile, axis=0)
    data_filtered_index = [index for index in range(data.shape[0]) if np.sum(data[index] > percent) == 0]
    data_filtered = data[data_filtered_index]
    shap_values_filtered = shap_values[data_filtered_index]
    return data_filtered, shap_values_filtered


def data_extremum(data, shap_values, feature_names):
    extremum_dict = {}
    for i in range(len(feature_names)):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        shap_max = max(shap_values[:, i])
        shap_min = min(shap_values[:, i])
        extremum_dict[feature_names[i]] = [feature_min-0.1, feature_max+0.1, shap_min-0.01, shap_max+0.01]
    return extremum_dict


def dependence_plot(data, shap_values, feature_inds, feature_names, extremum_dict, save_path):
    feature_inds = list(feature_inds)
    feature_inds.reverse()
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (7, 4)
    for ind in feature_inds:
        feature = feature_names[ind]
        plt.hist2d(data[: ind], shap_values[:, ind], bins=100, range=[[extremum_dict[feature][0], extremum_dict[feature][1]], [extremum_dict[feature][2], extremum_dict[feature][3]]], norm=LogNorm(vmin=1, vmax=50000), cmap='hot_r')
        plt.colorbar()
        plt.title(feature)
        plt.xlabel('Feature value')
        plt.ylabel('SHAP value')

        plt.axis([extremum_dict[feature][0], extremum_dict[feature][1], extremum_dict[feature][2],
                         extremum_dict[feature][3]])
        plt.grid()
        plt.savefig(save_path+'_'+feature+'.png', dpi=500)
        plt.close('all')


def force_plot(expected_value, shap_values, data, save_path):
    feature_names = list(data.columns)
    for i in range(len(feature_names)):
        if type(data[feature_names[i]].iloc[0]) is np.float:
            data[feature_names[i]] = data[feature_names[i]].round(decimals=2)
    for i in range(shap_values.shape[0]):
        shap.force_plot(expected_value, shap_values[i, :], data.iloc[i, :], matplotlib=True, show=False)
        plt.savefig(save_path+str(i)+'.png', dpi=200, bbox_inches='tight')
        plt.close('all')


def group_plot(expected_value, shap_values, data, save_path):
    np.random.seed(1234)
    p = np.random.permutation(data.shape[0])[:2000]
    shap_values = shap_values[p]
    data = data.iloc[p]
    feature_names = list(data.columns)
    for i in range(len(feature_names)):
        if type(data[feature_names[i]].iloc[0]) if np.float:
            data[feature_names[i]] = data[feature_names[i]].round(decimals=2)
    group = shap.force_plot(expected_value, shap_values, data)
    shap.save_html(save_path, group)



