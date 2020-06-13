# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:53:06 2019

@author: KGU2BAN
"""
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from numpy import zeros, extract

def ml_relevance_matrix_etr(feature_array, target_array, n_estimators):
    relevance_matrix = zeros((feature_array.shape[1], target_array.shape[1]))
    for i in range(target_array.shape[1]):
        #Evaluating the relavance of input to outputs, model prepared with only test to all ratio as 0.2
        model = ExtraTreesRegressor(n_estimators=n_estimators)
        model.fit(feature_array,target_array[:,i])
        relevance_matrix[:,i] = model.feature_importances_
    return relevance_matrix

def ml_r2_list(y_true_array, y_predict_array):
    r2_list = []
    if y_true_array.shape[1] == y_predict_array.shape[1]:
        for i in range(y_true_array.shape[1]):
            y_predict_temp = y_predict_array[:,i]
            y_true_temp = y_true_array[:,i]
            y_predict_temp = extract(y_true_temp>0, y_predict_temp)
            y_true_temp = extract(y_true_temp>0, y_true_temp)
            r2_list.append(r2_score(y_true_temp, y_predict_temp))
    return r2_list
