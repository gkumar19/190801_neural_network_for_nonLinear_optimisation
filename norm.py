# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:07:36 2019

@author: KGU2BAN
"""
import tensorflow as tf
import numpy as np
import pandas as pd

class NormalizationTensor():
    '''normalize the data to be inserted into neural network and store
    mean and std for usage later on in an object, also exponential to target is applied by default,
    during conversion'''
    def __init__(self, feature_array, target_array):
        #fit the feature_array
        self.feature_mean = np.mean(feature_array, axis = 0, keepdims = True)
        self.feature_std = np.std(feature_array, axis = 0, keepdims = True)
        
        #fit the target_array
        target_array = np.log(target_array)
        self.target_mean = np.mean(target_array, axis = 0, keepdims = True)
        self.target_std = np.std(target_array, axis = 0, keepdims = True)
    def f2n(self,input_array):
        '''convert actual feature to normalised feature'''
        return (input_array - self.feature_mean) / self.feature_std
    def t2n(self,input_array):
        '''convert actual target to normalised target,
        also natural log is applied by default
        (because target of engine should be handeled in exponential space)'''
        array = np.log(input_array)
        return (array - self.target_mean) / self.target_std
    def n2f(self,input_array):
        '''convert normalised feature to actual feature'''
        return input_array*self.feature_std + self.feature_mean
    def n2t(self,input_array):
        '''convert normalised target to actual target,
        also exponential function is applied by default
        (because target of engine should be handeled in exponential space)'''
        array = input_array*self.target_std + self.target_mean
        return np.exp(array)
    def t_t2n(self,input_array):
        '''convert actual target tensor to normalised target tensor,
        also natural log is applied by default
        (because target of engine should be handeled in exponential space)'''
        tensor = tf.math.log(input_array)
        return (tensor - self.target_mean) / self.target_std
    def t_n2t(self,input_array):
        '''convert normalised target tensor to actual target tensor,
        also exponential function is applied by default
        (because target of engine should be handeled in exponential space)'''
        tensor = input_array*self.target_std + self.target_mean
        return tf.math.exp(tensor)
    def t_f2n(self,input_array):
        '''convert actual feature to normalised feature'''
        return (input_array - self.feature_mean) / self.feature_std
    def t_n2f(self,input_array):
        '''convert normalised feature to actual feature'''
        return input_array*self.feature_std + self.feature_mean
    def load_dynamic_targets(self, data, data_dyn0):
        '''takes data and data_dyn0 and store infomation of dynamic targets'''
        self.dynamic_targets = []
        for name in data_dyn0['y0'].columns:
            self.dynamic_targets.append(list(data['y0'].columns).index(name.lower()))
        return self.dynamic_targets
    def d_t2n(self, array):
        full_target_length = self.target_mean.shape[1]
        empty_array = np.ones((array.shape[0], full_target_length))
        for i, j in enumerate(self.dynamic_targets):
            empty_array[:,j] = array[:,i]
        normalised_array = self.t2n(empty_array)
        return normalised_array[:,self.dynamic_targets]
    def d_n2t(self, array):
        full_target_length = self.target_mean.shape[1]
        empty_array = np.ones((array.shape[0], full_target_length))
        for i, j in enumerate(self.dynamic_targets):
            empty_array[:,j] = array[:,i]
        normalised_array = self.n2t(empty_array)
        return normalised_array[:,self.dynamic_targets]
    def save_norm(self,LOC, FILE, feature_column_names, target_column_names):
        writer = pd.ExcelWriter(LOC + FILE, engine='xlsxwriter')
        feature_names = ['feature_mean', 'feature_std']
        feature_values = [self.feature_mean, self.feature_std]
        target_names = ['target_mean', 'target_std']
        target_values = [self.target_mean, self.target_std]
        for name, value in zip(feature_names, feature_values):
            df = pd.DataFrame(value, columns=feature_column_names)
            df.to_excel(writer, sheet_name=name, index=False)
        for name, value in zip(target_names, target_values):
            df = pd.DataFrame(value, columns=target_column_names)
            df.to_excel(writer, sheet_name=name, index=False)
        writer.save()
    def a_t2n(self,input_array):
        '''convert actual feature to normalised feature'''
        return (input_array - self.target_mean) / self.target_std
    def a_n2t(self,input_array):
        '''convert normalised feature to actual feature'''
        return input_array*self.target_std + self.target_mean

    def load_norm(self,LOC, FILE):
        self.feature_mean = pd.read_excel(LOC+FILE, sheet_name='feature_mean').values
        self.feature_std = pd.read_excel(LOC+FILE, sheet_name='feature_std').values
        self.target_mean = pd.read_excel(LOC+FILE, sheet_name='target_mean').values
        self.target_std = pd.read_excel(LOC+FILE, sheet_name='target_std').values