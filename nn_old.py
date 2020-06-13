# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 23:00:54 2019

@author: KGU2BAN
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from constraints import constraints_load_excel, constraints_calibration_sheet, minmax_constructor
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K

#class to store the losses
class EvalCls():
    '''eval_ten goes into tensorflow graph, eval_val goes into callback'''
    def __init__(self,axis):
        self.eval_ten = tf.Variable(np.ones((1,axis))*3.14, dtype=tf.float32, trainable=True)
        self.eval_val = []

class CustomConstraint(tf.keras.constraints.Constraint):
    '''custom constraints, restricts the weights to exceed min_value and max value'''
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        desired = tf.keras.backend.clip(w, self.min_value, self.max_value)
        w = w * (desired / (tf.keras.backend.epsilon() + w))
        return w


#%%
def nn_loss2(target_conditions, boundary_conditions, boundary_condition_noise, N, engine_map, data, eval_obj1, eval_obj2, eval_obj3, eval_obj4, eval_obj5, qmin_m, qmin_c):
    '''
    This is an older version refer the new version in nn.py
    Four kinds of losses
    a. cycle integrated should be lower than the boundary limit strictly,
        no incentive for going lower
    b. low at all points separately, incentive for being as low as possible
    c. strictly low at all points separately,
        no incentive for going further low
    d. high incentive till certain value, low incentive while going even further
    '''
    
    def loss_function(y_true, y_pred):
        
        def n2t_3d(t):
            all_tensors = []
            for i in range(num_op_point):
                all_tensors.append(N.t_n2t(t[:,i,:]))
            return tf.stack(all_tensors, axis=1)
        
        def n2f_3d(t):
            all_tensors = []
            for i in range(num_op_point):
                all_tensors.append(N.t_n2f(t[:,i,:]))
            return tf.stack(all_tensors, axis=1)
        
        def arange_tensor(tensor, arange_sequence):
            '''sequence the tensor '''
            tensor_list = [tensor[:,i,None] for i in arange_sequence]
            concat = tf.concat(tensor_list, axis=1)
            return concat

        def a_loss_func(t, target_conditions, factor):
            factor_array = np.array(factor)[np.newaxis,:,np.newaxis]
            mul = tf.math.multiply(t, factor_array) #3d
            reduce_sum = tf.reduce_sum(mul, axis=1) #reduce_sum is 2-dimentional
            reduce_sum2 = reduce_sum/(target_conditions*engine_map.avg_speed) #target_conditions in g/km --> converted to g/km
            loss_sep = tf.math.multiply(tf.cast(reduce_sum2 > 1, reduce_sum2.dtype)*10, reduce_sum2) #2d (None,number of emission parameters)
            return loss_sep
        
        def b_loss_func(t, min_array, max_array):
            temp = (t - min_array)/(max_array-min_array)
            reduce_mean = tf.reduce_mean(temp, axis=1) #2d (None,number of bsfc type parameters)
            return reduce_mean
        
        def c_loss_func(t, boundary_array):
            temp = t/boundary_array
            temp2 = tf.math.multiply(tf.cast(temp > 1, temp.dtype)*10, temp) #3d
            reduce_max = tf.reduce_max(temp2, axis=1) #2d
            return reduce_max
        
        def d_loss_func(t, min_array, max_array, boundary_array):
            b_loss_temp = tf.abs((t - min_array)/(max_array-min_array)) #3d
            temp = t/boundary_array
            c_loss_temp = tf.math.multiply(tf.cast(temp > 1, temp.dtype)*10, temp) #3d
            minimum = tf.minimum(b_loss_temp, c_loss_temp)
            return tf.reduce_max(minimum, axis=1) #2d
        
        def e_loss_func(t, couple_rpm_axis, couple_inj_axis):
            #across rpm axis
            t_prev_rpm = arange_tensor(t, [i[0] for i in couple_rpm_axis])
            t_next_rpm = arange_tensor(t, [i[1] for i in couple_rpm_axis])
            t_diff_rpm = tf.abs(t_next_rpm-t_prev_rpm)
            t_diff_norm_rpm = t_diff_rpm/smoothness_factor[:,0]
            t_loss_rpm = tf.math.multiply(tf.cast(t_diff_norm_rpm > 1, t_diff_norm_rpm.dtype)*10, t_diff_norm_rpm) #3d
            #across inj axis
            t_prev_inj = arange_tensor(t, [i[0] for i in couple_inj_axis])
            t_next_inj = arange_tensor(t, [i[1] for i in couple_inj_axis])
            t_diff_inj = tf.abs(t_next_inj-t_prev_inj)
            t_diff_norm_inj = t_diff_inj/smoothness_factor[:,1]
            t_loss_inj = tf.math.multiply(tf.cast(t_diff_norm_inj > 1, t_diff_norm_inj.dtype)*10, t_diff_norm_inj) #3d
            #concat both
            t_loss = tf.concat([t_loss_rpm, t_loss_inj], axis=1) #3d
            reduce_max = tf.reduce_max(t_loss, axis=1) #2d
            return reduce_max
        
        smoothness_factor = np.array([[2/10,2/7],
                      [20/10,20/7],
                      [5000/10,5000/7],
                      [100/10,100/7],
                      [1600/10,1600/7]]) #RPM axis and Injection axis
            
        a_num = 4
        b_num = 1
        c_num = 12
        d_num = 4
        num_op_point = engine_map.df.shape[0]
        
        minmax = minmax_constructor(engine_map.df.iloc[:,0].values, engine_map.df.iloc[:,1].values, 750,7.5, data)
        min_array = minmax['min'].values
        max_array = minmax['max'].values
        
        
        y_pred_output = y_pred[:,:,:21]
        y_pred_input = y_pred[:,:,21:]
        
        y_pred_real = n2t_3d(y_pred_output)
        y_pred_split = tf.split(y_pred_real, num_or_size_splits=[a_num,b_num,c_num,d_num], axis=-1)
        
        y_pred_input_real = n2f_3d(y_pred_input)
        y_pred_input_split = tf.split(y_pred_input_real, num_or_size_splits=[3,5,1], axis=-1) #3 --> rpm,inj,ctemp,1 -->p0
        

        a_loss = a_loss_func(y_pred_split[0], target_conditions, engine_map.factor)
        b_loss = b_loss_func(y_pred_split[1], min_array[:,a_num:a_num+b_num], max_array[:,a_num:a_num+b_num]) * 3
        c_loss = c_loss_func(y_pred_split[2], boundary_conditions)
        d_loss = d_loss_func(y_pred_split[3], min_array[:,17:21], max_array[:,17:21], boundary_condition_noise)
        e_loss = e_loss_func(y_pred_input_split[1], engine_map.couple_rpm_axis, engine_map.couple_inj_axis)
        
        #save the losses during the training
        eval_a_loss = eval_obj1.eval_ten.assign(a_loss)
        eval_b_loss = eval_obj2.eval_ten.assign(b_loss)
        eval_c_loss = eval_obj3.eval_ten.assign(c_loss)
        eval_d_loss = eval_obj4.eval_ten.assign(d_loss)
        eval_e_loss = eval_obj5.eval_ten.assign(e_loss)
        
        with tf.control_dependencies([eval_a_loss, eval_b_loss, eval_c_loss, eval_d_loss, eval_e_loss]):
            concat = tf.concat([a_loss, b_loss, c_loss, d_loss, e_loss], axis=-1)
            return tf.reduce_max(concat, axis=-1) #1dimentional
        
    return loss_function


def nn_model_base(num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3):
    '''two parallel models working at same time'''
    X1 = tf.keras.layers.Input(shape = (num_features,),name = 'input',dtype = 'float32')
    
    #First branch of DNN connecting to num_targets1 targets
    X2 = tf.keras.layers.Dense(nodes1, activation = 'tanh',use_bias = True, name='x2')(X1)
    X3 = tf.keras.layers.Dropout(drop1, name='x3')(X2)
    X4 = tf.keras.layers.Dense(nodes2, activation = 'tanh',use_bias = True, name='x4')(X3)
    X5 = tf.keras.layers.Dropout(drop2, name='x5')(X4)
    X6 = tf.keras.layers.Dense(nodes3, activation = 'tanh',use_bias = True, name = 'x6')(X5)
    X7 = tf.keras.layers.Dropout(drop3, name='x7')(X6)
    X8 = tf.keras.layers.Dense(num_targets1, activation = 'linear', name='x8')(X7)
    
    #First branch of DNN connecting to num_targets1 targets
    Y2 = tf.keras.layers.Dense(nodes1, activation = 'tanh',use_bias = True, name='y2')(X1)
    Y3 = tf.keras.layers.Dropout(drop1, name='y3')(Y2)
    Y4 = tf.keras.layers.Dense(nodes2, activation = 'tanh',use_bias = True, name='y4')(Y3)
    Y5 = tf.keras.layers.Dropout(drop2, name='y5')(Y4)
    Y6 = tf.keras.layers.Dense(nodes3, activation = 'tanh',use_bias = True, name='y6')(Y5)
    Y7 = tf.keras.layers.Dropout(drop3, name='y7')(Y6)
    Y8 = tf.keras.layers.Dense(num_targets2, activation = 'linear', name='y8')(Y7)
    
    #concatenating both the branches of DNNs and creating the model
    output = tf.keras.layers.concatenate([X8, Y8], axis = 1, name='outputs')
    model = tf.keras.Model(inputs=X1, outputs=output)
    model_properties = (num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3)
    
    return model, model_properties

def nn_model_base_temp(num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3):
    '''two parallel models working at same time'''
    l_input = tf.keras.layers.Input(shape = (num_features,),name = 'l_input',dtype = 'float32')
    
    #branch of nn for nox, hc, co and afs
    l_nhca0 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_nhca0')(l_input)
    l_nhca0_d = tf.keras.layers.Dropout(0.2, name='l_nhca0_d')(l_nhca0)
    l_nhca1 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_nhca1')(l_nhca0_d)
    l_nhca1_d = tf.keras.layers.Dropout(0, name='l_nhca1_d')(l_nhca1)
    l_nhca2 = tf.keras.layers.Dense(40, activation = 'tanh',use_bias = True, name = 'l_nhca2')(l_nhca1_d)
    l_nhca2_d = tf.keras.layers.Dropout(0, name='l_nhca2_d')(l_nhca2)
    l_nhca_f = tf.keras.layers.Dense(4, activation = 'linear', name='l_nhca_f')(l_nhca2_d)
    
    #concatenate nox, hc, co and afs with the input to create NN for further points
    l_concat0 = tf.keras.layers.concatenate([l_input, l_nhca_f], axis=1, name='l_concat0')
    
    #branch of NN for rest of the points
    l_extra0 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_extra0')(l_concat0)
    l_extra0_d = tf.keras.layers.Dropout(0.2, name='l_extra0_d')(l_extra0)
    l_extra1 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_extra1')(l_extra0_d)
    l_extra1_d = tf.keras.layers.Dropout(0, name='l_extra1_d')(l_extra1)
    l_extra2 = tf.keras.layers.Dense(40, activation = 'tanh',use_bias = True, name='l_extra2')(l_extra1_d)
    l_extra2_d = tf.keras.layers.Dropout(0, name='l_extra2_d')(l_extra2)
    l_extra_f = tf.keras.layers.Dense(15, activation = 'linear', name='l_extra_f')(l_extra2_d)
    
    #concatenate the above layers
    l_concat1 = tf.keras.layers.concatenate([l_nhca_f, l_extra_f], axis = 1, name='l_concat1')
    
    #arange the tensor in right order
    def arange_tensor(tensor):
        '''
        change of sequence from --> to
        nox_g/h to nox_g/h
        hc_g/h to hc_g/h
        russ_g/h to co_g/h
        co_g/h to afs_mairpercyl
        bsfc_g/kwh to russ_g/h
        smoke to bsfc_g/kwh
        afs_mairpercyl to smoke
        t3_c to t3_c
        pzmax_bar to pzmax_bar
        toel_c to toel_c
        poel_bar to poel_bar
        p4a1abs_mbar to p4a1abs_mbar
        dpic_mbar to dpic_mbar
        t2_i1_c to t2_i1_c
        turb_spd_/min to turb_spd_/min
        p2_l1abs_mbar to p2_l1abs_mbar
        lambda to lambda
        lambda_inv to lambda_inv
        ger_inm_db to ger_inm_db

        '''
        arange_sequence = [0,1,3,6,2,4,5,7,8,9,10,11,12,13,14,15,16,17,18]
        tensor_list = [tensor[:,i,None] for i in arange_sequence]
        concat = tf.concat(tensor_list, axis=1)
        return concat
    l_output = tf.keras.layers.Lambda(arange_tensor, name='l_output')(l_concat1)
    
    model = tf.keras.Model(inputs=l_input, outputs=l_output)
    model_properties = (num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3)
    
    return model, model_properties

def nn_model_base_temp2(num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3):
    '''two parallel models working at same time'''
    l_input = tf.keras.layers.Input(shape = (num_features,),name = 'l_input',dtype = 'float32')
    
    #branch of nn for nox, hc, co and afs
    l_nhca0 = tf.keras.layers.Dense(10, activation = 'tanh',use_bias = True, name='l_nhca0')(l_input)
    l_nhca0_d = tf.keras.layers.Dropout(0.2, name='l_nhca0_d')(l_nhca0)
    l_nhca1 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l_nhca1')(l_nhca0_d)
    l_nhca2 = tf.keras.layers.Dense(20, name='l_nhca12')(l_nhca1)
    l_nhca_f = tf.keras.layers.Dense(1, activation = 'linear', name='l_nhca_f')(l_nhca2)
    
    #concatenate afs with the input to create NN for further points
    l_concat0 = tf.keras.layers.concatenate([l_input, l_nhca_f], axis=1, name='l_concat0')
    
    #branch of NN for rest of the points
    l_extra0 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_extra0')(l_concat0)
    l_extra0_d = tf.keras.layers.Dropout(0.2, name='l_extra0_d')(l_extra0)
    l_extra1 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l_extra1')(l_extra0_d)
    l_extra1_d = tf.keras.layers.Dropout(0, name='l_extra1_d')(l_extra1)
    l_extra2 = tf.keras.layers.Dense(40, activation = 'tanh',use_bias = True, name='l_extra2')(l_extra1_d)
    l_extra2_d = tf.keras.layers.Dropout(0, name='l_extra2_d')(l_extra2)
    l_extra_f = tf.keras.layers.Dense(18, activation = 'linear', name='l_extra_f')(l_extra2_d)
    
    #concatenate the above layers
    l_concat1 = tf.keras.layers.concatenate([l_nhca_f, l_extra_f], axis = 1, name='l_concat1')
    
    #arange the tensor in right order
    def arange_tensor(tensor):
        '''
        send the afs from 1st to 7th position, i = 0 to i = 6

        '''
        arange_sequence = [1,2,3,4,5,6,0,7,8,9,10,11,12,13,14,15,16,17,18]
        tensor_list = [tensor[:,i,None] for i in arange_sequence]
        concat = tf.concat(tensor_list, axis=1)
        return concat
    l_output = tf.keras.layers.Lambda(arange_tensor, name='l_output')(l_concat1)
    
    model = tf.keras.Model(inputs=l_input, outputs=l_output)
    model_properties = (num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3)
    
    return model, model_properties


def nn_model_base_temp3(num_x, num1=50, num2=50, num3=40, dropout=0.2):
    l_i = tf.keras.layers.Input(shape = (num_x,),name = 'l_input',dtype = 'float32')
    l_e0 = tf.keras.layers.Dense(num1, activation = 'tanh',use_bias = True, name='l_e0')(l_i) # 50
    l_e0_d = tf.keras.layers.Dropout(dropout, name='l_e0_d')(l_e0)
    l_e1 = tf.keras.layers.Dense(num2, activation = 'tanh',use_bias = True, name='l_e1')(l_e0_d) #50
    l_e1_d = tf.keras.layers.Dropout(dropout, name='l_e1_d')(l_e1)
    l_e2 = tf.keras.layers.Dense(num3, activation = 'tanh',use_bias = True, name='l_e2')(l_e1_d) #40
    l_e2_d = tf.keras.layers.Dropout(0, name='l_e2_d')(l_e2)
    l_e_f = tf.keras.layers.Dense(20, activation = 'linear', name='l_e_f')(l_e2_d)
    model_e = tf.keras.Model(inputs=l_i, outputs=l_e_f)
    return model_e

def nn_model_base_temp4(dropout=0.2):
    l_i = tf.keras.layers.Input(shape = (9,),name = 'l_input',dtype = 'float32')
    l_e0 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l_e0')(l_i)
    l_e0_d = tf.keras.layers.Dropout(dropout, name='l_e0_d')(l_e0)
    l_e1 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l_e1')(l_e0_d)
    l_e1_d = tf.keras.layers.Dropout(0, name='l_e1_d')(l_e1)
    l_e2 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l_e2')(l_e1_d)
    l_e2_d = tf.keras.layers.Dropout(0, name='l_e2_d')(l_e2)
    l_e_f = tf.keras.layers.Dense(1, activation = 'linear', name='l_e_f')(l_e2_d)
    model_a = tf.keras.Model(inputs=l_i, outputs=l_e_f)
    return model_a

def nn_model_base_temp5(model_base_e, model_base_a, N_a):
    def manipulate_air(t):
        temp1 = t*N_a.target_std + N_a.target_mean
        temp2 = tf.math.abs(temp1)
        temp3 = (tf.math.log(temp2) - N_a.target_mean) / N_a.target_std
        return temp3
    def chop_tensor(t):
        #return t[:,0:8]
        return t[...,0:8]
        
    
    #air model
    l1_i = tf.keras.layers.Input(shape = (9,),name = 'l1_input',dtype = 'float32')
    l1_e0 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l1_e0')(l1_i)
    l1_e1 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l1_e1')(l1_e0)
    l1_e2 = tf.keras.layers.Dense(20, activation = 'tanh',use_bias = True, name='l1_e2')(l1_e1)
    l1_e_f = tf.keras.layers.Dense(1, activation = 'linear', name='l1_e_f')(l1_e2)
    
    chopped_input_layer = tf.keras.layers.Lambda(chop_tensor, name='chop_layer')(l1_i)
    #input_layer_for_model_base_e = tf.keras.layers.concatenate([chopped_input_layer, l1_e_f], axis = 1, name='input_layer_for_model_base_e')
    input_layer_for_model_base_e = tf.keras.layers.Concatenate(axis=-1, name='input_layer_for_model_base_e_revised')([chopped_input_layer, l1_e_f])
    
    #emission model
    l0_e0 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l0_e0')(input_layer_for_model_base_e)
    l0_e1 = tf.keras.layers.Dense(50, activation = 'tanh',use_bias = True, name='l0_e1')(l0_e0)
    l0_e2 = tf.keras.layers.Dense(40, activation = 'tanh',use_bias = True, name='l0_e2')(l0_e1)
    l0_e_f = tf.keras.layers.Dense(20, activation = 'linear', name='l0_e_f')(l0_e2)
    
    output_layer_wo_air = l0_e_f
    air_layer = tf.keras.layers.Lambda(manipulate_air, name='lambda')(l1_e_f)
    output_layer_w_air = tf.keras.layers.concatenate([output_layer_wo_air, air_layer], axis = -1, name='output_layer_w_air')
    model_base_o = tf.keras.models.Model(inputs=l1_i, outputs=output_layer_w_air)
    
    for from_name, to_name in zip(['l_e0', 'l_e1', 'l_e2', 'l_e_f'], ['l0_e0', 'l0_e1', 'l0_e2', 'l0_e_f']):
        from_weights = model_base_e.get_layer(from_name).get_weights()
        model_base_o.get_layer(to_name).set_weights(from_weights)
    
    for from_name, to_name in zip(['l_e0', 'l_e1', 'l_e2', 'l_e_f'], ['l1_e0', 'l1_e1', 'l1_e2', 'l1_e_f']):
        from_weights = model_base_a.get_layer(from_name).get_weights()
        model_base_o.get_layer(to_name).set_weights(from_weights)
    
    
    '''
    input_layer_for_model_base_e = tf.keras.layers.concatenate([model_base_a.input[:,:-1], model_base_a.output], axis = 1, name='input_layer_for_model_base_e')
    output_layer_wo_air = model_base_e(input_layer_for_model_base_e)
    air_layer = tf.keras.layers.Lambda(manipulate_air, name='lambda')(model_base_a.output)
    output_layer_w_air = tf.keras.layers.concatenate([output_layer_wo_air, air_layer], axis = 1, name='output_layer_w_air')
    model_base_o = tf.keras.models.Model(inputs=model_base_a.input, outputs=output_layer_w_air)
    '''
    return model_base_o


#%%

def nn_model_base2(num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3):
    def x1_lambda_layer(x1):
        #temp = x1[:,4, None] / x1[:,0, None]
        #concat = tf.concat([x1[:,:4], temp, x1[:,4:]], axis=1)
        return x1
    def y_lambda_layer(tensors_list):
        x1 = tensors_list[0]
        y = tensors_list[1]
        #inj = x1[:,1, None]
        #rpm = x1[:,0, None]
        #temp = y[:,:4] * inj * rpm
        temp0 = tf.math.pow(2.0, y[:,0, None])
        temp1 = tf.math.pow(3.0, y[:,1, None])
        temp2 = tf.math.pow(4.0, y[:,2, None])
        temp3 = tf.math.pow(5.0, y[:,3, None])
        
        concat = tf.concat([temp0, temp1, temp2, temp3, y[:,4:]], axis=1)
        return concat
    '''two parallel models working at same time'''
    X1 = tf.keras.layers.Input(shape = (num_features,),name = 'input',dtype = 'float32')
    X1_LAMBDA = tf.keras.layers.Lambda(x1_lambda_layer, name='input_lambda1')(X1)
    
    #First branch of DNN connecting to num_targets1 targets
    X2 = tf.keras.layers.Dense(nodes1, activation = 'tanh',use_bias = True, name='x2')(X1_LAMBDA)
    X3 = tf.keras.layers.Dropout(drop1, name='x3')(X2)
    X4 = tf.keras.layers.Dense(nodes2, activation = 'tanh',use_bias = True, name='x4')(X3)
    X5 = tf.keras.layers.Dropout(drop2, name='x5')(X4)
    X6 = tf.keras.layers.Dense(nodes3, activation = 'tanh',use_bias = True, name = 'x6')(X5)
    X7 = tf.keras.layers.Dropout(drop3, name='x7')(X6)
    X8 = tf.keras.layers.Dense(num_targets1, activation = 'linear', name='x8')(X7)
    
    #First branch of DNN connecting to num_targets1 targets
    Y2 = tf.keras.layers.Dense(nodes1, activation = 'tanh',use_bias = True, name='y2')(X1_LAMBDA)
    Y3 = tf.keras.layers.Dropout(drop1, name='y3')(Y2)
    Y4 = tf.keras.layers.Dense(nodes2, activation = 'tanh',use_bias = True, name='y4')(Y3)
    Y5 = tf.keras.layers.Dropout(drop2, name='y5')(Y4)
    Y6 = tf.keras.layers.Dense(nodes3, activation = 'tanh',use_bias = True, name='y6')(Y5)
    Y7 = tf.keras.layers.Dropout(drop3, name='y7')(Y6)
    Y8 = tf.keras.layers.Dense(num_targets2, activation = 'linear', name='y8')(Y7)
    
    #concatenating both the branches of DNNs and creating the model
    output = tf.keras.layers.concatenate([X8, Y8], axis = 1, name='outputs')
    output_lambda = tf.keras.layers.Lambda(y_lambda_layer, name='output_lambda')([X1, output])
    model = tf.keras.Model(inputs=X1, outputs=output_lambda)
    model_properties = (num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3)
    
    return model, model_properties


def nn_model2_old(model1, model_properties, engine_map, N, LOC, CONSTRAINT_FILE):
    '''create a net of model'''
    #parameters specific to second stage of model
    (num_features,num_targets1,num_targets2, nodes1,nodes2,nodes3, drop1,drop2,drop3) = model_properties
    num_operating_points = engine_map.df.shape[0]
    model_input_list = [None] * (num_operating_points * num_features)
    model_output_list = [None] * num_operating_points
    reshape_list_output = [None] * num_operating_points
    model_cal_list = [None] * num_operating_points
    reshape_list_input = [None] * num_operating_points
    
    #setting the constraints globally
    #load information from constaints file
    LOC, CONSTRAINT_FILE = 'C:/Work/Common_Resources/MyPythoFolder/AI/modules/', 'constraints_v2.xlsx'
    constraints_loaded = constraints_load_excel(LOC, CONSTRAINT_FILE)
    calibration_max_sheet, calibration_min_sheet = constraints_calibration_sheet(engine_map.df, constraints_loaded)
    
    constraint_list = []
    for i in range(num_operating_points):
        for j in range(num_features):
            min_calibration = calibration_min_sheet.iloc[i,:num_features].values
            max_calibration = calibration_max_sheet.iloc[i,:num_features].values
            min_weights = np.ravel(N.f2n(min_calibration))
            max_weights = np.ravel(N.f2n(max_calibration))
            customconstraint = CustomConstraint(min_weights[j],max_weights[j])
            constraint_list.append(customconstraint)
        
    #setting input list and trainable layer list
    for i in range(num_operating_points*num_features):
        model_input_list[i] = tf.keras.layers.Input(shape = (1,),name = f"i{i}",
                              dtype = 'float32')
    
    model_temp = model1
    for layer in model1.layers:
        layer.trainable = False
    
    #join layers
    for i in range(num_operating_points):
        model_trainable_list = []
        for j in range(num_features):
            if j <= 1:
                trainable = False
                initializer = 'ones'
                model_trainable_list.append(tf.keras.layers.Dense(1, activation = 'linear', use_bias = False,
                                                 name = f"t{i*num_features+j}", kernel_initializer=initializer,
                                                 kernel_constraint = None, trainable=trainable)(model_input_list[i*num_features+j]))
            else:
                trainable = True
                initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) #stddev: 0.05
                model_trainable_list.append(tf.keras.layers.Dense(1, activation = 'linear', use_bias = False,
                                                     name = f"t{i*num_features+j}", kernel_initializer=initializer,
                                                     kernel_constraint = constraint_list[i*num_features+j], trainable=trainable)(model_input_list[i*num_features+j]))
                
            #print(f'total calibration values: {i*num_features+j}/{num_operating_points * num_features}')
        
        trainable_layer = tf.keras.layers.concatenate(model_trainable_list, axis = 1, name=f'tc{i}')
        model_output_list[i] = model_temp(trainable_layer)
        model_cal_list[i] = trainable_layer
        print(f'calibration_points: {i+1}/{num_operating_points}')
        reshape_list_output[i] = tf.keras.layers.Reshape((1,num_targets1+num_targets2), name=f'reshape_output{i}')(model_output_list[i])
        reshape_list_input[i] = tf.keras.layers.Reshape((1,num_features), name=f'reshape_input{i}')(model_cal_list[i])
    model_output = tf.keras.layers.concatenate(reshape_list_output , axis=1, name='final_concat_output')
    model_input = tf.keras.layers.concatenate(reshape_list_input , axis=1, name='final_concat_input')
    
    model_output_input = tf.keras.layers.concatenate([model_output, model_input], axis=2, name='final')
    #create model
    model2 = tf.keras.Model(inputs=model_input_list, outputs=model_output_input)
    
    return model2


def nn_model2_training(model2, engine_map, N, data, model_base_o, epochs=100, model_weights=None):
    '''train model2'''
    if model_weights != None:
        model2.load_weights(model_weights)
    df_target_conditions, df_boundary_conditions = data['n1'], data['b1']
    #parameters specific to second stage of model
    num_features = engine_map.num_calibrations + 2
    num_operating_points = engine_map.df.shape[0]
    num_emissions, num_boundaries = data['n0'].shape[1], data['b0'].shape[1]
    num_targets = num_emissions + num_boundaries
    
    #setting up the input for the training purpose:
    input_array = np.ones((num_operating_points,num_features))
    input_array[:,:2] = N.f2n(engine_map.df.values[:,:num_features])[:,:2]
    input_list = list(np.ravel(input_array).reshape(-1,1))
    #output_list = [np.zeros((1, num_targets))] * num_operating_points
    output_list = np.zeros((1, num_targets))
    
    #Define loss
    eval_obj1 = EvalCls(4)
    eval_obj2 = EvalCls(1)
    eval_obj3 = EvalCls(12)
    eval_obj4 = EvalCls(4)
    eval_obj5 = EvalCls(5)
    saved_weights_list = []
    saved_layer_output = []
    input_to_callback = []
    for input_value in input_list:
        input_to_callback.append(np.ones((1,1))*input_value[0])
    
    # define the qmin
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(np.array(data['q1'].columns)[:,None], data['q1'].values.T)
    qmin_m = reg.coef_
    qmin_c = reg.intercept_
    
    loss_list = nn_loss2(df_target_conditions.iloc[:,:-1].values,
                         df_boundary_conditions.iloc[:,:-4].values,
                         df_boundary_conditions.iloc[:,-4:].values, N, engine_map, data,
                         eval_obj1, eval_obj2, eval_obj3, eval_obj4, eval_obj5, qmin_m, qmin_c)
    #callbacks
    patience = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=300)
    
    
    class EvalCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            eval_obj1.eval_val.append(eval_obj1.eval_ten.numpy())
            eval_obj2.eval_val.append(eval_obj2.eval_ten.numpy())
            eval_obj3.eval_val.append(eval_obj3.eval_ten.numpy())
            eval_obj4.eval_val.append(eval_obj4.eval_ten.numpy())
            eval_obj5.eval_val.append(eval_obj5.eval_ten.numpy())

    class WeightsSaver(tf.keras.callbacks.Callback):    
        def on_train_batch_end(self, batch, logs=None):
            saved_weights_list.append(self.model.get_weights())
    
    class LayerSaver(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            input1 = self.model.input               # input placeholder
            saved_layer = ['tc0']
            output1 = [layer.output for layer in self.model.layers if layer.name in saved_layer]# all layer outputs
            fun = tf.keras.backend.function(input1,output1)# evaluation function
            saved_layer_output.append(fun(input_to_callback))
    
    evalcallback = EvalCallback()
    weightsaver = WeightsSaver()
    layersaver = LayerSaver()
    callback_list = [evalcallback, weightsaver]
    
    #train the model
    optimizer = tf.keras.optimizers.Adam(lr = 1.2*1e-2, decay = 1e-8)
    model2.compile(optimizer=optimizer ,loss = loss_list)
    
    #model2.fit(input_list, output_list, epochs=epochs, callbacks=callback_list)
    model2.fit(np.zeros((1,9)), np.zeros((1, 129, 21)), epochs=5000, callbacks=callback_list)
    
    a_loss_stacked = np.vstack(eval_obj1.eval_val)
    b_loss_stacked = np.vstack(eval_obj2.eval_val)
    c_loss_stacked = np.vstack(eval_obj3.eval_val)
    d_loss_stacked = np.vstack(eval_obj4.eval_val)
    e_loss_stacked = np.vstack(eval_obj5.eval_val)
    
    
    #save the calibation
    #weights = model2.get_weights()
    #cropped_weights_list = [i[0][0] for i in weights[:num_operating_points*num_features]]
    #normalized_calibration_array = np.array(cropped_weights_list).reshape(num_operating_points,num_features)
    #feature_array = N.n2f(normalized_calibration_array)
    #engine_map.df.iloc[:,2:num_features] = feature_array[:,2:num_features]
    
    #save the targets
    #targets = model2.predict(input_list)[0]
    #target_array = targets[:,:21]
    #engine_map.df.iloc[:,-num_targets:] = N.n2t(target_array)[:,-num_targets:]
    
    predict = model2.predict(np.zeros((1,9)))[0]
    calibration_weights,  target_weights = predict[:,-num_features:], predict[:,:-num_features]
    engine_map.df.iloc[:,:num_features] = N.n2f(calibration_weights)
    engine_map.df.iloc[:,num_features:] = N.n2t(target_weights)
    engine_map.calibration_factors = model2.get_weights()[1]
    
    engine_map.refresh()
    #pd.DataFrame(history.history).iloc[10:,1:].plot()
    loss_names = list(engine_map.emissions_and_boundaries.columns) + list(engine_map.calibrations.columns[1:-1])
    loss_stacked = (a_loss_stacked, b_loss_stacked, c_loss_stacked, d_loss_stacked, e_loss_stacked)
    loss_combined = pd.DataFrame(np.concatenate(loss_stacked, axis=1), columns=loss_names)
    
    #saved weights to be converted to output
    calibration_progress = np.zeros((epochs, num_operating_points, num_features))
    target_progress = np.zeros((epochs, num_operating_points, num_targets))
    #for epoch_num in range(epochs):
    #    weights = saved_weights_list[epoch_num]
    #    cropped_weights_list = [i[0][0] for i in weights[:num_operating_points*num_features]]
    #    normalized_calibration_array = np.array(cropped_weights_list).reshape(num_operating_points,num_features)
    #    calibration_progress[epoch_num,:,2:num_features] = N.n2f(normalized_calibration_array)[:,2:num_features]
    #    calibration_progress[epoch_num,:,0:2] = engine_map.df.iloc[:,0:2]
    #    target_progress[epoch_num,:,:] = nn_predict(model_base_o, N, calibration_progress[epoch_num,:,:])
        
    return model2, engine_map, (loss_combined,calibration_progress, target_progress)

def nn_predict(model, N, feature_array, special=False):
    if special == False:
        return N.n2t(model.predict(N.f2n(feature_array)))
    else:
        return N.a_n2t(model.predict(N.f2n(feature_array)))


#%%transfer learning [codes may not work]

#import tensorflow.keras.backend as K
#model2, _ = nn_model(8,4,11, 50,50,40, 0,0,0)
#model2.set_weights(model1.get_weights())
#def tl_loss():
#    def loss_function(y_true, y_pred):
#        tensor = y_pred - y_true
#        return K.mean(K.square(tensor[:,:4]), axis=-1)
#    return loss_function
#
##Save weights for comparision
#model1_weights = model2.get_weights()
#for layer in model2.layers:
#    layer.trainable = False
#
#e1 = model2.get_layer('x6').output
#e2 = model2.get_layer('y6').output
#ec = tf.keras.layers.concatenate([e1, e2], axis=1)
#
#x = model2.output
#extra_layer1 = tf.keras.layers.Dense(40, activation='tanh')(ec)
#extra_layer2 = tf.keras.layers.Dense(15, activation='tanh')(extra_layer1)
#model_fml = tf.keras.models.Model(inputs=model2.input, outputs=extra_layer2)
#
## here the excel file has been loaded
#optimizer1 = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-6)
#model_fml.compile(optimizer=optimizer1 ,loss=tl_loss())
#history_stage1 = model_fml.fit(N.f2n(x_trn.values),
#                            N.t2n(y_trn.values),
#                            epochs = 50,
#                            steps_per_epoch= 200,
#                            validation_data=(N.f2n(x_vld.values),N.t2n(y_vld.values)),
#                            validation_steps=50)
#model_fml_weights = model_fml.get_weights()


