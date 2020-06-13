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
from tensorflow.keras.layers import Dense, Input, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential, Model

#class to store the losses
class EvalCls():
    '''eval_ten goes into tensorflow graph, eval_val goes into callback'''
    def __init__(self,axis):
        self.eval_ten = tf.Variable(np.ones((1,axis))*3.14, dtype=tf.float32, trainable=True)
        self.eval_val = []

#%%

def make_model1(save = False, name=None, trained_model=None):
    model1 = Sequential([Dense(50,input_shape= (9,), activation = 'tanh'),
                    Dropout(0.2),
                    Dense(50, activation = 'tanh'),
                    Dense(40, activation = 'tanh'),
                    Dense(21, activation = 'linear')])
    model1.compile('adam', nn_loss_model1())
    if save == True:
        model1 = Sequential([Dense(50, input_shape= (9,) ,activation = 'tanh'),
                             Dense(50, activation = 'tanh'),
                             Dense(40, activation = 'tanh'),
                             Dense(21, activation = 'linear')])
        model1.compile('adam', 'mse')
        model1.set_weights(trained_model.get_weights())
        print('model1 saved without the dropout!')
        tf.keras.models.save_model(model1,name)
    return model1

def make_model2(model1,  N, engine_map):
    #Apply Constraints on the my_kernel
    num_opnts = int(engine_map.df.shape[0]/5)
    class CustomConstraintMap(tf.keras.constraints.Constraint):
        '''custom constraints, restricts the weights to exceed min_value and max value'''
        def __init__(self, min_value, max_value):
            self.min_value = min_value
            self.max_value = max_value
        def __call__(self, w):
            desired = tf.clip_by_value(w, self.min_value, self.max_value)
            #w = w * (desired / (tf.keras.backend.epsilon() + w))
            return desired
    calibration_max_sheet = engine_map.calibration_max_sheet
    calibration_min_sheet = engine_map.calibration_min_sheet
    calibration_min_sheet = N.f2n(calibration_min_sheet.values[:,:9])
    calibration_max_sheet = N.f2n(calibration_max_sheet.values[:,:9])
    map_constraint = CustomConstraintMap(calibration_min_sheet[:num_opnts, 3:], calibration_max_sheet[:num_opnts, 3:])

    
    #Create the model
    class MyDenseLayer(tf.keras.layers.Layer):
        def __init__(self, model1):
            super(MyDenseLayer, self).__init__()
            self.model1 = model1
            self.e_i_c = tf.constant(calibration_max_sheet[:, :3], dtype='float32')
        def build(self, input_shape):
            #self.kernel = self.add_weight("my_kernel",shape=(129, 9), constraint=constraint)
            self.kernel1 = self.add_weight("my_kernel1",shape=(num_opnts, 9-3), constraint=map_constraint)
            self.kernel2 = self.add_weight("my_kernel2",shape=(num_opnts, 9-3), constraint=map_constraint)
            self.kernel3 = self.add_weight("my_kernel3",shape=(num_opnts, 9-3), constraint=map_constraint)
            self.kernel4 = self.add_weight("my_kernel4",shape=(num_opnts, 9-3), constraint=map_constraint)
            self.kernel5 = self.add_weight("my_kernel5",shape=(num_opnts, 9-3), constraint=map_constraint)
            
        def call(self, input_x):
            input_x = tf.expand_dims(input_x, axis=1)
            #x = self.kernel
            x1 = self.kernel1 # low temperature
            x2 = self.kernel2
            x3 = self.kernel3
            x4 = self.kernel4
            x5 = self.kernel5 #high temperature
            x = tf.concat([x1, x2, x3, x4, x5],axis=0) #2 dimentional
            x = tf.concat([self.e_i_c, x], axis=-1)
            
            x = tf.add(input_x, x) # (None, 1, 9) + (53,9) --> (None, 53, 9) broadcasting step
            target = self.model1(x) #This is a kind of layer broadcasting, prediction happens on last dimention in tf
            return tf.concat([target, x], axis=-1)
    for layer in model1.layers:
        layer.trainable = False
    layer = MyDenseLayer(model1)
    input_layer = Input(shape=(9,), name='input_model2')
    x = layer(input_layer)
    return Model(input_layer, x)

#%%
def nn_loss_model2(target_conditions, boundary_conditions, boundary_condition_noise, N, engine_map, data, eval_obj, qmin_m, qmin_c):
    '''
    Four kinds of losses
    a. cycle integrated should be lower than the boundary limit strictly,
        no incentive for going lower
    b. low at all points separately, incentive for being as low as possible
    c. strictly low at all points separately,
        no incentive for going further low
    d. high incentive till certain value, low incentive while going even further
    '''
    #Example: saving custom internal losses during the training
    #sess = tf.Session()
    #tf.keras.backend.set_session(sess)
    #modelx = tf.keras.models.Sequential([tf.keras.layers.Dense(10,activation='relu', input_shape=(10,)),
    #                                     tf.keras.layers.Dense(10,activation='relu')], name='temp_model')
    #class eval_cls():
    #    def __init__(self):
    #        self.eval_ten = tf.Variable(np.ones((1,))*3.14, dtype=tf.float32, trainable=True)
    #        self.eval_val = []
    #eval_obj = eval_cls()
    #def loss_func(y_true, y_pred):
    #    b_loss = tf.keras.backend.mean(y_true+1, axis=-1)
    #    eval_op1 = eval_obj.eval_ten.assign(b_loss)
    #    loss = tf.keras.backend.mean(y_true-y_pred, axis=-1)
    #    with tf.control_dependencies([eval_op1]):
    #        return loss + b_loss
    #class CustomCallback(tf.keras.callbacks.Callback):
    #    def on_train_batch_end(self, batch, logs=None):
    #        eval_obj.eval_val.append(eval_obj.eval_ten.numpy())
    #mycallback = CustomCallback()
    #inputx= np.ones((10,10))
    #outputx = np.ones((10,10)) * np.arange(0,10)[:,None]
    #modelx.compile(optimizer='adam', loss=loss_func)
    #history = modelx.fit(inputx, outputx, epochs=3, batch_size=10, shuffle=False, callbacks=[mycallback])
    #temp3 = np.vstack(eval_obj.eval_val)
    
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
        
        def f_loss_func(t,min_array, max_array):
            temp = ((2*(t - min_array))/(max_array-min_array)) - 1 # this is allowed to take value from -1 to 1
            temp1 = tf.math.multiply(tf.cast(temp < -1, t.dtype)*10, -temp)
            temp2 = tf.math.multiply(tf.cast(temp > 1, t.dtype)*10, temp)
            temp = tf.concat([temp1, temp2], axis=-1) #3d
            reduce_max = tf.reduce_max(temp, axis=1) #2d
            return reduce_max
            
        
        calibration_room = engine_map.calibration_max_sheet.max(axis=0) - engine_map.calibration_min_sheet.min(axis=0)
        calibration_room = calibration_room[3:3+5]
        calibration_window_rpm_axis = calibration_room/len(engine_map.calibration_rpms)
        calibration_window_inj_axis = calibration_room/len(engine_map.calibration_injs)
        smoothness_factor_rom_inj_axis = pd.concat([calibration_window_rpm_axis, calibration_window_inj_axis],axis=1).values
        smoothness_factor = smoothness_factor_rom_inj_axis * 4 #RPM axis and Injection axis
            
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
        #y_pred_input_split = tf.split(y_pred_input_real, num_or_size_splits=[3,5,1], axis=-1) #3 --> rpm,inj,ctemp,1 -->p0
        y_pred_input_split2 = tf.split(y_pred_input_real, num_or_size_splits=[3,6], axis=-1) #3 --> rpm,inj,ctemp

        a_loss = a_loss_func(y_pred_split[0], target_conditions, engine_map.factor)
        b_loss = b_loss_func(y_pred_split[1], min_array[:,a_num:a_num+b_num], max_array[:,a_num:a_num+b_num]) * 3 * 5
        c_loss = c_loss_func(y_pred_split[2], boundary_conditions)
        d_loss = d_loss_func(y_pred_split[3], min_array[:,17:21], max_array[:,17:21], boundary_condition_noise) * 0.3
        #e_loss = e_loss_func(y_pred_input_split[1], engine_map.couple_rpm_axis, engine_map.couple_inj_axis) * 0
        f_loss = f_loss_func(y_pred_input_split2[1], engine_map.calibration_min_sheet.values[:,3:9], engine_map.calibration_max_sheet.values[:,3:9])
                
        loss_concat = tf.concat([a_loss, b_loss, c_loss, d_loss, f_loss], axis=-1)
        #save the losses during the training
        eval_loss = eval_obj.eval_ten.assign(loss_concat)
        with tf.control_dependencies([eval_loss]):
            return tf.reduce_max(loss_concat, axis=-1) #1dimentional
    return loss_function

#%%

def nn_model2_training(model2, engine_map, N, data, model_base_o, epochs=100, model_weights=None, target_factors=1):
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
    
    #Define loss
    eval_obj = EvalCls(26-5+6*2)
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
    
    loss_list = nn_loss_model2(df_target_conditions.iloc[:,:-1].values*target_factors,
                         df_boundary_conditions.iloc[:,:-4].values,
                         df_boundary_conditions.iloc[:,-4:].values, N, engine_map, data,
                         eval_obj, qmin_m, qmin_c)
    #callbacks
    
    class EvalCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            eval_obj.eval_val.append(eval_obj.eval_ten.numpy())

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
            
    #saved weights to be converted to output
    calibration_progress = np.zeros((epochs, num_operating_points, num_features))
    target_progress = np.zeros((epochs, num_operating_points, num_targets))
    
    class SaveCalibration(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
                predict = model2.predict(np.zeros((1,9)))[0]
                calibration_weights,  target_weights = predict[:,-num_features:], predict[:,:-num_features]
                calibration_progress[epoch,:,:] = N.n2f(calibration_weights)
                target_progress[epoch,:,:] = N.n2t(target_weights)
    
    evalcallback = EvalCallback()
    weightsaver = WeightsSaver()
    layersaver = LayerSaver()
    savecalibration = SaveCalibration() #making the epoch slow
    callback_list = [evalcallback, savecalibration]
    #train the model
    optimizer = tf.keras.optimizers.Adam(lr = 0.1*3*1e-3, decay = 1e-8)
    model2.compile(optimizer=optimizer ,loss = loss_list)
    
    model2.fit(np.zeros((1,9)), np.zeros((1, 129, 21)), epochs=epochs, callbacks=callback_list)
    
    loss_stacked = np.vstack(eval_obj.eval_val)
    
    predict = model2.predict(np.zeros((1,9)))[0]
    calibration_weights,  target_weights = predict[:,-num_features:], predict[:,:-num_features]
    engine_map.df.iloc[:,:num_features] = N.n2f(calibration_weights)
    engine_map.df.iloc[:,num_features:] = N.n2t(target_weights)
    engine_map.calibration_factors = model2.get_weights()[1]
    
    engine_map.refresh()
    loss_names = list(engine_map.emissions_and_boundaries.columns) + list(engine_map.calibrations.columns[1:]) + list(engine_map.calibrations.columns[1:])
    loss_combined = pd.DataFrame(loss_stacked, columns=loss_names)
   
    return model2, engine_map, (loss_combined,calibration_progress, target_progress)


def nn_predict(model, N, feature_array, special=False):
    if special == False:
        return N.n2t(model.predict(N.f2n(feature_array)))
    else:
        return N.a_n2t(model.predict(N.f2n(feature_array)))


def nn_merge_predict(x_train, y_train, x_test, y_test, feature_column_list, target_column_list, model, N):
    df_check_train = pd.concat([pd.DataFrame(x_train, columns=feature_column_list),
                          pd.DataFrame(y_train, columns=target_column_list),
                          pd.DataFrame(N.n2t(model.predict(N.f2n(x_train))),columns='pred_'+target_column_list)],axis=1)
    df_check_train['type'] = 'train'
    
    df_check_test = pd.concat([pd.DataFrame(x_test, columns=feature_column_list),
                          pd.DataFrame(y_test, columns=target_column_list),
                          pd.DataFrame(N.n2t(model.predict(N.f2n(x_test))),columns='pred_'+target_column_list)],axis=1)
    df_check_test['type'] = 'test'
    
    df_check = pd.concat([df_check_train, df_check_test], axis=0)
    return df_check


def nn_all_cal_check(feature_array, model, N):
    '''returns: rpms, injs, prediction_array'''
    rpms = np.linspace(1000, 3500, 5)
    injs = np.linspace(40, 5, 5)
    rpms, injs = np.meshgrid(rpms, injs)
    rpms = np.ravel(rpms)
    injs = np.ravel(injs)
    prediction_array = np.zeros((rpms.shape[0],50000, model.output_shape[1]))
    minimum = np.min(feature_array,axis = 0)
    maximum = np.max(feature_array,axis = 0)
    input_array = np.zeros((50000,feature_array.shape[1]))
    for j in range(feature_array.shape[1]):
        input_array[:,j] = np.random.random_sample((50000,)) *(maximum[j]-minimum[j]) + minimum[j]
    for i in range(rpms.shape[0]):
        rpm = rpms[i]
        inj = injs[i]
        input_array[:,0] = rpm
        input_array[:,1] = inj
        prediction_array[i,:,:] = nn_predict(model, N, input_array)
    return rpms, injs, prediction_array


def nn_loss_model1():
    def loss(y_true, y_pred):  # pylint: disable=missing-docstring
        y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        loss = math_ops.abs(math_ops.squared_difference(y_pred, y_true))
        loss = K.mean(loss, axis=-1)
        return loss # shape (None,)
    return loss
