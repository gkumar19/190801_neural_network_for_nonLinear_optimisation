# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:51:38 2020

@author: KGU2BAN
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import ExcelWriter
from plot import plot_tricontourf, plot_true_pred, plot_distribution, plot_covariance_matrix, plot_bar
from nn import make_model1, make_model2, nn_loss_model1, nn_model2_training
from helper import helper_dataframe, save_calibration_excel
from utils import load_excel, save_calibration_progress, save_cycle_progress
from dyn import dyn_0, plot_cycle, generate_dyns_from_folder, cycle_progress
from ml import ml_relevance_matrix_etr



#%% load data for model1
data_gdoe, N = load_excel('data.xlsx')
data, _ = load_excel('data.xlsx', 'dynamic_files/')
#Save N for future , saved as excel
N.save_norm('', 'N.xlsx', data['x0'].columns, data['y0'].columns)
N.load_norm('','N.xlsx')



#%%Visualise the datset
plt.figure()
sns.scatterplot('y1', 'y3', data=data_gdoe['xy0'], size='x2', hue='x1', linewidth=0, cmap='RdYlGn_r')
plt.figure()
plt.tricontourf(data['xy0']['x1'],
                data['xy0']['x2'],
                data['xy0']['y1'],20, cmap='RdYlGn_r')
plt.colorbar()
plot_distribution('target_distribution', 5, 5, data_gdoe['y0'].columns, data_gdoe['y0'].values)
#plot ditribution of target array after normalisation
plot_distribution('target_distribution_normalised', 5, 5, data_gdoe['y0'].columns, N.t2n(data_gdoe['y0'].values))
#plot distribution of input array, also can be helpfull in setting calibration constraints
plot_distribution('input_distribution_normalised', 3, 3, data_gdoe['x0'].columns, data_gdoe['x0'].values)
#plot the interdependencies among features
plot_covariance_matrix(data_gdoe['x0'].values, data_gdoe['x0'].columns)
#plot relevance between input and output arrays using random forrest with n-estimators = 50
relevance = ml_relevance_matrix_etr(data_gdoe['x0'].values, data_gdoe['y0'].values, 50)
plot_bar('relevance', 5, 5, data_gdoe['y0'].columns,data_gdoe['x0'].columns[0:], relevance[0:,:])



#%% making model1 : An Iterative process
tf.random.set_seed(1)
tf.keras.backend.clear_session()
model1 = make_model1(save=False, name=None)
r_evaluator = []
dat_dyn_list, r_values = generate_dyns_from_folder('dynamic_files/', model1 ,data,N, None)
for i in range(1):
    history_stage1 = model1.fit(N.f2n(data['x_trn'].values),
                                    N.t2n(data['y_trn'].values),
                                    epochs = 100,
                                    batch_size= 500,
                                    validation_data=(N.f2n(data['x_vld'].values),N.t2n(data['y_vld'].values)),
                                    validation_steps=50)
    dat_dyn_list, r_values = generate_dyns_from_folder('dynamic_files/', model1 ,data,N, dat_dyn_list)
    plot_true_pred(data_gdoe, model1, N)
    plot_true_pred(data, model1, N)
    r_evaluator.append(i*100)
    for key in dat_dyn_list.keys():
        data_dyn0 = dat_dyn_list[key]
        r_value = r_values[key]
        #since hc is learned over tail pipe
        data_dyn0['y0']['y2'] = data_dyn0['y1']['y2_t']
        plot_cycle(data_dyn0, 'y2', ['y0','pb0'],r_value)
        r_evaluator.append(r_value/(data_dyn0['pb1']['y3'].mean()/data_dyn0['v0']['v1'].mean()))
    #model_save = make_model1(save=True, name='model1.h5', trained_model = model1)
    del key, r_value, r_values



#%% dynamic predictions to check for sanity
dat_dyn_list, russ_values = generate_dyns_from_folder('dynamic_files/', model1 ,data,N, None)
dat_dyn_list, russ_values = generate_dyns_from_folder('dynamic_files/', model1 ,data,N, dat_dyn_list)
data_dyn0 = dat_dyn_list['dynamic_2']
russ_value = russ_values['dynamic_2']
plot_cycle(data_dyn0, 'y1', ['y0','pb0'],russ_value)
plot_cycle(data_dyn0, 'y4', ['y0','pb0'],russ_value)
plot_cycle(data_dyn0, 'y2', ['y0','pb0'],russ_value)
plot_cycle(data_dyn0, 'y6', ['pb0'],russ_value)
plot_cycle(data_dyn0, 'y3', ['pb0'],russ_value)



#%%model 2 calibration can start from here if model1 is already saved in past
data, N = load_excel('data.xlsx')
N.load_norm('','N.xlsx')
model1 = tf.keras.models.load_model('model1.h5')
model1.compile('adam', nn_loss_model1())
tf.keras.backend.clear_session()
#data_dyn0_list = [dyn_0(LOC,'cold.dat'), dyn_0(LOC,'jan_hot.dat')]
data_dyn0_list = [dyn_0('dynamic_files/dynamic_1/','1.dat'),
                  dyn_0('dynamic_files/dynamic_2/','2.dat')]
calibration_rpms = list(np.linspace(600/1000,3600/1000,5))
calibration_injs = list(np.linspace(2/1000,46/1000,3))
calibration_temp = list(np.linspace(20/1000,90/1000,5)) # this 5 is fixed, any change need change in model1
buffer_quantity = 2/1000
engine_map = helper_dataframe(calibration_rpms, calibration_injs, buffer_quantity,
                              data, data_dyn0_list, 'constraints.xlsx', calibration_temp)
del buffer_quantity



#%% model 2 training

model2 = make_model2(model1, N, engine_map)
model_weights = 'model2_weights.h5'
model_weights = None
model2, engine_map, calibration_progress = nn_model2_training(model2, engine_map, N, data,
                                                              model1, epochs=1000,model_weights=model_weights,
                                                              target_factors=np.array([1.1,1.15,1.15,1.15]))
def count_argmax(df):
    argmax = np.argmax(df.values, axis=-1)
    df_count = pd.DataFrame(argmax, columns=['argmax']).replace(list(range(len(df.columns))),list(df.columns))
    plt.figure()
    sns.countplot(data=df_count, x='argmax')
count_argmax(calibration_progress[0].iloc[-500:,:])
plt.figure()
calibration_progress[0].iloc[-500:,:].plot()
#model2.fit(np.zeros((1,9)), np.zeros((1, 129, 21)), epochs=10000) #model 2 can be trained over here also
#model2.save_weights('model2_weights.h5')
weights = model2.get_weights()



#%% plot the calibrated xs, ys
num_opts = int(engine_map.df.shape[0]/len(calibration_temp))
b = np.linspace(0,engine_map.df.shape[0],6).astype(int)
a = [(b[0], b[1]), (b[1], b[2]), (b[2], b[3]), (b[3], b[4]), (b[4], b[5])]
colorbarmin = engine_map.calibrations.values[:,1:-1].min(axis=0)
colorbarmax = engine_map.calibrations.values[:,1:-1].max(axis=0)
for i, t in enumerate(calibration_temp):
    plot_tricontourf(f'features at x3: {t}', 3, 5, engine_map.calibrations.columns[1:-1],
                     engine_map.engine_rpms.values[a[i][0]:a[i][1]], engine_map.injection_quantities[a[i][0]:a[i][1]],
                     engine_map.calibrations.values[a[i][0]:a[i][1]].T[1:-1], 'x1', 'x2',
                     calibration_rpms, calibration_injs, colorbarmin, colorbarmax)
colorbarmin = engine_map.emissions_and_boundaries.values.min(axis=0)
colorbarmax = engine_map.emissions_and_boundaries.values.max(axis=0)
for i, t in enumerate(calibration_temp):    
    plot_tricontourf(f'targets at x3: {t}', 4, 6, engine_map.emissions_and_boundaries.columns,
                     engine_map.engine_rpms.values[a[i][0]:a[i][1]], engine_map.injection_quantities[a[i][0]:a[i][1]],
                     engine_map.emissions_and_boundaries.values[a[i][0]:a[i][1]].T, 'x1', 'x2',
                     calibration_rpms, calibration_injs, colorbarmin, colorbarmax)



#%% saving data for animation simulation
#These files can be used to create animations
#anim to be referred for the same
save_calibration_excel(engine_map, '', 'saved_calibration.xlsx')
save_calibration_progress('',calibration_progress)
FILE_DAT = 'dynamic_files/dynamic_1/1.dat'
#[Fri_Jun_12_12_27_48_2020] is created automatically from above code
FILE_NUMPY = 'simulation_run/[Fri_Jun_12_14_09_07_2020]/calibration.npy' 
drive_pattern, model_input, model_output = cycle_progress(FILE_DAT, FILE_NUMPY, model1, N)
save_cycle_progress('',(drive_pattern, model_input, model_output))
df1 = pd.DataFrame(engine_map.factor, columns=['factor'])
df2 = pd.DataFrame([engine_map.avg_speed], columns=['avg_speed_km/h'])
df3 = (data_dyn0['v0']['time_s'].values[-1] - data_dyn0['v0']['time_s'].values[0])/3600
df3 = pd.DataFrame([df3], columns=['cycle_time_h'])
with ExcelWriter('simulation_run/[Fri_Jun_12_14_09_07_2020]/details.xlsx') as writer:
    df1.to_excel(writer, sheet_name='factor', index=False)
    df2.to_excel(writer, sheet_name='avg_speed', index=False)
    df2.to_excel(writer, sheet_name='cycle_time', index=False)



#%% Extra: Codes below maynot work as it is only kept for informations
from nn_old import nn_model_base_temp3, nn_model_base_temp4, nn_model_base_temp5, nn_model2_old
from dyn import nn_model_dyn, dyn_3, nn_dyn_train



#%% old model1 made by combining two models
model_base_e = nn_model_base_temp3(data_e['x_trn'].shape[1], 50,50,40, dropout=0)
model_base_a = nn_model_base_temp4()
model1 = nn_model_base_temp5(model_base_e, model_base_a, N_a) #N_a is normalier for model_base_a
model_properties = (9, 10, 11, 50, 50, 40, 0, 0, 0)
model2 = nn_model2_old(model1, model_properties, engine_map, N, '', 'constraints.xlsx')
model2.summary()
model2, engine_map, calibration_progress = nn_model2_training(model2, engine_map, N, data, model1, epochs=3500)



#%% addition of dynamics into the model prediction
N.load_dynamic_targets(data, data_dyn0)
model_dyn = nn_model_dyn(model1, 10, N, 3)
model_dyn.summary()
model_dyn = nn_dyn_train(data_dyn0, model_dyn, N, 'offset_compile')
model_dyn = nn_dyn_train(data_dyn0, model_dyn, N, 'offset_fit', epoch=500)
data_dyn0 = dyn_3(data_dyn0, model_dyn, N, kind='offset')
plot_cycle(data_dyn0, 'afs_mairpercyl', ['y0','pb0', 'po0'], plot_accumulated=False)
model_dyn = nn_dyn_train(data_dyn0, model_dyn, N, 'gru_compile')
model_dyn = nn_dyn_train(data_dyn0, model_dyn, N, 'gru_fit', epoch=1000)
data_dyn0 = dyn_3(data_dyn0, model_dyn, N, kind='gru')
plot_cycle(data_dyn0, 'afs_mairpercyl', ['y0','pb0','pg0'], plot_accumulated=False)
