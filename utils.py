# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:51:06 2019

@author: KGU2BAN
"""
import matplotlib.pyplot as plt
import numpy  as np
from nn import nn_predict
from norm import NormalizationTensor
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
from glob import glob

def load_excel(FILE, FOLDER=None):
    data = pd.read_excel(FILE, sheet_name=None)
    #filter based on criteria
    filter_in = data['c0']['x8'] < 0.1
    for i in ['c0', 'e0', 'i0', 'n0', 'b0']:
        data[i] = data[i][filter_in]
        data[i].reset_index(inplace=True, drop=True)
    for i in ['c0']:
        data[i].drop(['x8'], inplace=True, axis=1)
    data['b0']['y20'].loc[data['b0']['y20'] == 0] = 0.000001 #remove zeros, else issue will be while taking log
    data['x0'] = pd.concat([data['e0'], data['i0'], data['c0']], axis=1)
    data['y0'] = pd.concat([data['n0'], data['b0']], axis=1)
    data['xy0'] = pd.concat([data['x0'], data['y0']], axis=1)
    #shuffle the dataframe, for test split use keras inbuilt functions
    data['x_trn'], data['x_vld'], data['y_trn'], data['y_vld'] = train_test_split(data['x0'], data['y0'], test_size=0.2, shuffle=True)
    
    for i in ['x_trn', 'x_vld', 'y_trn', 'y_vld']:
        data[i].reset_index(inplace=True, drop=True)
        
    #normalise the feature and target labels, and include logarithmic transformation
    N = NormalizationTensor(data['x_trn'].values, data['y_trn'].values)
    if FOLDER != None:
        dat = generate_dyns(FOLDER)
        # replacement from data_dyn0 --> data
        replacement_map = [(('x0','x1'),('x0','x1')),
                           (('x0','x2'),('x0','x2')),
                           (('x0','x3'),('x0','x3')),
                           (('x0','x4'),('x0','x4')),
                           (('x0','x5'),('x0','x5')),
                           (('x0','x6'),('x0','x6')),
                           (('y1','y21'),('y0','y21')),
                           (('x0','x7'),('x0','x7')),
                           (('x0','x10'),('x0','x10')),
                           (('x0','x9'),('x0','x9')),
                           (('y0','y1'),('y0','y1')),
                           (('y1','y2_t'),('y0','y2')),
                           (('y0','y4'),('y0','y4')),
                           (('y0','y19'),('y0','y19')),
                           (('y0','y20'),('y0','y20'))]
        
        roll = {'y1': -5, 'y2_t': -6, 'y4': -7, 'y19': -4}
        
        for test_name, data_dyn0 in dat.items():
            for ((a,b), (c,d)) in replacement_map:
                array = data_dyn0[a][b].values
                if b in roll.keys():
                    print(b)
                    array = np.roll(array, roll[b])
                array_new = np.zeros((1469,1))
                for i in range(1469):
                    array_new[i] = array[i*8]
                array_new = pd.DataFrame(array_new, columns=[d])
                if ~data[c].isna().any().any():
                    data[c] = data[c].append(array_new, sort=False)
                else:
                    data[c].iloc[-len(array_new):,data[c].columns.get_loc(d)] = array_new.values.flatten()
            for ((a,b), (c,d)) in replacement_map:
                data[c] = data[c].replace(np.nan,-1)
                data[c].reset_index(inplace=True, drop=True)
    data['x_trn'], data['x_vld'], data['y_trn'], data['y_vld'] = train_test_split(data['x0'], data['y0'], test_size=0.2, shuffle=True)
    return data, N


def generate_dyns(LOC):
    folders = sorted(glob(LOC+'/*'))
    folders = [file.replace('\\','/') for file in folders]
    from dyn import dyn_0, dyn_1
    dat_df = {}
    for folder in folders:
        folder_name = os.path.basename(folder)
        dat_file = glob(folder+'/*.dat')[0].replace('\\','/')
        asc_file = glob(folder+'/*.csv')[0].replace('\\','/')
        data_dyn0 = dyn_0(dat_file,'')
        data_dyn0 = dyn_1(asc_file,'', data_dyn0)
        dat_df[folder_name] = data_dyn0
    return dat_df

def save_calibration_progress(LOC, calibration_progress):
    time = timestamp().replace(':','_').replace(' ', '_')
    directory = LOC + 'simulation_run/'+time
    os.makedirs(directory)
    np.save(directory+'/calibration',calibration_progress[1])
    np.save(directory+'/targets',calibration_progress[2])
    calibration_progress[0].to_excel(directory+'/losses.xlsx')

def save_cycle_progress(LOC, calibration_progress):
    time = timestamp().replace(':','_').replace(' ', '_')
    directory = LOC + 'simulation_run/'+time
    os.makedirs(directory)
    np.save(directory+'/cycle_calibration',calibration_progress[1])
    np.save(directory+'/cycle_targets',calibration_progress[2])
    calibration_progress[0].to_excel(directory+'/cycle_pattern.xlsx')

def closest_operating_point(r, ij, operating_points):
    min_index = 0
    min_distance = float('inf')
    for i in range(operating_points.shape[0]):
        distance = np.square(operating_points[i,0]-r) + np.square(operating_points[i,1]-ij)
        if distance < min_distance:
            min_index = i
            min_distance = distance
    return operating_points[None,min_index,:]


def sweep(rpm, inj, operating_points, sweep_label, min_calibrations, max_calibrations, model, N):
    #takes rpm, inj, operating_points_array from df, etc, and generate calibration and pediction
    calibration_array = closest_operating_point(rpm, inj, operating_points)[:, :min_calibrations.shape[0]]
    calibration_array = np.tile(calibration_array, (50,1))
    calibration_array[:,2+sweep_label] = np.linspace(min_calibrations[2+sweep_label],max_calibrations[2+sweep_label],50)
    predict = nn_predict(model, N, calibration_array)
    return calibration_array, predict

def plot_sweep(rpm, inj, operating_points, min_calibrations, max_calibrations, model, N, engine_map, df_y, target, factor, scale=True):
    
    #creating the figure
    rpm_point = closest_operating_point(rpm, inj, operating_points)[0, 0]
    inj_point = closest_operating_point(rpm, inj, operating_points)[0, 1]
    fig = plt.figure(figsize=(15,9))
    if scale == True:
        fig.suptitle('target scaled, engine rpm: {}, injection quantity: {}'.format(int(rpm_point), int(inj_point)), fontsize=16, color='b')
    if scale == False:
        fig.suptitle('full plot, engine rpm: {}, injection quantity: {}'.format(int(rpm_point), int(inj_point)), fontsize=16, color='b')
    min_y, max_y = np.min(df_y.values, axis=0), target*factor
    for sweep_label in range(0, min_calibrations.shape[0]-2):
        sweep_array,predict =  sweep(rpm, inj, operating_points, sweep_label, min_calibrations, max_calibrations, model, N)
        
        ax0 = fig.add_subplot(2,3,sweep_label+1)
        ax0.grid(False)
        plt.plot(sweep_array[:,sweep_label+2], predict[:,0], color='r')
        plt.xlabel(engine_map.calibrations.columns[sweep_label])
        plt.ylabel('Nox (g/h)', color='r')
        if scale == True:
            plt.ylim(0, max_y[0])
        ax1 = ax0.twinx()
        ax1.grid(False)
        ax1.spines["right"].set_position(("axes", 0.03))
        plt.plot(sweep_array[:,sweep_label+2], predict[:,1], color='g')
        plt.ylabel('hc (g/h)', color='g')
        if scale == True:
            plt.ylim(0, max_y[1])
        ax2 = ax0.twinx()
        ax2.grid(False)
        ax2.spines["right"].set_position(("axes", 0.8))
        plt.plot(sweep_array[:,sweep_label+2], predict[:,2], color='b')
        plt.ylabel('russ (g/h)', color='b')
        if scale == True:
            plt.ylim(0, max_y[2])
        ax3 = ax0.twinx()
        ax3.grid(False)
        ax3.spines["right"].set_position(("axes", 0.9))
        plt.plot(sweep_array[:,sweep_label+2], predict[:,3], color='k')
        plt.ylabel('co (g/h)', color='k')
        if scale == True:
            plt.ylim(0, max_y[3])
        ax4 = ax0.twinx()
        ax4.grid(False)
        ax4.spines["right"].set_position(("axes", 0.9))
        plt.plot([closest_operating_point(rpm, inj, operating_points)[0,2+sweep_label], closest_operating_point(rpm, inj, operating_points)[0,2+sweep_label]], [0, 0.8], color='y')
        plt.ylabel('')
        plt.yticks([])
        plt.ylim(0, 1)


def gdoe_chassis_comp(data, data_dyn2, time=(1067,1130), relevance_for='nox_g/h'):
    #evaluate the relevance
    from ml import ml_relevance_matrix_etr
    relevance = ml_relevance_matrix_etr(data['x0'].values, data['y0'].values, 50)
    relevance_index = list(data['y0'].columns).index(relevance_for)
    #normalise the input for proper euclidean distance
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    doe_x_transform = scale.fit_transform(data['x0'].values)
    
    chassis_x = data_dyn2['x0'][data_dyn2['v0']['time_s'].between(time[0], time[1])].mean()
    chassis_x_transform = scale.transform(chassis_x[np.newaxis,:])
    
    error_array = (doe_x_transform - chassis_x_transform)
    error_square_array = np.multiply(error_array, error_array)
    error_sum = np.sum(error_square_array, axis=1)
    index_wo_rel = np.argmin(error_sum)
    
    error_array = (doe_x_transform - chassis_x_transform) *relevance[:,relevance_index]
    error_square_array = np.multiply(error_array, error_array)
    error_sum = np.sum(error_square_array, axis=1)
    index_w_rel = np.argmin(error_sum)
    
    chassis_x = chassis_x
    chassis_y = data_dyn2['y0'][data_dyn2['v0']['time_s'].between(time[0], time[1])].mean()
    chassis_pred = data_dyn2['pb0'][data_dyn2['v0']['time_s'].between(time[0], time[1])].mean()
    
    dyn_y_columns = [s.lower() for s in data_dyn2['y0'].columns]
    doe_x_wo_rel = data['x0'].loc[index_wo_rel,:]
    doe_y_wo_rel = data['y0'].loc[index_wo_rel,dyn_y_columns]
    doe_x_w_rel = data['x0'].loc[index_w_rel,:]
    doe_y_w_rel = data['y0'].loc[index_w_rel,dyn_y_columns]
    
    df_tuple = chassis_x, chassis_y, chassis_pred, doe_x_wo_rel, doe_y_wo_rel, doe_x_w_rel, doe_y_w_rel
    
    for i in range(7):
        df_tuple[i].index = df_tuple[i].index.map(str.lower)
    for i in [0,3,5]:
        df_tuple[i].rename(index={'tw_r':'cengdst_t', 'injcrv_qpii1des_mp':'injcrv_qpii1des'},inplace=True)
    
    relevance_df = pd.Series(relevance[:,relevance_index], index=chassis_x.index)
    calibrations = pd.concat([df_tuple[0], df_tuple[3], df_tuple[5]], axis=1)
    calibrations.columns = ['Average calibration in cycle',
                            'Global DOE closest meas wo relevance', 'Global DOE closest meas w relevance']
    
    emissions = pd.concat([df_tuple[1], df_tuple[4], df_tuple[6], df_tuple[2]], axis=1)
    emissions.columns = ['Average Actual emissions in cycle','Global DOE closest meas wo relevance',
                         'Global DOE closest meas w relevance', 'Average Predicted emissions in cycle']
    
    plt.figure()
    plt.grid(False)
    plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['v0'].iloc[:,1], label='vehicle_speed', color='r')
    plt.ylabel('vehicle speed')
    plt.legend(loc='upper left')
    plt.plot([time[0],time[0]],[0,90], color='y', linewidth=2)
    plt.plot([time[1],time[1]],[0,90], color='y', linewidth=2)
    plt.twinx()
    
    plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['y0'].loc[:,relevance_for], label='actual')
    plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['pb0'].loc[:,relevance_for], label='predicted')
    plt.ylabel(relevance_for)
    plt.legend()
    plt.grid(False)
    
    plt.figure()
    plt.bar(data['x0'].columns , relevance[:,relevance_index])
    plt.ylabel('relevance_score')
    plt.xticks(rotation=45, va = 'center')
    
    #calibrations_combined = 
    return calibrations, emissions, relevance_df

def timestamp():
    temp = '[' + datetime.now().strftime('%c') + ']'
    return temp.replace(':','_').replace(' ', '_')

#%% calculate the histogram for chassis dyno cycle
def dat_to_hist(df, x_centroid, y_centroid, z_centroid=None):
    def edge(centroid):
        edges = np.zeros((len(centroid) + 1))
        for i in range(len(centroid)-1):
            edges[i + 1] = (centroid[i] + centroid[i+ 1])/2
        edges[-1] = 2*centroid[-1] - centroid[0]
        return edges
    x_edges = edge(x_centroid)
    y_edges = edge(y_centroid)
    if z_centroid ==None:
        hist,_,_ = np.histogram2d(df['x1'], df['x2'], bins=[x_edges, y_edges])
    else:
        z_edges = edge(z_centroid)
        hist,_ = np.histogramdd(df[['x1', 'x2', 'x3']].values, bins=[x_edges, y_edges, z_edges])
    hist = hist/np.sum(hist)
    return hist

#%%
    
'''
color = ['r', 'g', 'b', 'k']

#add evaluation bar
ex = fig.add_subplot(2,3,6)
plt.xlabel('emission parameters')
def target_ratio(t):target_ratio(0), color=color)
#plt.xticks([0,1,
    return np.ravel(output_list[t, None]/TARGET_EMISSION_LOSS_START)
    
#plt.bar([0,1,2,3] ,2,3], ['Nox', 'hc', 'russ', 'co'])
plt.bar([0], [0])
plt.ylim((0,100))
plt.ylabel('calibration progress (%)')

#putting legends as per our wish
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=color[0], lw=2),
                Line2D([0], [0], color=color[1], lw=2),
                Line2D([0], [0], color=color[2], lw=2),
                Line2D([0], [0], color=color[3], lw=2)]

#Create the figure for time = 0
for i in range(5):
    predict = sweep(i+1, input_list[None, 0, :])
    ax[i][3].legend(custom_lines, ['nox', 'hc', 'russ', 'co'], loc='upper center', ncol=2, shadow=True, frameon=True)
    cx[i], = ax[i][3].plot([input_list[0,i+1],input_list[0,i+1]],[0,300], color='y')
    for j in range(4):
        ax[i][j].grid(False)
        bx[i][j], = ax[i][j].plot(np.linspace(minimum[i+1],maximum[i+1], 50), predict[:,j], color = color[j])

from matplotlib.animation import FuncAnimation
def animate(t):
    ex.collections = []
    #ex.bar([0,1,2,3] ,target_ratio(t), color=color)
    ex.bar([0], [t*100/90])
    for i in range(5):
        predict2 = sweep(i+1, input_list[None, t,:])
        cx[i].set_xdata([input_list[t,i+1],input_list[t,i+1]])
        for j in range(4):
            #ax[i][j].collections = []
            bx[i][j].set_ydata(predict2[:,j])

anim = FuncAnimation(fig, animate, interval=1, frames=TIME-1)
'''

#%%
def match_pick(File, num_rows, num_match):
    import pandas as pd
    import numpy as np
    df = pd.read_excel(File, sheet_name=None, skiprows=[1])
    num_last_comp = df['emap'].shape[1] - num_match
    emap_x = df['emap'].iloc[:,:-num_last_comp]
    gdoe_x = df['gdoe'].iloc[:,:-num_last_comp]
    
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    gdoe_xs = scale.fit_transform(gdoe_x)
    emap_xs = scale.transform(emap_x)
    n_gdoe = gdoe_xs.shape[0]
    n_emap = emap_xs.shape[0]
    array = np.zeros((n_gdoe, n_emap))
    
    for i in range(n_gdoe):
        for j in range(n_emap):
            array[i,j] = np.square(gdoe_xs[i] - emap_xs[j]).sum()
    
    def sort_minimum(array):
        array = array.copy()
        sorted_minimum = []
        for i in range(num_rows):
            temp = np.unravel_index(array.argmin(), array.shape)
            sorted_minimum.append(temp)
            array[temp[0], temp[1]] = array.max()
        return sorted_minimum
    
    sorted_minimum = sort_minimum(array)
    
    comp_gdoe = pd.DataFrame(np.zeros((num_rows,df['gdoe'].shape[1])), columns=df['gdoe'].columns)
    for i in range(num_rows):
        comp_gdoe.iloc[i,:] = df['gdoe'].iloc[sorted_minimum[i][0]].values
    
    comp_emap = pd.DataFrame(np.zeros((num_rows,df['emap'].shape[1])), columns=df['emap'].columns)
    for i in range(num_rows):
        comp_emap.iloc[i,:] = df['emap'].iloc[sorted_minimum[i][1]].values
    return comp_gdoe, comp_emap

def melt2pivot(calibration, target, num):
    '''
    calibration: 3d , 1st axis is time
    target: 3d , 1st axis is time
    num: to be mapped column [3rd axis] from target
    as helper for animation of boundary progress
    3d to 3d, first axis is time
    '''
    all_time_rpm = calibration[:,:,0]
    all_time_inj = calibration[:,:,1]
    all_time_target = target[:,:,num]
    rpm = all_time_rpm[0][:,None]
    inj = all_time_inj[0][:,None]
    
    array_list = []
    for time in range(all_time_rpm.shape[0]):
        target = all_time_target[time][:,None]
        df = pd.DataFrame(np.concatenate([rpm,inj,target], axis=1),
                          columns=['rpm','inj','target'])
        df = df.pivot_table('target', 'rpm', 'inj').fillna(0)
        array_list.append(df.values[None,:])
        if time == 0:
            rpm_index = list(df.index.astype('int'))
            inj_col = list(df.columns.astype('int'))
    array = np.concatenate(array_list, axis=0)
    return array, rpm_index, inj_col

def average_emission(emission_progress, factor, avg_speed, cycle_time):
    emission_progress_km_h = emission_progress/avg_speed
    emission_progress_avg_km_h = np.sum(emission_progress_km_h*factor, axis=1, keepdims=True)
    return emission_progress_km_h, emission_progress_avg_km_h

if __name__ == '__main__':
    comp_gdoe, comp_emap = match_pick('C:/Users/kgu2ban/Desktop/AI/modules/gdoe_emap.xlsx', 99, 8)
    for i in range(1,16):   
        plt.figure()
        plt.scatter(comp_gdoe.iloc[:,-i], comp_emap.iloc[:,-i], c=np.linspace(1,10,99), cmap='RdYlGn_r')
        plt.title(comp_gdoe.columns[-i] + ' ' + comp_emap.columns[-i])
        plt.colorbar()
        print(comp_emap.columns[-i])

    File = 'C:/Users/kgu2ban/Desktop/nox.xlsx'
    df = pd.read_excel(File, sheet_name=None, skiprows=[1])
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    df['x_poly']  = poly.fit_transform(df['x'])
    df['x_poly'] = df['x']
    
    X_train, X_test, y_train, y_test = train_test_split(df['x_poly'], df['y'], test_size=0.33, random_state=42)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print(reg.score(X_train, y_train))
    print(reg.score(X_test, y_test))
    plt.figure()
    plt.scatter(reg.predict(X_test), y_test)
    
    
    