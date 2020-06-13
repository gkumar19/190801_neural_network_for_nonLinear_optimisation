# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:11:25 2019

@author: KGU2BAN
"""
import mdfreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nn import nn_predict

def dyn_dat(file, channel_list, raster):
    '''load the dat file and return in pandas dataframe'''
    df = pd.DataFrame(columns=channel_list)
    yop = mdfreader.Mdf(file)
    yop.resample(raster)
    for channel in channel_list:
        values = yop.get_channel_data(channel)
        df[channel] = values
    return df

def dyn_0(LOC,FILE):
    '''step0 in 3 sequence step: load the dat file'''
    channel_list = ['x1', 'x2', 'x3',
                'x4', 'x5', 'x6',
                 'x7','x9', 'x10']
    data_dyn0 = {'x0': dyn_dat(LOC+FILE, channel_list, 0.1),
                'y0': dyn_dat(LOC+FILE, ['y21'], 0.1),
                'v0': dyn_dat(LOC+FILE, ['v1'], 0.1)}
    data_dyn0['x0'] = data_dyn0['x0'].astype('float64')
    data_dyn0['x0'].loc[:,'x10'] = 910/1000
    #eliminate switched off condition
    running_condition = data_dyn0['x0']['x1'] > 600/1000
    for key in data_dyn0.keys():
        data_dyn0[key] = data_dyn0[key][running_condition]
        data_dyn0[key].reset_index(inplace=True, drop=True)
    
    #convert rail pressure to bar
    data_dyn0['x0']['x7'] = data_dyn0['x0']['x7']/1000
    for identifier in ['x0', 'y0']:
        data_dyn0[identifier].columns = [column_name.lower() for column_name in data_dyn0[identifier].columns]
    return data_dyn0

def dyn_1(LOC,FILE, data_dyn0):
    '''step1 in 3 sequence step: load the ascii file'''
    def load_asc(LOC,FILE):
        '''loads the asci file and return resampled dataframe'''
        from scipy.signal import resample
        replacement_list = ['nox_g/h', 'hc_g/h', 'co_g/h', 'time_s', 'vehv_v_km/h', 'nox_t_g/h', 'hc_t_g/h', 'co_t_g/h', 'co2_kg/h', 'opac_ana']
        asci_list = ['NOXEMODGM_g/h', 'THCEMODGM_g/h', 'COLEMODGM_g/h',
                     'X_TESTTIME_s', 'SPEED_km/h', 'NOXUMODGM_g/h', 'THCUMODGM_g/h', 'COLUMODGM_g/h', 'CO2EMODGM_g/h', 'OPACITY_DIGITAL_unity']
        
        #load file and merge first index to header
        df = pd.read_csv(LOC+FILE, encoding='mbcs', sep=' ', low_memory=False)
        header_list = [i+'_'+j for i,j in zip(df.columns, df.iloc[0,:])]
        df.columns = header_list
        df = df.loc[1:,asci_list]
        df.reset_index(drop=True, inplace=True)
        df.columns = replacement_list
        df.iloc[:,:] = df.values.astype('float')
        
        #resampling at 0.1 raster
        num_samples = df.shape[0]
        t_start = df.loc[0,'time_s']
        t_end = df.loc[num_samples-1,'time_s']
        array_resampled = np.zeros((len(np.arange(t_start, t_end, 0.1)), len(replacement_list)))
        df_resampled = pd.DataFrame(array_resampled, columns=replacement_list)
        df_resampled['time_s'] =  np.arange(t_start, t_end, 0.1)
        replacement_list.remove('time_s')
        for value in replacement_list:
            (df_resampled[value],_) = resample(df[value].values, df_resampled.shape[0], t=df_resampled['time_s'].values)
        df_resampled['co2_kg/h'] = df_resampled['co2_kg/h']/1000 #converting from grams to kg
        df_resampled['opac_ana'].iloc[:] = np.clip(df_resampled['opac_ana'].iloc[:].values*100, 0.001, 100) #converting into percentage
        return df_resampled
    def load_csv(LOC,FILE):
        return pd.read_csv(LOC+FILE)
    
    def align_asci_to_dat(asc_array, dat_array):
        '''align the ascii array to dat array, if positive means ascii is leading'''
        max_length = min([len(asc_array), len(dat_array)])
        asc_array = asc_array[:max_length]
        dat_array = dat_array[:max_length]
        asc_dat_diff_array = np.zeros(max_length)
        half_max_length = int(max_length/2) + 1
        for i in range(-half_max_length,half_max_length):
            asc_roll = np.roll(asc_array, i)
            asc_dat_diff_array[i] = abs(dat_array-asc_roll).sum()
        return np.argmin(asc_dat_diff_array)
    df_resampled = load_csv(LOC,FILE)
    ascii_movement = align_asci_to_dat(df_resampled['v2'].values,
                                       data_dyn0['v0'].values.flatten())
    print('ascii size: ', len(df_resampled))
    print('correction: ', ascii_movement)
    length_dat = data_dyn0['v0'].shape[0]
    length_ascii = df_resampled.shape[0]
    if ascii_movement < 0:
        ascii_movement = - ascii_movement
        dat_min = 0
        dat_max = min([length_dat, length_ascii-ascii_movement])
        ascii_min = ascii_movement
        ascii_max = ascii_movement + min([length_dat, length_ascii-ascii_movement])
    if ascii_movement >= 0:
        dat_min = ascii_movement
        dat_max = ascii_movement + min([length_ascii, length_dat-ascii_movement])
        ascii_min = 0
        ascii_max = min([length_ascii, length_dat-ascii_movement])
    
    
    data_dyn1 = data_dyn0.copy()
    #generate ['x0']
    data_dyn1['x0'] = data_dyn0['x0'].iloc[dat_min:dat_max,:]
    data_dyn1['x0'].reset_index(inplace=True, drop=True)
    
    #generate ['y0']
    dat_y0 = data_dyn0['y0'].iloc[dat_min:dat_max,:]
    dat_y0.reset_index(inplace=True, drop=True)
    ascii_y0 = df_resampled.iloc[ascii_min:ascii_max,[0,1,2,5,6,7,8,9]]
    ascii_y0.reset_index(inplace=True, drop=True)
    data_dyn1['y0'] = pd.concat([ascii_y0, dat_y0], axis=1)
    
    #generate ['v0']
    dat_v0 = data_dyn0['v0'].iloc[dat_min:dat_max,:]
    dat_v0.reset_index(inplace=True, drop=True)
    ascii_v0 = df_resampled.iloc[ascii_min:ascii_max,4]
    ascii_v0.reset_index(inplace=True, drop=True)
    time_v0 = df_resampled.iloc[ascii_min:ascii_max,3]
    time_v0.reset_index(inplace=True, drop=True)
    data_dyn1['v0'] = pd.concat([time_v0, dat_v0, ascii_v0], axis=1)
    
    #generate y1 from y0 and remove those labels
    data_dyn1['y1'] =  data_dyn1['y0'].iloc[:,[3,4,5,7,8]]
    data_dyn1['y0'] = data_dyn1['y0'].iloc[:,[0,1,2,6, 7]]
    #plot for info on alignment
    plt.figure('aligned velocities from chassis dyno and inca file')
    plt.plot(data_dyn1['v0'].iloc[:,1], label='dat file')
    plt.plot(data_dyn1['v0'].iloc[:,2], label='ascii file')
    plt.ylabel('vehicle speed in km/h')
    plt.xlabel('time in seconds')
    plt.legend()
    steady_states = steady_indexer(data_dyn1['x0']['x1'].values, signal_delta=30)
    data_dyn1['ss'] = pd.DataFrame(steady_states[:,None], columns=['steady_state'])
    return data_dyn1

def dyn_2(data_dyn1, model_base, data, N, plot=True):
    '''predict and store the value in y pred'''
    data_dyn2 = data_dyn1.copy()
    predicted_targets = list(data['y0'].columns)
    #predicted_targets.append('afs_mairpercyl')
    df = pd.DataFrame(nn_predict(model_base, N, data_dyn1['x0'].values), columns = predicted_targets)
    
    chassis_targets = data_dyn1['y0'].columns
    other_targets = [i for i in predicted_targets if i not in data_dyn1['y0'].columns]
    
    data_dyn2['pb0'] = df.loc[:,chassis_targets]
    data_dyn2['pb1'] = df.loc[:,other_targets]
    #plots for chassis measurement
    if plot==True:
        plt.figure('prediction with base model')
        for i in range(4):
            plt.subplot(2,2,i+1)
        
            plt.grid(False)
            plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['v0'].iloc[:,1], label='vehicle_speed', color='r')
            plt.yticks([])
            plt.legend()
            plt.title(data_dyn2['y0'].columns[i])
            plt.twinx()
            plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['y0'].iloc[:,i], label='actual', color='k')
            plt.plot(data_dyn2['v0']['time_s'].values, data_dyn2['pb0'].iloc[:,i], label='predicted', color='b')
            plt.ylabel(data_dyn2['y0'].columns[i])
            plt.legend()
            plt.grid(False)
    return data_dyn2

def dyn_3(data_dyn1, model_dyn, N, kind='offset'):
    data_dyn3 = data_dyn1.copy()
    if kind == 'offset':
        data_dyn3['po0'] = pd.DataFrame(nn_predict_dyn(model_dyn, N, data_dyn1['x0'].values), columns = data_dyn1['y0'].columns)
    if kind == 'gru':
        data_dyn3['pg0'] = pd.DataFrame(nn_predict_dyn(model_dyn, N, data_dyn1['x0'].values), columns = data_dyn1['y0'].columns)
    
    return data_dyn3

def steady_indexer(signal, signal_delta=30, check_time=5, raster=0.1):
    '''indexes signal based on steady state conditions'''
    total_index_length = signal.shape[0]
    check_index = int(check_time/raster)
    start_index = int(check_time/raster)+1
    end_index = total_index_length - start_index
    steady_state_index = np.zeros_like(signal)
    for index in range(start_index, end_index):
        if (np.max(signal[index-check_index : index+check_index]) - np.min(signal[index-check_index : index+check_index])) < signal_delta:
            steady_state_index[index] = 1
    return steady_state_index

def plot_cycle_emission(df, raster):
    '''first column should be time in second
    second column should be velocity in km/h'''
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    df = pd.concat([pd.DataFrame(np.linspace(0, df.shape[0]*raster, df.shape[0]), columns=['time']) ,df], axis=1)
    fig = plt.figure('cycle emission')
    fig.tight_layout()
    ax_list = [None]
    ax_list[0] = fig.add_subplot(111)
    ax_list[0].plot(df.iloc[:,0], df.iloc[:,1], color=color[0])
    ax_list[0].set_ylabel(df.columns[1])
    plt.grid(False)
    for i in range(df.shape[1]-2):
        ax_list.append(ax_list[0].twinx())
        ax_list[i+1].spines["right"].set_position(("axes", 1 + 0.04*(i)))
        ax_list[i+1].plot(df.iloc[:,0], df.iloc[:,i+2], color=color[i+1])
        ax_list[i+1].set_ylabel(df.columns[i+2])
        ax_list[i+1].grid(False)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color[i], lw=2) for i in range(df.shape[1]-1)]

    for i in range(df.shape[1]-1):
        ax_list[i].legend(custom_lines, df.columns[1:], loc='upper center', ncol=2, shadow=True, frameon=True)

def convert_to_sequence(array, sample_size):
    '''takes 2D array for batch x features and returns 3D array of batch x time sequence x features'''
    reduced_size = array.shape[0] - sample_size + 1
    array_sequence = np.zeros((reduced_size, sample_size, array.shape[1]))
    for i in range(reduced_size):
        array_sequence[i,:,:] = array[i:i+sample_size,:]
    return [array[sample_size-1:], array_sequence]
#%%
class TransferLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                      shape=(1, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.built = True
        
        super(TransferLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        return (x * self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

def nn_model_dyn(model, sample_size, N, dynamic_layers=3):
    model_base, _ = nn_model_base(8,4,11, 50,50,40, 0,0,0)
    model_base.set_weights(model.get_weights())
    for layer in model_base.layers:
        layer.trainable = False
    
    #create base output
    def base_model_output(x):
        #x1 = N.t_n2t(x)[:,0,None]
        #x2 = N.t_n2t(x)[:,1,None]
        #x3 = N.t_n2t(x)[:,3,None]
        #x4 = N.t_n2t(x)[:,5,None]
        #return tf.concat([x1,x2,x3,x4], axis=1)
        x_output = []
        for i in N.dynamic_targets:
            x_output.append(x[:,i,None])
        return tf.concat(x_output, axis=1)

    #base_transformed = tf.keras.layers.Lambda(lambda x: N.t_n2t(x)[:,5,None], name='base_transformation')(model_base.output)
    base_transformed = tf.keras.layers.Lambda(base_model_output, name='base_transformation')(model_base.output)
    
    #create offset
    offset = TransferLayer(name='add_offset')(base_transformed)
    
    #create dynamic layer
    dynamic_input = tf.keras.layers.Input(shape=(sample_size,8), name='dynamic_insertion')
    dynamic_layer1 = tf.keras.layers.GRU(4, return_sequences=True, activation='relu', name='gru1')(dynamic_input)
    dynamic_output = tf.keras.layers.GRU(4, return_sequences=False, activation='relu', name='gru2')(dynamic_layer1)
    if dynamic_layers == 3:
        added_layers = [base_transformed, offset, dynamic_output]
    if dynamic_layers == 2:
        added_layers = [base_transformed, dynamic_output]
    if dynamic_layers == 1:
        added_layers = [base_transformed, offset]
    
    add_layer = tf.keras.layers.Add(name='add')(added_layers)
    dyn_model = tf.keras.models.Model(inputs=[model_base.input, dynamic_input], outputs=add_layer)
    return dyn_model

def nn_dyn_train(data_dyn2, model_dyn, N, kind='offset_compile', epoch=2000, validation_split=0.5):
    steady_states = steady_indexer(data_dyn2['x0']['epm_neng'].values, signal_delta=30)
    if kind == 'offset_compile':
        #plot considered steady state
        plt.figure('steady state points')
        plt.plot(steady_states, color='y', label='steady_states')
        plt.legend(loc='upper left')
        plt.twinx()
        plt.plot(data_dyn2['v0']['vehv_v_km/h'], color='r', label='vehicle_speed')
        plt.legend()
        
        #input for fit
        model_dyn.get_layer('add_offset').trainable = True
        for layer_name in ['gru1', 'gru2']:
            model_dyn.get_layer(layer_name).trainable = False
            w = [np.zeros_like(i) for i in model_dyn.get_layer(layer_name).get_weights()]
            model_dyn.get_layer(layer_name).set_weights(w)
        model_dyn.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        model_dyn.summary()
        
    if kind == 'offset_fit':
        input_array = N.f2n(data_dyn2['x0'][steady_states==1].values)
        input_array = convert_to_sequence(input_array, 10)
        output_array = N.d_t2n(data_dyn2['y0'][steady_states==1].values)
        output_array = convert_to_sequence(output_array, 10)[0]
        model_dyn.fit(input_array,output_array, epochs=epoch, batch_size=2000, validation_split=validation_split)
        
        #plot after fit
        input_array_full = N.f2n(data_dyn2['x0'].values)
        input_array_full = convert_to_sequence(input_array_full, 10)
        output_array_full = data_dyn2['y0'].values
        output_array_full = convert_to_sequence(output_array_full, 10)[0]
        
        pred = N.d_n2t(model_dyn.predict(input_array_full))
        plt.figure('after_offset')
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(output_array_full[:,i], label='actual', color='k')
            plt.plot(pred[:,i], label='predicted', color='b')
            plt.legend()
            
    if kind == 'gru_compile':
        model_dyn.get_layer('add_offset').trainable = False
        for layer_name in ['gru1', 'gru2']:
            model_dyn.get_layer(layer_name).trainable = True
            w = [np.random.rand(*i.shape)*0.1 for i in model_dyn.get_layer(layer_name).get_weights()]
            model_dyn.get_layer(layer_name).set_weights(w)
        model_dyn.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4))
        model_dyn.summary()
    
    if kind == 'gru_fit':
        input_array = N.f2n(data_dyn2['x0'].values)
        input_array = convert_to_sequence(input_array, 10)
        output_array = N.d_t2n(data_dyn2['y0'].values)
        output_array = convert_to_sequence(output_array, 10)[0]
        
        model_dyn.fit(input_array,output_array, epochs=epoch, batch_size=2000, validation_split=validation_split)
        
        pred = N.d_n2t(model_dyn.predict(input_array))
        plt.figure('after_dynamics')
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(N.d_n2t(output_array)[:,i], label='actual', color='k')
            plt.plot(pred[:,i], label='predicted', color='b')
            plt.legend()
    return model_dyn


#%%
def construct_cycle(LOC,FILE_BASE_DAT, FILE, no_ascii=False):
    from scipy.interpolate import interp2d
    '''load excel file calibration maps and reconstructs new data_dyn0 file, which has
    atleast engine rpm, injection qty and temperature already'''
    data_dyn0 = dyn_0(LOC,FILE_BASE_DAT)
    if no_ascii==True:
        data_dyn0['v0']['time_s'] = np.linspace(0,0.1*data_dyn0['v0'].shape[0],num=data_dyn0['v0'].shape[0])
    mycalibration = pd.read_excel(LOC+FILE, sheet_name=None, index_col=0)
    
    #store inc and dec maps separately
    rail_inc = mycalibration['rail_inc']
    rail_dec = mycalibration['rail_dec']
    del mycalibration['rail_inc'], mycalibration['rail_dec']
    
    data_dync = data_dyn0.copy()
    df_x0 = pd.DataFrame()
    df_x0['epm_neng'] = data_dyn0['x0']['epm_neng']
    df_x0['injctl_qsetunbal'] = data_dyn0['x0']['injctl_qsetunbal']
    df_x0['cengdst_t'] = data_dyn0['x0']['cengdst_t']
    rpms = list(data_dync['x0'].loc[:,'epm_neng'])
    injs = list(data_dync['x0'].loc[:,'injctl_qsetunbal'])
    for cal, df in mycalibration.items():
        f = interp2d(df.columns, df.index, df.values, kind='linear')
        array = [f(epm,inj)[0] for epm, inj in zip(rpms, injs)]
        df_x0[cal] = array
    data_dync['x0'] = df_x0
    
    data_dync['y0'] = pd.DataFrame(columns= data_dync['y0'].columns)
    #handeling inc and dec map
    rail_base = data_dync['x0']['railp_pflt'].values
    #data_dync['x0']['railp_pbase'] = rail_base #if rail base is also needed to be plotted
    rail_set = np.zeros_like(rail_base) # supposed to be final value
    f_inc = interp2d(rail_inc.columns, rail_inc.index, rail_inc.values, kind='linear')
    f_dec = interp2d(rail_dec.columns, rail_dec.index, rail_dec.values, kind='linear')
    for i in range(1, rail_set.shape[0]):
        previous_value = rail_set[i-1]
        max_allowed = previous_value + (f_inc(rpms[i-1], injs[i-1])[0] * 0.1) #0.1 is raster
        min_allowed = previous_value - (f_dec(rail_set[i-1], injs[i-1])[0] * 0.1) #0.1 is raster
        rail_set[i] = np.clip(rail_base[i], min_allowed, max_allowed)
    data_dync['x0']['railp_pflt'] = rail_set
    #plot rail pressure base and setpoint
    plt.plot(data_dync['x0']['railp_pflt'], label='rail pressure actual applicated')
    plt.plot(rail_base, label='rail pressure from base map')
    plt.ylabel('rail_pessure_bar')
    plt.xlabel('time_s')
    plt.legend()
    
    return data_dync
#%% construct cycle based on new inputs
def cycle_progress(FILE_DAT, FILE_NUMPY,model_base_o, N_o, data_reduce=100):
    from scipy.interpolate import interp2d
    
    def chop_numpy(array, data_reduce):
        #array is 3d
        shortened_data = np.zeros_like(array)[:100]
        window = int(array.shape[0]/data_reduce)
        for i in range(data_reduce):
            shortened_data[i] = array[i*window]
        return shortened_data
    calibration_progress = np.load(FILE_NUMPY)
    shortened_progress = chop_numpy(calibration_progress, data_reduce)
    del calibration_progress
    
    def generate_dat(FILE_DAT, channel_list):
        temp_dat =  dyn_dat(FILE_DAT, channel_list, raster=1)
        temp_dat = temp_dat[temp_dat['x1'] > 600/1000]
        temp_dat.reset_index(inplace=True, drop=True)
        return temp_dat
    temp_dat = generate_dat(FILE_DAT, ['x1', 'x2', 'x3'])
    
    model_input = np.zeros((shortened_progress.shape[0], temp_dat.shape[0], shortened_progress.shape[2]))
    
    for time in range(shortened_progress.shape[0]):
        temp_slice = shortened_progress[time]
        df_slice = pd.DataFrame(temp_slice, index=None)
        
        for cal_index in [3,4,5,6,7]:
            df_pivot = df_slice.pivot_table(values=cal_index, index=0, columns=1).fillna(method='ffill', axis=1)
            f = interp2d(df_pivot.index, df_pivot.columns, df_pivot.values.T, kind='linear')
            model_input[time, :, cal_index] = np.array([f(epm,inj)[0] for epm, inj in zip(temp_dat['x1'], temp_dat['x2'])])
        model_input[time, :, 0] = temp_dat['x1']
        model_input[time, :, 1] = temp_dat['x2']
        model_input[time, :, 2] = temp_dat['x3']
        model_input[time, :, 8] = 910/1000
    
    from nn import nn_predict
    model_output = [nn_predict(model_base_o, N_o, model_input[i])[None,:] for i in range(100)]
    model_output = np.concatenate(model_output, axis=0)
    
    drive_pattern = temp_dat = generate_dat(FILE_DAT, channel_list = ['x1', 'v1'])
    
    return drive_pattern, model_input, model_output

#%%
import matplotlib.gridspec as gridspec
def plot_cycle(data_dyn2, target, a, russ_value=0, plot_accumulated=True):
    
    data_dyn2 = data_dyn2.copy()
    if 'pg0' in data_dyn2:
        data_dyn2['pg0'] = pd.concat([data_dyn2['pg0'], data_dyn2['pb1']], axis=1)
    if "po0" in data_dyn2:
        data_dyn2['po0'] = pd.concat([data_dyn2['po0'], data_dyn2['pb1']], axis=1)
    if 'pb0' in data_dyn2:
        data_dyn2['pb0'] = pd.concat([data_dyn2['pb0'], data_dyn2['pb1']], axis=1)
    data_dyn2['x0'] = pd.concat([data_dyn2['x0'], data_dyn2['y0']], axis=1)
    data_dyn2['y0'] = pd.concat([data_dyn2['y0'], data_dyn2['y1']], axis=1)
    
    name_transfer = {'x0': 'calibration',
                     'y0': 'actual',
                     'y1': 'actual',
                     'pb0': 'prediction',
                     'pb1': 'prediction',
                     'po0': 'offset_+_steady_prediction',
                     'pg0': 'dynamics_offset_+_steady_prediction'}
    
    target_emissions = {'y1': 0.27,
                    'y4': 2.0,
                    'y2': 0.24,
                    'y3': 0.029}
    average_speed = data_dyn2['v0'].mean()[1]
    
    def plot_scatter(x_label, y_label, kind='y0', insert_manually=False, insert_location='x', insert_value=0):
        if ((insert_manually==True) & (insert_location == 'x')):
            average_x_emission = insert_value
        else:
            average_x_emission = data_dyn2[kind].mean()[x_label] / average_speed # in g/km
        
        if ((insert_manually==True) & (insert_location == 'y')):
            average_y_emission = insert_value
        else:
            average_y_emission = data_dyn2[kind].mean()[y_label] / average_speed # in g/km
            
        plt.scatter(average_x_emission, average_y_emission, label=name_transfer[kind], s=150)
        
    def plot_box(x_label, y_label):
        plt.plot([ 0, target_emissions[x_label], target_emissions[x_label] ],
         [ target_emissions[y_label], target_emissions[y_label], 0 ], color='blue', label='limits')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(0,target_emissions[x_label]*5)
        plt.ylim(0,target_emissions[y_label]*5)
    
    time = data_dyn2['v0']['time_s']
    
    fig = plt.figure(figsize=(20,9), constrained_layout=True)
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    fig.suptitle(target, fontsize=20, color='b')
    
    #First plot
    ax = fig.add_subplot(gs[0, 0:-1])
    for i in a:
        plt.plot(time, data_dyn2[i][target], label=name_transfer[i])
    plt.xticks([])
    plt.ylabel(target)
    plt.legend(loc='upper left')
    
    #second plot
    fig.add_subplot(gs[1, 0:-1], sharex=ax)
    if plot_accumulated==True:
        for i in a:
            plt.plot(time, data_dyn2[i][target].cumsum()/36000, label=name_transfer[i])
        plt.xlabel('time_s')
        plt.ylabel(target[:-2])
        plt.legend(loc='upper left')
    else:
        plt.yticks([])
    plt.twinx()
    plt.grid(False)
    plt.plot(time, data_dyn2['v0']['v1'], color='r', label='v1')
    plt.yticks([])
    plt.legend(loc='upper center')
    
    #third plot
    fig.add_subplot(gs[0, -1])
    #plot once with actuals
    plot_box('y1', 'y3')
    plot_scatter('y1', 'y3', insert_manually= True, insert_location='y', insert_value=russ_value, kind='y0') #0.0325, october file result
    
    #plot when it's predicted
    for i in a:
        if i !='y0':
            plot_scatter('y1', 'y3', kind=i)
    plt.legend()
    
    fig.add_subplot(gs[1, -1])
    #plot once with actuals
    plot_box('y2', 'y4')
    plot_scatter('y2', 'y4', kind='y0')
    
    #plot if it's predicted
    for i in a:
        if i != 'y0':
            plot_scatter('y2', 'y4', kind=i)
    
    plt.legend()

def nn_predict_dyn(model_dyn, N, feature_array):
    input_array_full = N.f2n(feature_array)
    input_array_full = convert_to_sequence(input_array_full, 10)
    pred_cut = N.d_n2t(model_dyn.predict(input_array_full))
    pred = np.zeros((feature_array.shape[0], pred_cut.shape[1]))
    pred[-(pred_cut.shape[0]):,:] = pred_cut
    return pred

def generate_dyns_from_folder(LOC, model1, data, N, data_dyn_list=None):
    from glob import glob
    import os
    folders = sorted(glob(LOC+'/*'))
    folders = [file.replace('\\','/') for file in folders]
    dat_df = {}
    russ_value = {}
    for folder in folders:
        folder_name = os.path.basename(folder)
        dat_file = glob(folder+'/*.dat')[0].replace('\\','/')
        asc_file = glob(folder+'/*.csv')[0].replace('\\','/')
        if data_dyn_list == None:
            data_dyn = dyn_0(dat_file,'')
            data_dyn = dyn_1(asc_file,'', data_dyn)
        else:
            data_dyn = data_dyn_list[folder_name]
        data_dyn = dyn_2(data_dyn, model1, data, N, plot=False)
        dat_df[folder_name] = data_dyn
        
        df = pd.read_excel(glob(folder+'/*.xlsx')[0].replace('\\','/'), sheet_name='EmissionValues')
        df.set_index('Test Equipments', inplace=True)
        russ_value[folder_name] = (df.loc['PM             Total', 'Make'])
    return dat_df, russ_value