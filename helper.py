# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:52:40 2019

@author: KGU2BAN
"""
import pandas as pd
import numpy as np
from constraints import constraints_load_excel, constraints_calibration_sheet

def helper_dataframe(calibration_rpms, calibration_injs, buffer_quantity, data, data_dyn_list, CONSTRAINT_FILE,calibration_temp=None):
    '''prepare dataframe for storing calibrations and emissions of stage 2'''
    #extraxt data
    df_qlim = data['q0']
    df_x = data['x0']
    df_y = data['y0']
    df_engine_rpm = data['e0']
    df_injection_quantity = data['i0']
    num_calibrations = data['c0'].shape[1]
    num_emissions = data['n0'].shape[1]
    
    #store constraints in engine map to be used in loss function of stage 2
    
    
    
    class EngineMap():
        '''store dataframe and arrays'''
        def __init__(self,df,num_calibrations, num_emissions):
            self.df = df
            self.num_calibrations = num_calibrations
            self.num_emissions = num_emissions
            self.calibration_rpms = calibration_rpms
            self.calibration_injs = calibration_injs
        def refresh(self):
            self.engine_rpms = self.df.iloc[:,0]
            self.injection_quantities = self.df.iloc[:,1]
            self.calibrations = self.df.iloc[:,2:2+num_calibrations]
            self.emissions = self.df.iloc[:,2+num_calibrations:2+num_calibrations+num_emissions]
            self.boundaries = self.df.iloc[:,2+num_calibrations+num_emissions:]
            self.emissions_and_boundaries = self.df.iloc[:,2+num_calibrations:]            
        
    from scipy.interpolate import interp1d
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from utils import dat_to_hist
    fac_map_list = [dat_to_hist(data_dyn['x0'], calibration_rpms, calibration_injs, calibration_temp) for data_dyn in data_dyn_list]
    fac_map = np.maximum.reduce(fac_map_list)
    qlim_func = interp1d(list(df_qlim.columns), list(df_qlim.iloc[0,:]), kind='linear')
    rpm_list = []
    injection_list = []
    temp_list = []
    factor_list = []
    injection_num = [0]*len(calibration_rpms)
    for temp_num, temp in enumerate(calibration_temp):
        for rpm_num, rpm in enumerate(calibration_rpms):
            is_odd = rpm_num % 2
            is_odd = 0
            if is_odd == 1:
                calibration_injs_modified = reversed(calibration_injs)
            if is_odd == 0:
                calibration_injs_modified = calibration_injs
            for inj_num,injection_quantity in enumerate(calibration_injs_modified):
                if injection_quantity < qlim_func([rpm])[0]+buffer_quantity:
                    injection_list.append(injection_quantity)
                    rpm_list.append(rpm)
                    temp_list.append(temp)
                    factor_list.append(fac_map[rpm_num, inj_num, temp_num])
                    if temp_num == 0:
                        injection_num[rpm_num] = injection_num[rpm_num] + 1
                else:
                    pass
    df_engine_map = pd.DataFrame(np.zeros((len(rpm_list),df_x.shape[1]+df_y.shape[1])), columns=list(df_x.columns)+list(df_y.columns))
    df_engine_map.iloc[:,0] = rpm_list
    df_engine_map.iloc[:,1] = injection_list
    df_engine_map.iloc[:,2] = temp_list
    
    #Plot the distribution of points
    plt.figure()
    plt.title('engine load points')
    plt.scatter(rpm_list, injection_list, label='calibration points')
    plt.plot(calibration_rpms, qlim_func(calibration_rpms), label='interpolated qlim')
    #plt.plot(list(df_qlim.columns), list(df_qlim.iloc[0,:]), label='actual qlim')
    plt.scatter(df_engine_rpm.iloc[:,0], df_injection_quantity.iloc[:,0], s=3, color='k', label='measured points')
    plt.xlabel('engine rpm')
    plt.ylabel('injection quantity (mg)')
    plt.legend()
    engine_map = EngineMap(df_engine_map, num_calibrations, num_emissions)
    engine_map.fac_map = pd.DataFrame(fac_map.sum(axis=-1).T, columns=calibration_rpms, index=calibration_injs)
    engine_map.factor = factor_list
    engine_map.avg_speed = data_dyn_list[0]['v0']['v1'].mean()
    
    def couple_generator(injection_num):
        '''
        function creates couple of operating points for gradient limitation further in loss function of nn
        '''
        opt_num = 0
        rpm_num_list = []
        inj_num_list = []
        opt_num_list = []
        for rpm_num, inj_num_max in enumerate(injection_num):
            for inj_num in range(inj_num_max):
                rpm_num_list.append(rpm_num)
                inj_num_list.append(inj_num)
                opt_num_list.append(opt_num)
                opt_num+= 1
        couple_inj_axis = []
        couple_rpm_axis = []
        for opt_num1, rpm_num1, inj_num1 in zip(opt_num_list, rpm_num_list, inj_num_list):
            for opt_num2, rpm_num2, inj_num2 in zip(opt_num_list, rpm_num_list, inj_num_list):
                if (rpm_num2 == rpm_num1) and (inj_num2-inj_num1==1):
                    couple_rpm_axis.append([opt_num1, opt_num2])
                if (rpm_num2 - rpm_num1 ==1) and (inj_num2 == inj_num1):
                    couple_inj_axis.append([opt_num1, opt_num2])
        
        num_single_temp = int(len(engine_map.df) / len(calibration_temp))
        num_original_couple_rpm = len(couple_rpm_axis)
        num_original_couple_inj = len(couple_inj_axis)
        
        for i in range(num_original_couple_rpm * (len(calibration_temp)-1)):
            temp = couple_rpm_axis[-num_original_couple_rpm]
            couple_rpm_axis.append([temp[0]+num_single_temp, temp[1]+num_single_temp])
        
        for i in range(num_original_couple_inj * (len(calibration_temp)-1)):
            temp = couple_inj_axis[-num_original_couple_inj]
            couple_inj_axis.append([temp[0]+num_single_temp, temp[1]+num_single_temp])
            
        return couple_rpm_axis, couple_inj_axis

    engine_map.couple_inj_axis, engine_map.couple_rpm_axis = couple_generator(injection_num)
    engine_map.rpm_window = calibration_rpms[1] - calibration_rpms[0]
    engine_map.inj_window = calibration_injs[1] - calibration_injs[0]
    
    constraints_loaded = constraints_load_excel('', CONSTRAINT_FILE)
    engine_map.calibration_max_sheet, engine_map.calibration_min_sheet = constraints_calibration_sheet(engine_map.df, constraints_loaded)
    
    temp = engine_map.calibration_max_sheet - engine_map.calibration_min_sheet
    assert(temp.min(axis=0)[3:3+6].min() > 0) # if there is not room for calibration
    
    
    #plot the 3d:
    def plot_3d():
        df = engine_map.df
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import axes3d
        x = np.array(calibration_temp)
        y = np.array(data['q0'].columns)
        x, y = np.meshgrid(x, y)
        z = np.tile(data['q0'].values, (len(calibration_temp),1)).T
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df.iloc[:,2], df.iloc[:,0], df.iloc[:,1], s = np.array(engine_map.factor)*1000, c='k')
        ax.plot_surface(x, y, z, linewidth=0, alpha=0.3, cmap='autumn_r')
        ax.set(xlabel='coolant temperature', ylabel='engine rpm', zlabel='Injection Qty (mg)', title='engine load points')
    plot_3d()
    
    return engine_map

#%%
def save_calibration_excel(engine_map, LOC, FILE):
    calibration_sheet = engine_map.df
    factor_df = engine_map.fac_map
    calibration_sheet = calibration_sheet.copy()
    writer = pd.ExcelWriter(LOC + FILE, engine='xlsxwriter')
    for column_name in calibration_sheet.columns[2:]:
        df = pd.pivot_table(calibration_sheet, index=['x2'],
                               columns=['x1'], values=column_name)
        df.to_excel(writer, sheet_name=column_name.replace('/','_'))
    factor_df.to_excel(writer, sheet_name='factor')
    writer.save()

#save_calibration_excel(engine_map, LOC, 'calibrations_with_smoothness.xlsx')
