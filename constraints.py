# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:27:55 2019

@author: KGU2BAN
"""

import numpy as np
import pandas as pd
#%%
def constraints_constructor(epm_array, inj_array,epm_window, inj_window, data):
    constraints_dict = {}
    for cal_id, cal in enumerate(data['c0'].columns):
        epm_array = epm_array.astype(int)
        inj_array = inj_array.astype(int)
        def empty_dataframe():
            empty_dataframe = pd.DataFrame(np.zeros((inj_array.shape[0],
                                            epm_array.shape[0])),
                                    columns=epm_array,
                                    index=inj_array)
            return empty_dataframe
        df_max_cal = empty_dataframe()
        df_min_cal = empty_dataframe()
        for epm_id, epm in enumerate(epm_array):
            for inj_id, inj in enumerate(inj_array):
                dataframe = data['c0'][((epm-epm_window) < data['x0']['epm_neng'])
                                        & ((epm+epm_window) > data['x0']['epm_neng'])
                                        & ((inj-inj_window) < data['x0']['injctl_qsetunbal'])
                                        & ((inj+inj_window) > data['x0']['injctl_qsetunbal'])]
                df_max_cal.iloc[inj_id, epm_id] = dataframe.max()[cal]
                df_min_cal.iloc[inj_id, epm_id] = dataframe.min()[cal]
                if dataframe.shape[0] == 0:
                    df_max_cal.iloc[inj_id, epm_id] = df_max_cal.iloc[inj_id-1, epm_id]
                    df_min_cal.iloc[inj_id, epm_id] = df_min_cal.iloc[inj_id-1, epm_id]
                    
                constraints_dict[cal+'_max'] = df_max_cal
                constraints_dict[cal+'_min'] = df_min_cal
    return constraints_dict

#%%
def constraints_save_excel(constraints_dict, LOC, FILE):
        writer = pd.ExcelWriter(LOC + FILE, engine='xlsxwriter')
        print(LOC+FILE)
        for cal, dataframe in constraints_dict.items():
            dataframe.to_excel(writer, sheet_name=cal)
        writer.save()




#%%
def constraints_load_excel(LOC, FILE):
    constraints_dict = pd.read_excel(LOC+FILE, sheet_name=None, index_col=0)
    return constraints_dict


#%%
from scipy.interpolate import interp2d
def constraints_calibration_sheet(calibration_sheet, constraints_dict):
    calibration_max = calibration_sheet.copy()
    calibration_max.loc[:,3:] = 0
    calibration_min = calibration_sheet.copy()
    calibration_min.loc[:,3:] = 0
    for cal, df in constraints_dict.items():
        f = interp2d(df.columns,df.index, df.values.astype('float'), kind='linear')
        for index in calibration_sheet.index:
            epm, inj = calibration_sheet.iloc[index, 0], calibration_sheet.iloc[index, 1]
            if cal[-4:] == '_max':
                calibration_max.loc[index,cal[:-4]] = f(epm, inj)
            if cal[-4:] == '_min':
                calibration_min.loc[index,cal[:-4]] = f(epm, inj)
    return calibration_max, calibration_min

#%%
def minmax_constructor(epm_array, inj_array,epm_window, inj_window, data):
    minmax_dict = {'min': pd.DataFrame(np.zeros((epm_array.shape[0], data['y0'].shape[1])), columns=data['y0'].columns), 
                   'max': pd.DataFrame(np.zeros((epm_array.shape[0], data['y0'].shape[1])), columns=data['y0'].columns)}
    for num, bry in enumerate(data['y0'].columns):
        for i, (epm, inj) in enumerate(zip(list(epm_array), list(inj_array))):
            dataframe = data['y0'][((epm-epm_window) < data['x0']['x1'])
                                    & ((epm+epm_window) > data['x0']['x1'])
                                    & ((inj-inj_window) < data['x0']['x2'])
                                    & ((inj+inj_window) > data['x0']['x2'])]
            if dataframe.shape[0] != 0:
                minmax_dict['min'].iloc[i, num] = dataframe.min()[bry]
                minmax_dict['max'].iloc[i, num] = dataframe.max()[bry]
    return minmax_dict

#%%
if __name__ == '__main__':
    minmax = minmax_constructor(np.linspace(1000,4000,8),
                                         np.linspace(10,50,8),750, 7.5, data)

#%%

if __name__ == '__main__':
    
    constraints = constraints_constructor(np.linspace(1000,4000,8),
                                         np.linspace(10,50,7),750, 7.5, data)
    constraints_save_excel(constraints, LOC, 'constraints.xlsx')
    constraints_loaded = constraints_load_excel(LOC, 'constraints.xlsx')
    calibration_max, calibration_min = constraints_calibration_sheet(df, constraints_loaded)
