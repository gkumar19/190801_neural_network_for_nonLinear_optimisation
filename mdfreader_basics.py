"""
Created on Wed May 20 17:42:05 2020

@author: Gaurav
"""

import mdfreader
import pandas as pd

#Loading dat file

yop = mdfreader.Mdf('jan_hot.dat') #This is the dat file
yop.resample(0.1) #resampling the data to 0.1s
channel_list = list(yop.keys()) 

df = pd.DataFrame(columns=channel_list) #pandas dataframe to store dat file content
for channel in channel_list:
    values = yop.get_channel_data(channel) #to load entire dat file , an expensive operation
    df[channel] = values


channel_list_filter = ['Epm_nEng', 'InjCtl_qSetUnBal', 'CEngDsT_t',
                'InjCrv_qPiI1Des_mp', 'InjCrv_phiMI1Des', 'InjCrv_tiPiI1Des',
                 'RailP_pFlt','AFS_mAirPerCyl', 'EnvP_p']

df_filter = pd.DataFrame(columns=channel_list_filter) #pandas dataframe to store dat file content
for channel in channel_list_filter:
    values = yop.get_channel_data(channel) #to load selected channels from dat file, an inexpensive operation
    df_filter[channel] = values

#loading ascii file
df_ascii = pd.read_csv('jan_hot.asc', encoding='mbcs', sep=' ', low_memory=False)
