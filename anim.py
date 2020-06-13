# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:50:35 2020

@author: KGU2BAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from utils import timestamp, melt2pivot, average_emission

LOC = 'simulation_run/'
LOC = LOC + '[Fri_Jun_12_14_09_07_2020]/'
FILE_CALIBRATION = 'calibration.npy'
FILE_LOSSES = 'losses.xlsx'
FILE_TARGET = 'targets.npy'
FILE_CYCLE_CALIBRATION = 'cycle_calibration.npy'
FILE_CYCLE_PATTERN = 'cycle_pattern.xlsx'
FILE_CYCLE_TARGET = 'cycle_targets.npy'
FILE_DETAILS = 'details.xlsx'

calibration_progress = (pd.read_excel(LOC+FILE_LOSSES, index_col=0),
                        np.load(LOC+FILE_CALIBRATION),
                        np.load(LOC+FILE_TARGET))
cycle_progress = (pd.read_excel(LOC+FILE_CYCLE_PATTERN, index_col=0),
                  np.load(LOC+FILE_CYCLE_CALIBRATION),
                  np.load(LOC+FILE_CYCLE_TARGET))
details = pd.read_excel(LOC+FILE_DETAILS, index_col=None, sheet_name=None)
array, rpm_index, inj_col = melt2pivot(calibration_progress[1] ,calibration_progress[2], 7)
emission_progress_km_h, emission_progress_avg_km_h = average_emission(calibration_progress[2],details['factor'].values,
                                                                      details['avg_speed'].values[0,0], details['cycle_time'].values[0,0])

del FILE_CALIBRATION, FILE_CYCLE_CALIBRATION, FILE_CYCLE_PATTERN, FILE_CYCLE_TARGET, FILE_DETAILS, FILE_LOSSES, FILE_TARGET
#%%
class Animate():
    def __init__(self, title, grids = (27,60), figsize=(20,9), fontsize=20, color='b', save=False):
        '''
        sequence of animated plotting is fixed and has to be like that:
        animated scatter
        animated line
        animated tricontour
        animated tricontourf
        animated bar
        animated surface plot        
        '''
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.fig.set_facecolor('ghostwhite')
        self.gs = gridspec.GridSpec(ncols=grids[1], nrows=grids[0], figure=self.fig)
        self.fig.suptitle(title, fontsize=fontsize, color=color)
        self.save = save
        self.ax = []
        self.ax_nums = 0
        self.data = []
        self.data_nums = 0
        self.data_name = {}
        self.scats = []
        self.scat_nums = 0
        self.lines = []
        self.line_nums = 0
        self.contours = []
        self.contour_nums = 0
        self.contourfs = []
        self.contourf_nums = 0
        self.bars = []
        self.bar_nums = 0
        self.surfaces = []
        self.surface_nums = 0
        self.color_bars = []
        self.gbars = []
        self.gbar_nums = 0
    def load_data(self, data, name, data_reduce=100):
        '''
        first axis: time frame
        second axis: operating point
        '''
        shortened_data = np.zeros_like(data)[:100]
        window = int(data.shape[0]/data_reduce)
        for i in range(data_reduce):
            shortened_data[i] = data[i*window]
        self.data.append(shortened_data)
        self.data_name[name] = self.data_nums
        self.data_nums = self.data_nums + 1
    def place_axes(self,grid_loc= (0,0,0,0), projection = '2d'):
        if projection == '2d':
            self.ax.append(self.fig.add_subplot(self.gs[grid_loc[0]:grid_loc[1]+1, grid_loc[2]:grid_loc[3]+1]))
        if projection == '3d':
            self.ax.append(self.fig.add_subplot(self.gs[grid_loc[0]:grid_loc[1]+1, grid_loc[2]:grid_loc[3]+1], projection ='3d'))
        self.ax[-1].set_facecolor('ghostwhite')
        self.ax[-1].grid()
        self.ax_nums = self.ax_nums + 1
    def add_axes(self,ax_num):
        self.ax.append(self.ax[ax_num].twinx())
        self.ax[-1].set_facecolor('ghostwhite')
    def place_title(self, ax_num, title):
        self.ax[ax_num].set_title(title[0])
        self.ax[ax_num].set_xlabel(title[1])
        self.ax[ax_num].set_ylabel(title[2])
    def set_animate_scatter(self,ax_num, color = 'r', s=10):
        self.scats.append(self.ax[ax_num].scatter([],[],color = color, s=s))
        self.scat_nums = self.scat_nums + 1
    def set_animate_line(self,ax_num, color = 'b'):
        self.lines.append(self.ax[ax_num].plot([],[])[0])
        self.line_nums = self.line_nums + 1
    def set_animate_contour(self,ax_num):
        self.contours.append(self.ax[ax_num])
        self.contour_nums = self.contour_nums + 1
    def set_animate_contourf(self,ax_num):
        self.contourfs.append(self.ax[ax_num])
        self.contourf_nums = self.contourf_nums + 1
    def set_animate_bar(self,ax_num, names, width=0.3):
        self.bars.append(self.ax[ax_num].bar(names, [0]*len(names), width=[width]*len(names)))
        self.bar_nums = self.bar_nums + 1
    def set_animate_surface(self, ax_num):
        self.surfaces.append(self.ax[ax_num])
        self.surface_nums = self.surface_nums + 1
    def set_animate_gbar(self,ax_num, rpm_index, inj_col, width=0.1):
        ax = self.ax[ax_num]
        x = np.arange(len(rpm_index))
        bar_list = []
        for i, inj in enumerate(inj_col):
            bar_list.append(ax.bar(x + i*width, [0]*len(rpm_index), width, label=inj, color=cm.tab20(i*0.1)))
            plt.legend()
            plt.xlabel('rpm')
        self.gbars.append(bar_list)
        self.gbar_nums = self.gbar_nums + 1
        
    def animate_all(self, data_name_list):
        data_seq = [self.data_name[item] for t in data_name_list for item in t]
        def animate(i):
            data_num = 0
            for cb in self.color_bars:
                cb.remove()
                self.color_bars = []
            
            for scat in self.scats:
                x = self.data[data_seq[data_num]][i,:, None]
                y = self.data[data_seq[data_num+1]][i,:, None]
                xy = np.concatenate([x, y], axis=1)
                scat.set_offsets(xy)
                data_num = data_num + 2
            for line in self.lines:
                x = self.data[data_seq[data_num]][i,:, None]
                y = self.data[data_seq[data_num+1]][i,:, None]
                line.set_data(x,y)
                data_num = data_num + 2
            for contour in self.contours:
                x = self.data[data_seq[data_num]][i,:]
                y = self.data[data_seq[data_num+1]][i,:]
                z = self.data[data_seq[data_num+2]][i,:]
                z_min, z_max = np.min(self.data[data_seq[data_num+2]]), np.max(self.data[data_seq[data_num+2]])
                contour.collections = []
                temp = contour.tricontour(x, y, z,levels = np.linspace(z_min, z_max, 20), cmap = 'RdYlGn_r', norm=plt.Normalize(z_min, z_max))
                if i == 0:
                    self.fig.colorbar(temp, ax=contour)
                data_num = data_num + 3
            for contourf in self.contourfs:
                x = self.data[data_seq[data_num]][i,:]
                y = self.data[data_seq[data_num+1]][i,:]
                z = self.data[data_seq[data_num+2]][i,:]
                z_min, z_max = np.min(self.data[data_seq[data_num+2]]), np.max(self.data[data_seq[data_num+2]])
                contourf.collections = []
                temp = contourf.tricontourf(x, y, z,levels = np.linspace(z_min, z_max, 20), cmap = 'RdYlGn_r')
                #self.color_bars.append(self.fig.colorbar(temp, ax=contourf))
                if i == 0:
                    self.fig.colorbar(temp, ax=contourf)
                data_num = data_num + 3
            for bar in self.bars:
                h = self.data[data_seq[data_num]]
                [br.set_height(height) for br,height in zip(bar,h[i])]
                data_num = data_num + 1
            for surface in self.surfaces:
                surface.collections = []
                x = self.data[data_seq[data_num]][i, :]
                y = self.data[data_seq[data_num+1]][i, :]
                z = self.data[data_seq[data_num+2]][i, :]
                triang = mtri.Triangulation(x, y)
                temp = surface.plot_trisurf(triang, z, cmap='coolwarm')
                surface.scatter(x,y,z, marker='.', s=25, c="black", alpha=0.5)
                total_time = self.data[data_seq[0]].shape[0]-1
                surface.view_init(elev=33, azim=-58 - 20*np.sin(i*2*np.pi/total_time))
                #self.color_bars.append(self.fig.colorbar(temp, ax=surface))
                #if i == 0:
                #    self.fig.colorbar(temp, ax=surface)
                data_num = data_num + 3
            for bar_list in self.gbars:
                for inj_num, bar in enumerate(bar_list):
                    array = self.data[data_seq[data_num]]
                    h = array[:,:,inj_num]
                    [br.set_height(height) for br,height in zip(bar,h[i])]
                data_num = data_num + 1
            plt.savefig(LOC+'animate_'+'/image_{:03d}.png'.format(i))
        if self.save == False:
            self.anim = FuncAnimation(self.fig, animate, interval=10, frames=self.data[data_seq[0]].shape[0]-1)
        if self.save == True:
            #self.anim.save(LOC+'animate_'+timestamp()+'.gif', writer = 'pillow')
            for i in range(100):
                animate(i)
            

#%% animate
animate0 = Animate('AI Self Calibration', grids=(2,4), save=True, figsize=(20,9))
array, rpm_index, inj_col = melt2pivot(calibration_progress[1] ,calibration_progress[2], 5)

animate0.load_data(np.tile(np.arange(1181)[None,:],(100,1)) , 'time')
animate0.load_data(np.linspace(0,100,num=1000)[:,None], 'cal_pro')
animate0.load_data(calibration_progress[2][:,:,0], 'nox')
animate0.load_data(calibration_progress[2][:,:,2] , 'russ')
animate0.load_data(calibration_progress[2][:,:,1], 'hc')
animate0.load_data(calibration_progress[2][:,:,3], 'co')
animate0.load_data(cycle_progress[2][:,:,0], 'nox_cycle')
animate0.load_data(cycle_progress[2][:,:,1], 'hc_cycle')
animate0.load_data(cycle_progress[2][:,:,2], 'russ_cycle')
animate0.load_data(cycle_progress[2][:,:,2], 'co_cycle')
animate0.load_data(calibration_progress[1][:,:,0], 'rpm')
animate0.load_data(calibration_progress[1][:,:,1], 'qty')
animate0.load_data(calibration_progress[1][:,:,7], 'railp')
animate0.load_data(emission_progress_km_h[:,:,0], 'nox_km/h')
animate0.load_data(emission_progress_avg_km_h[:,:,0], 'nox_avg_km/h')
animate0.load_data(emission_progress_km_h[:,:,1], 'hc_km/h')
animate0.load_data(emission_progress_avg_km_h[:,:,1], 'hc_avg_km/h')
animate0.load_data(emission_progress_km_h[:,:,2], 'russ_km/h')
animate0.load_data(emission_progress_avg_km_h[:,:,2], 'russ_avg_km/h')
animate0.load_data(emission_progress_km_h[:,:,3], 'co_km/h')
animate0.load_data(emission_progress_avg_km_h[:,:,3], 'co_avg_km/h')
animate0.load_data(array, 'array')


animate0.place_axes([0,0,0,1])
animate0.place_title(0,['nox_on_cycle', 'time(s)', 'nox (g/h)'])
animate0.ax[0].set(ylim=(0, 120))

animate0.add_axes(0)
animate0.ax[1].plot(np.arange(1181), cycle_progress[0]['v1'], color='r')

animate0.place_axes([0,0,2,2], projection='3d')
animate0.place_title(2,['rail pressure calibration (bar)', 'rpm', 'injection quantity (mg/hub)'])


animate0.place_axes([0,0,3,3])
animate0.place_title(3,['nox_russ tradeoff during self calibration', 'nox (g/h)', 'russ (g/h)'])
animate0.ax[3].set(xlim=(0,2), ylim=(0, 0.2))
animate0.ax[3].plot([0,0.22, 0.22], [0.025,0.025, 0], color='k')

animate0.place_axes([1,1,0,0])
animate0.place_title(4,['nox (g/h)', 'rpm', 'injection quantity'])

animate0.place_axes([1,1,1,1])
animate0.place_title(5,['russ (g/h)', 'rpm', 'injection quantity'])

animate0.place_axes([1,1,2,3])
animate0.place_title(6,['boundary condition: Smoke', '_', 'smoke (FSN)'])
animate0.ax[6].set(ylim=(0, 3))
animate0.ax[6].plot([-1, 10], [2,2])

animate0.set_animate_scatter(3, color='r', s=details['factor'].values*1000)
animate0.set_animate_scatter(3, color='b', s =150)
animate0.set_animate_line(0)
animate0.set_animate_contourf(4)
animate0.set_animate_contourf(5)
animate0.set_animate_surface(2)
animate0.set_animate_gbar(6, rpm_index, inj_col)

data = [['nox_km/h', 'russ_km/h'],
        ['nox_avg_km/h', 'russ_avg_km/h'],
        ['time', 'nox_cycle'],
        ['rpm', 'qty', 'nox'],
        ['rpm', 'qty', 'russ'],
        ['rpm', 'qty', 'railp'],
        ['array']]

animate0.animate_all(data)

#%% plot progress bar
fig, ax = plt.subplots(figsize=(10,1))  
bar = ax.barh(['progress'], [0], color='tab:blue')
ax.set(xlim=(0,100), ylim=(-0.2,0.2), xticks=[])

def animate(i):
    array = np.arange(100)
    array[-10:] = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    ax.clear()
    ax.barh(['progress'], [array[i]], color='tab:blue')
    ax.set(xlim=(0,100), ylim=(-0.2,0.2), xticks=[])
    ax.text(array[i], 0, f'{int(min(i/0.89, 100))} %', color='black', fontweight='bold', size=20)

anim = FuncAnimation(fig, animate, interval=10, frames=100)
anim.save('simulation_run/progress.gif', writer='pillow')