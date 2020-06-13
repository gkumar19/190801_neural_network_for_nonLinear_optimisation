# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:25:41 2019

@author: KGU2BAN
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ml import ml_r2_list
from nn import nn_merge_predict
from utils import timestamp

def plot_main(fig_title, num_graphs, num_rows, num_columns, graph_title_list, main_func, bottom_func, left_func):
    '''function for plotting generic plot'''
    fig = plt.figure(fig_title, figsize=(15,10))
    fig.suptitle(fig_title, fontsize=16, color='g')
    num_xticks_plot = num_graphs - num_columns - 1
    for i in range(num_graphs):
        plt.subplot(num_rows,num_columns,i+1)
        plt.title(graph_title_list[i], color='b', fontsize='medium')
        main_func(i)
        if i > num_xticks_plot:
            bottom_func(i)
        if (i%num_columns == 0):
            left_func(i)


def plot_distribution(fig_title,num_rows, num_columns,graph_title_list,array):
    num_graphs = array.shape[1]
    plot_main(fig_title,num_graphs,num_rows,num_columns,
                graph_title_list,
                lambda i: (sns.distplot(array[:,i],hist_kws=dict(cumulative=False), kde=False),
                           plt.ylabel(''),
                           plt.xlabel(''),
                           plt.yticks([]),
                           plt.twinx(),
                           sns.distplot(array[:,i],kde_kws=dict(cumulative=True), hist=False)),
                lambda i: None,
                lambda i: None)
     
           
def plot_covariance_matrix(input_array, column_names):
    df = pd.DataFrame(input_array, columns=column_names)
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn_r")
    for item in g.get_yticklabels():
        item.set_rotation(45)
    for item in g.get_xticklabels():
        item.set_rotation(45)


def plot_bar(fig_title,num_rows, num_columns, graph_title_list, bar_label_list, array):
    num_graphs = array.shape[1]
    plot_main(fig_title,num_graphs,num_rows,num_columns,
                graph_title_list,
                lambda i: (plt.bar(bar_label_list, array[:,i]),
                           plt.ylabel(''),
                           plt.xlabel(''),
                           plt.yticks([]), plt.xticks(rotation=60, va = 'center')),
                lambda i: None,
                lambda i: None)


def plot_r2(fig_title,num_rows, num_columns, graph_title_list, df, df_shape):
    '''df should be stacked in column like: features,target_actual,target_predicted,test_train_classifier
    df_shape defines the tuple with number (num_features, num_targets)'''
    num_features, num_targets = df_shape
    num_graphs = num_targets
    plot_main(fig_title,num_graphs,num_rows,num_columns,
                graph_title_list,
                lambda i: (sns.scatterplot(df.columns[i+num_features],
                                           df.columns[i+num_features+num_targets],
                                           data=df.loc[df.iloc[:,i+num_features]>0,:], hue=df.columns[-1],linewidth=0,
                                           s=5, palette=sns.color_palette(['y','k'])),
                           plt.ylabel(''),
                           plt.xlabel(''), plt.yticks(), plt.xticks(rotation=60, va = 'center'),
                           plt.plot([df.iloc[:,i+num_features].min(),df.iloc[:,i+num_features].max()], [df.iloc[:,i+num_features].min(),df.iloc[:,i+num_features].max()], color='b'),
                           plt.xlim([df.iloc[:,i+num_features].min(),df.iloc[:,i+num_features].max()]),
                           plt.ylim([df.iloc[:,i+num_features].min(),df.iloc[:,i+num_features].max()])      ),
                lambda i: (plt.xlabel('actual')),
                lambda i: (plt.ylabel('predicted')))


def plot_scatter(fig_title,num_rows, num_columns, graph_title_list, x_axis, y_axis, data, model, N):
    '''x_axis and y_axis are array with num_graphs x data_point'''
    rpms = np.linspace(1000, 3500, 5)
    injs = np.linspace(40, 5, 5)
    rpms, injs = np.meshgrid(rpms, injs)
    rpms = np.ravel(rpms)
    injs = np.ravel(injs)
    dr_2 = abs((rpms[-1] - rpms[0])/8)
    di_2 = abs((injs[-1] - injs[0])/8)
    
    dataframe_actual = data['xy0']
    dataframe_actual['op'] = None
    for i, (rpm, inj) in enumerate(zip(list(rpms), list(injs))):
        condition1 = abs(rpm - data['e0']['epm_neng']).abs() <= dr_2
        condition2 = abs(inj - data['i0']['injctl_qsetunbal']).abs() <= di_2
        condition3 = (condition1 & condition2)
        dataframe_actual['op'][condition3] = i
    from nn import nn_predict
    pred = pd.DataFrame(nn_predict(model, N, data['x0'].values), columns=data['y0'].columns)
    pred = pd.concat([data['x0'], pred], axis=1)

    
    num_graphs = len(graph_title_list)
    plot_main(fig_title,num_graphs,num_rows,num_columns,
                graph_title_list,
                lambda i: (plt.scatter(x_axis[i,:],y_axis[i,:], s=0.4),
                           plt.ylabel(''),
                           plt.xlabel(''),
                           plt.xticks(rotation=60, va = 'center'),
                           sns.scatterplot('nox_g/h', 'russ_g/h', data=dataframe_actual[dataframe_actual['op']==i], linewidth=0, s=6, color='r'),
                           sns.scatterplot('nox_g/h', 'russ_g/h', data=pred[dataframe_actual['op']==i], linewidth=0, s=6, color='k')),
                lambda i: None,
                lambda i: None)

def plot_tricontourf(fig_title,num_rows, num_columns, graph_title_list, x_axis, y_axis, z_axis, x_axis_label, y_axis_label, x_ticks, y_ticks, colorbarmin, colorbarmax):
    num_graphs = z_axis.shape[0]
    plot_main(fig_title, num_graphs,num_rows, num_columns,
                        graph_title_list,
                        lambda i: (plt.tricontourf(x_axis, y_axis, z_axis[i,:],np.linspace(colorbarmin[i],colorbarmax[i],20), cmap='RdYlGn_r'),
                                   plt.colorbar(), plt.ylim(0,50/1000), plt.xticks([])),
                        lambda i: (plt.xticks(x_ticks, rotation=30), plt.xlabel(x_axis_label)),
                        lambda i: (plt.yticks(np.array(y_ticks).astype(int)), plt.ylabel(y_axis_label)))


def plot_losses(fig_title, history, x_y_lim=None):
    '''x_y_lim = (xlim_min,xlim_max,y_lim_min,y_lim_max)'''
    plt.figure(fig_title)
    if x_y_lim != None:
        a, b, c, d = x_y_lim
        plt.xlim(a, b)
        plt.ylim(c, d)
    stage1_number_of_epochs = len(history.history['loss'])
    plt.plot(range(stage1_number_of_epochs),history.history['loss'], label = 'train loss', color='k')
    plt.plot(range(stage1_number_of_epochs),history.history['val_loss'], label = 'validation loss', color='b')
    plt.xlabel('number of epochs')
    plt.ylabel('losses')
    plt.legend()

def plot_true_pred(data, model_base, N):
    df_check = nn_merge_predict(data['x_trn'].values, data['y_trn'].values, data['x_vld'].values, data['y_vld'].values, data['x_trn'].columns, data['y_trn'].columns, model_base, N)
    i = data['x_trn'].shape[1]
    j = data['y_trn'].shape[1]
    k = i+j
    r2_list_train = ml_r2_list(df_check[df_check['type'] == 'train'].values[:,i:k],
                               df_check[df_check['type'] == 'train'].values[:,k:-1])
    r2_list_test = ml_r2_list(df_check[df_check['type'] == 'test'].values[:,i:k],
                               df_check[df_check['type'] == 'test'].values[:,k:-1])
    r2_list = [n + ' ' + str(round(i, 2)) + ' '+ str(round(j,2)) for n,i,j in zip(data['y_trn'].columns, r2_list_train, r2_list_test)]
    plot_r2('sanity check ' + timestamp(), 4, 6, r2_list, df_check, (i, j))
    del df_check, r2_list, r2_list_test, r2_list_train, i, j, k