import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from matplotlib import style
style.use('seaborn')

'''importing my own libraries'''
import sys
sys.path.append('C:/Users/kgu2ban/Desktop/AI/modules')
from plot import plot_distribution, plot_covariance_matrix, plot_bar, plot_r2, plot_scatter, plot_tricontourf, plot_losses
from ml import ml_relevance_matrix_etr, ml_r2_list
from nn import nn_merge_predict, nn_all_cal_check, nn_model_base, nn_model2, nn_model2_training, nn_loss_model1, nn_model2_constraints
from helper import helper_dataframe
from utils import plot_sweep, load_excel, timestamp
from dyn import steady_indexer
#%%

class gui:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.state('zoomed')
        self.main_window.title('AIDEC:')
        self.main_window.minsize(1900,950)
        
        #tab for the main window
        self.tab_style = ttk.Style(self.main_window)
        
        self.tab_style.configure('TNotebook.Tab', font=('URW Gothic L', '11', 'bold'), width=30, sticky='e')
        self.tab_style.configure('lefttab.TNotebook', tabposition='wn', background='grey50', tabmargins=[0,0,0,0])
        self.main_tab = ttk.Notebook(self.main_window, style='lefttab.TNotebook')
        self.main_tab.pack(expand=1, fill='both')
    
    def tab(self, tab_para):
        self.num_tab = len(tab_para)
        self.tab_list = []
        for i in range(self.num_tab):
            self.tab_list.append(tk.Frame(self.main_tab, bg=tab_para[i]['bg'], borderwidth=tab_para[i]['borderwidth'], relief=tab_para[i]['relief']))
            self.main_tab.add(self.tab_list[i], text=tab_para[i]['text'])
    
    def frame(self, frame_para):
        self.frame_list_list = []
        for tab in range(self.num_tab):
            self.frame_list_list.append([])
            for frame in range(len(frame_para[tab])):
                self.frame_list_list[tab].append(tk.Frame(self.tab_list[tab], bg=frame_para[tab][frame]['bg'],
                                    borderwidth=1, relief="groove", width = frame_para[tab][frame]['width'],
                                    height = frame_para[tab][frame]['height']))
                self.frame_list_list[tab][frame].grid(row=frame_para[tab][frame]['row'], column=frame_para[tab][frame]['column'],
                                                        padx=frame_para[tab][frame]['padx'], pady=frame_para[tab][frame]['pady'],
                                                        columnspan=frame_para[tab][frame]['cs'])
                self.frame_list_list[tab][frame].grid_propagate(0)
    
    def label_frame(self):
        for tab in range(self.num_tab):
            for f, frame in enumerate(self.frame_list_list[tab]):
                label = tk.Label(frame, text=str(f))
                label.grid()
    
    def status(self):#status bar
        status_bar = tk.Label(self.main_window, text='status remark', bd=1, anchor='w', relief='sunken')
        status_bar.pack(side='bottom', fill='x')
    
    def mainloop(self):
        self.main_window.mainloop()
    
    def tab0(self):
        def open_file():
            string = filedialog.askopenfilename(initialdir = "/",title = "Select global doe file",
                                                filetypes = (("excel files","*.xlsx"),("all files","*.*")))
            self.entry_data.insert(0,string)
        
        def load_file():
            self.loc_data = self.entry_data.get()
            global data, N
            data, N = load_excel(self.loc_data,'')
        
        def plot_tkinter():
            fig = Figure(figsize=(15, 7), tight_layout=True)
            ax = fig.add_subplot(111)
            sns.scatterplot(choice_var0.get(), choice_var1.get(), data=data['xy0'], size=choice_var2.get(), hue=choice_var3.get(), linewidth=0, cmap='RdYlGn_r', ax=ax)
            canvas = FigureCanvasTkAgg(fig, self.frame_list_list[0][2])
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)
            #toolbar
            toolbarFrame = tk.Frame(self.frame_list_list[0][2])
            toolbarFrame.grid(row=2,column=0)
            toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
            toolbar.update()
            #canvas.get_tk_widget().pack(expand=1)
        
        
        self.entry_data = tk.Entry(self.frame_list_list[0][0], width=55, borderwidth=5)
        self.entry_data.grid(row=2, column=2, padx=20)
        self.entry_data.insert(0,'C:/Users/kgu2ban/Desktop/AI/modules/p15.xlsx')
        
        open_button = tk.Button(self.frame_list_list[0][0], text='open file', command=open_file)
        open_button.grid(row=2, column=3, padx=20)
        
        load_button = tk.Button(self.frame_list_list[0][0], text='load file', command=load_file)
        load_button.grid(row=2, column=4, padx=20)
        
        option_list = tuple(data['xy0'].columns)
        
        tk.Label(self.frame_list_list[0][1], text='x-axis:', bg='grey50').grid(row=2,column=2, sticky='e')
        tk.Label(self.frame_list_list[0][1], text='y-axis:').grid(row=2,column=4, sticky='e')
        tk.Label(self.frame_list_list[0][1], text='size:').grid(row=3,column=2, sticky='e')
        tk.Label(self.frame_list_list[0][1], text='hue:').grid(row=3,column=4, sticky='e')
        
        choice_var0 = tk.StringVar()
        choice_var1 = tk.StringVar()
        choice_var2 = tk.StringVar()
        choice_var3 = tk.StringVar()
        
        choice_var0.set('nox_g/h')
        choice_var1.set('russ_g/h')
        choice_var2.set('injctl_qsetunbal')
        choice_var3.set('epm_neng')
        
        tk.OptionMenu(self.frame_list_list[0][1], choice_var0, *option_list).grid(row=2, column=3, padx=20, sticky='w', pady=10)
        tk.OptionMenu(self.frame_list_list[0][1], choice_var1, *option_list).grid(row=2, column=5, padx=20, sticky='w', pady=10)
        tk.OptionMenu(self.frame_list_list[0][1], choice_var2, *option_list).grid(row=3, column=3, padx=20, sticky='w', pady=10)
        tk.OptionMenu(self.frame_list_list[0][1], choice_var3, *option_list).grid(row=3, column=5, padx=20, sticky='w', pady=10)
        
        plot_button = tk.Button(self.frame_list_list[0][1], text='plot', command=plot_tkinter, width=20)
        plot_button.grid(row=2, column=6, padx=20, rowspan=2)
    
def w(fac):
    return 1600*fac
def h(fac):
    return 950*fac

tab_para = [{'bg':'grey74', 'borderwidth':3, 'relief':'groove', 'text':'load global doe'},
                 {'bg':'grey74', 'borderwidth':3, 'relief':'groove', 'text':'model steady state'},
                 {'bg':'grey74', 'borderwidth':3, 'relief':'groove', 'text':'load dynamic file'}]

frame_para = []
frame_para.append([{'bg':'grey75', 'width':w(0.5), 'height':h(0.2), 'row':0, 'column':0, 'padx':0, 'pady':0, 'cs':1, 'rs':1},
                   {'bg':'grey75', 'width':w(0.5), 'height':h(0.2), 'row':0, 'column':1, 'padx':0, 'pady':0, 'cs':1, 'rs':1},
                   {'bg':'grey80', 'width':w(1), 'height':h(0.8), 'row':1, 'column':0, 'padx':2, 'pady':2, 'cs':2, 'rs':1}])

frame_para.append([{'bg':'yellow', 'width':200, 'height':200, 'row':0, 'column':0, 'padx':2, 'pady':2, 'cs':1},
                   {'bg':'blue', 'width':200, 'height':200, 'row':0, 'column':1, 'padx':2, 'pady':2, 'cs':1}])

frame_para.append([{'bg':'blue', 'width':200, 'height':200, 'row':0, 'column':0, 'padx':2, 'pady':2, 'cs':1},
                   {'bg':'yellow', 'width':200, 'height':200, 'row':1, 'column':1, 'padx':2, 'pady':2, 'cs':1}])

gui1 = gui()
gui1.tab(tab_para)
gui1.frame(frame_para)
gui1.label_frame()
gui1.status()
gui1.tab0()
gui1.mainloop()

#%%

root = tk.Tk()

option_list = ["Option 1", "Option 2", "Option 3", "Option 4"]
choice_var = tk.StringVar()
choice_var.set(option_list[0])
tk.OptionMenu(root, choice_var, *option_list).pack()

root.mainloop()

#%%

from tkinter import *
root = Tk()
topframe = Frame(root)
topframe.pack()
bottomframe = Frame(root)
bottomframe.pack(side='bottom')
button1 = Button(topframe, text='button 1', bg='red', fg='yellow')
button2 = Button(topframe, text='button 2')
button3 = Button(topframe, text='button 3')
button1.pack(side='left')
button2.pack(side='left', padx=50)
button3.pack(side='left', fill='x')
root.title("Welcome to LikeGeeks app")
thelabel = Label(root, text='this is text', bg='red')
thelabel.pack(fill='x')
thelabel2 = Label(root, text='this is text2', bg='yellow')
thelabel2.pack(side='left', fill='y')
root.mainloop()
#%%
root2 = Tk()
label1 = Label(root2, text='name')
label2 = Label(root2, text='password')
entry1 = Entry(root2)
entry2 = Entry(root2)
label1.grid(row=0,column=0, sticky='e')
label2.grid(row=1,column=0)
entry1.grid(row=0,column=1)
entry2.grid(row=1,column=1)
c = Checkbutton(root2, text='keep me signed in')
c.grid(columnspan=2)
root2.mainloop()
#%%
def printname():
    print('hello world')
root3 = Tk()
button1 = Button(root3, text='button 1', command=printname)
button1.pack()
root3.mainloop()


#%%
def printname1(event):
    print('hello world1')
def printname2(event):
    print('hello world2')
def printname3(event):
    print('hello world3')

root3 = Tk()
button1 = Button(root3, text='button 1')
button1.bind('<Button-1>', printname1)
button1.bind('<Button-2>', printname2)
button1.bind('<Button-3>', printname3)
button1.pack()
root3.mainloop()

#%%
top = Tk() 
Lb = Listbox(top) 
Lb.insert(1, 'Python') 
Lb.insert(2, 'Java') 
Lb.insert(3, 'C++') 
Lb.insert(4, 'Any other') 
Lb.pack() 
top.mainloop()

#%%
def do_nothing():
    print('ok ok doing nothing')

master = Tk()
menu = Menu(master)
master.config(menu=menu)
sub_menu = Menu(menu)
menu.add_cascade(label='file', menu=sub_menu)
sub_menu.add_command(label='new', command=do_nothing)

status = Label(master, text='prepare to do nothing...', bd=1, anchor='w', relief='sunken')
status.pack(side='bottom', fill='x')
master.mainloop()

#%%
from tkinter.messagebox import showinfo, askquestion, hfg
master = Tk()

showinfo('whatever', 'okay bro')
answer = askquestion('whatever', 'okay bro')
print(answer)
master.mainloop()
#%%
master = Tk()
canvas = Canvas(master, width=400, height=300)
canvas.pack()
master.mainloop()

#%%
master = Tk()
photo = PhotoImage(file="C:/Users/Gaurav/Desktop/1.png")
label = Label(master, image=photo)
label.pack()
master.mainloop()

#%% calculator

def button_add():
    pass

root = Tk()
root.title('my calculator')
e = Entry(root, width=35, borderwidth=5)
e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
e.insert(100,0)
button_1 = Button(root, text='1', padx=40, pady=5, command=button_add)
button_2 = Button(root, text='2', padx=40, pady=5, command=button_add)
button_3 = Button(root, text='3', padx=40, pady=5, command=button_add)
button_clear = Button(root, text='clear', padx=40)
button_plus = Button(root, text='+', padx=40)
button_equal = Button(root, text='=', padx=40)
button_1.grid(row=1, column=0)
button_2.grid(row=1, column=1)
button_3.grid(row=1, column=2)
button_clear.grid(row=2, column=0, columnspan=2)
button_plus.grid(row=3, column=0, columnspan=2)
button_equal.grid(row=2, column=2, rowspan=2)

root.mainloop()
