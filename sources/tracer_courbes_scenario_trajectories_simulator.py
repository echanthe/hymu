# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:13:08 2018

@author: @author: AA_OJ_SM
"""

import os
import pandas
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import *


#se mettre sur le dossier de travail : pour le lancer sur Spyder , sinon le commenter
#os.chdir('C:/Users/abdel_000/Desktop/2018 Projet Long N7/hymu')

# scenarios mode simulateur
'2018-02-01_16-05-18'
'scenario 1 --  2018-02-02_16-17-04'
'scenario 7 --  2018-02-02_16-18-17'
'scenario 9 --  2018-02-02_16-18-53'

file = './2019-03-27_13-31-10.csv'

data_simu = pandas.read_csv(file, sep='|')
print(data_simu)
print(data_simu.iloc[:,3])


def draw_scenario_trajectory(filedata, toshow = True, loc='best'):

    data = filedata
    # correct time
    datat = []
    for t in data.t:
        datat.append(t - data.t[0])

    modes = sorted(set(data.m))

    # map categories to y-values
    categories = modes
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))
    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i) for i in categories]))

    fig = figure(2, figsize=(12, max(3,round(len(modes)/1.5))), dpi=80, facecolor='w', edgecolor='k')
    plot(datat, [cat_dict[m] for m in data.m], '-', linewidth = 2, color='k')

    val_dict = {v: s.replace(' + ', '\n+ ') for v,s in val_dict.items()}
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))

    grid(True)
    #if not future:
    #    title("Mode trajectories at time {}s".format(round(datat[k],3)))
    #else:
    #    title("Mode trajectories at time 781s and future trajectories".format(round(datat[k],3)))
    xlabel('time (s)')
    ylabel('mode')
    gca().axes.get_yaxis().set_ticks(list(range(1, len(val_dict)+1))) # force y axis labels
    ylim(min(val_dict)-0.2, max(val_dict) + .2)
    #ylim(6.8, 8.2)
    #lim(4890, 5089)
    xlim(min(datat), max(datat))
    #lim(0, 8200)
    rc('font', size=20)
    fig.tight_layout()
    
    if toshow:
        show()

draw_scenario_trajectory(data_simu, toshow = True, loc='best')