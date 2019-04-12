# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:38:28 2018

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


#print(data_simu)
#print(data_simu.iloc[:,3])



def draw_health_mode(fileCSV, figName, toshow = True, reverse = False):

    data_simu = pandas.read_csv(fileCSV, sep='|')
    scenario_data = data_simu

    fig = plt.figure(fileCSV, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    
    categories = sorted([m for m in set(scenario_data.m)], key=lambda x:x[1])
    categories=sorted(categories)
    # map categories to y-values
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i).replace(',',',\n') for i in categories]))

    plotval = [cat_dict[m] for m in scenario_data.m]
    if reverse :
        plt.plot(plotval, scenario_data.t, 'k-', lw=4)
    else:
        plt.plot(scenario_data.t, plotval)
    if reverse :
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
        plt.gca().invert_yaxis()
    else :
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    plt.grid(True)
    plt.title("Real health mode")
    if reverse :
        plt.ylabel('time (s)')
        plt.xlabel('health mode')
        plt.xlim(.75, len(categories) +0.25)
    else:
        plt.xlabel('time (s)')
        plt.ylabel('health mode')
        plt.ylim(.75, len(categories) +0.25)
    rc('font', size=20)
    
    #def proj(i) :
    #    x = (t[i-1] + t[i])/2
    #    y = (cat_dict[data['real behavioral mode'][i-1]] + cat_dict[data['real behavioral mode'][i]])/2
    #    return x,y

    #events = data['events']
    #diagevents = data['observed events']
    #labels = {}
    #for i in range(len(t)):
    #    if events[i] != 'no event':
    #        x,y = proj(i)
    #        style = 'sawtooth,pad=0.3'
    #        fc = 'red'
    #        if diagevents[i] == events[i]:
    #            style = 'round4,pad=0.3'
    #            fc = 'white'
    #        if reverse :
    #            labels[i] = annotate(events[i], xy=(y,x), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    #        else:
    #            labels[i] = annotate(events[i], xy=(x,y), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.savefig(figName)
    if toshow:
        plt.ion()
        plt.show()
        
       


