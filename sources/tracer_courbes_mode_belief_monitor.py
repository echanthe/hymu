# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:46:52 2018

@author: AA_OJ_SM
"""


import os
import pandas
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import *
import gzip
import itertools as it
from matplotlib.legend_handler import HandlerLine2D

# se mettre sur le dossier de travail : pour le lancer sur Spyder , sinon le commenter
#os.chdir('C:\\Users\\ztot_\\Documents\\Salma Mouline\\ENSEEIHT\\Projet long N7\\hymu')



##afficher les  estimations :

    
def draw_mode_belief(fileGZ, figName, minimumB = 0, toshow = True, loc='best', selectedModes = None, minimum = 1, blacklist = None):

    #data = read_data(filename)
    
   
    data_moni = pandas.read_csv(fileGZ,compression='gzip', header=0, sep='|', quotechar=';', error_bad_lines=False)    
    #for j in (data_moni.iloc[:,3]) :
        #print(j + '\n')
    data=data_moni

    # correct time
    datat = []
    for t in data.t:
        datat.append(t - data.t[0])

    modes = sorted([m for ms in data['m'] for m in ms.split(';')])
    modes = set(m for m, nb in it.groupby(modes) if len(list(nb)) > minimum)
    modes = sorted(modes)
    #print(len(modes), 'modes')
    
    if selectedModes is not None:
        modes = sorted(selectedModes)

    if blacklist is not None:
        for m in blacklist:
            modes.remove(m)

    # map categories to y-values
    categories = modes
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))
    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i).replace(',',',\n') for i in categories]))

    def get_max_modes(i):
        bs = (float(b) for b in data.b[i].split(';'))
        ms = [cat_dict[m] for m in data.m[i].split(';') if cat_dict.get(m) is not None] # takes only selected modes
        res = {m:0 for m in set(ms)}
        for m, b in zip(ms, bs):
            if res.get(m) < b:
                res[m] = b
        return res

    # scatter version
    # color : (times, modes, sizes)
    res = {'k': ([], [], []), 'c' : ([], [], [])}
    for i, t in enumerate(datat):
        ms = get_max_modes(i) # each mode max belief
        if len(ms.values())>0:
            maxi = max(ms.values()) # max belief at t
        for m, b in ms.items():
            if b != 0:
                color = 'k' 
                if b == maxi: color = 'c'
                res[color][0].append(t)
                res[color][1].append(m)
                res[color][2].append(b/maxi * 200)

    fig = figure(figName, figsize=(12, max(3,round(len(modes)/1.3))), dpi=80, facecolor='w', edgecolor='k')
    c = 'k'
    scatter(res[c][0], res[c][1], s=res[c][2], c=c, marker='|')
    c = 'c'
    scatter(res[c][0], res[c][1], s=res[c][2], c=c, marker='|')

    l1, = plot([], [], 'k-', lw=10, label='mode belief')
    l2, = plot([], [], 'c-', lw=10, label='higher belief')
    legend(loc=loc, handler_map={line: HandlerLine2D(numpoints=1) for line in [l1,l2]}, prop={'size':30})
    
    val_dict = {v: s.replace(' + ', '\n+ ') for v,s in val_dict.items()}
    val_dict = {v: s.replace('BR FL', 'BR\nFL') for v,s in val_dict.items()}
    val_dict = {v: s.replace('FL FR', 'FL\nFR') for v,s in val_dict.items()}
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))

    grid(True)
    #title("Belief degrees of modes")
    xlabel('time (s)')
    ylabel('mode')
    ylim(min(val_dict)-0.2, max(val_dict) + .2)
    xlim(min(datat), max(datat))
    rc('font', size=30)
    fig.tight_layout()
    gca().axes.get_yaxis().set_ticks(list(range(1, len(val_dict)+1))) # force y axis labels

    for ax in fig.axes: ax.set_axisbelow(True)
    plt.savefig(figName)
    if toshow:
        plt.ion()       
        plt.show()        

        '''show()
        with PdfPages('mode_belief.pdf') as pdf:
            print('write figure...')
            pdf.savefig(fig)'''


