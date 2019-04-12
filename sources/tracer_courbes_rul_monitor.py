# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:51:49 2018

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
import math

# se mettre sur le dossier de travail : pour le lancer sur Spyder , sinon le commenter
#os.chdir('C:/Users/abdel_000/Desktop/2018 Projet Long N7/hymu')
file = 'water_tanks/Results scenario_1/2019-03-29_12-07-07/2019-03-29_12-07-07.csv.gz'


##afficher les  estimations :

    

def draw_rul(fileGZ, figName, toshow = True, loc='best', eol = None, eom=None, minimum=False, showEop=False):

    data_moni = pandas.read_csv(fileGZ,compression='gzip', header=0, sep='|', quotechar=';', error_bad_lines=False)

    #for j in (data_moni.iloc[:,3]) :
        #print(j + '\n')    
    data = data_moni

    # color : (times, ruls, weights)
    res = {'k': ([], [], []), 'c' : ([], [], [])}
    t_min_ruls = []
    min_ruls = []
    for t, fms, bs in zip(data.t, data.fms, data.b):
        t -= data.t[0]
        if fms is not None and (not isinstance(fms, float) or not math.isnan(fms)):
            fms = fms.split(';')
            bs = bs.split(';')
            fms2 = []
            bs2 = []
            for f,b in zip(fms, bs):
                if f != 'no':
                    fms2.append(f)
                    bs2.append(b)
            fms = fms2
            bs = bs2
            #print(t,fms)
            fms = [[f for f in fm.split('][')] for fm in fms]
            eols = []
            bs2 = []
            for f,b in zip(fms,bs):
                e = []
                for fm in f:
                    if fm in ['[]', '', '[', ']']:
                        # already in last mode (failure or not, hope failure)
                        e.append(0)
                    else:
                        e.append(fm.split(',')[-1].strip(']) '))
                eols += e
                bs2 += [b] * len(e)

            ruls = [max(float(eol)-data.t[0]-t, 0) for eol in eols]
            ruls = ruls
            bs = [float(b) for b in bs2]
            # sort by belief
            mix = [(b,r) for b,r in zip(bs, ruls)]
            mix.sort(key=lambda x:x[0])
            ruls = []
            bs = []
            for b, r in mix:
                ruls.append(r)
                bs.append(b)

            if len(ruls) > 0:
                maxi = max(bs)
                for b, rul in zip(bs,ruls):
                    color = 'k' 
                    if b == maxi: color = 'c'
                    res[color][0].append(t)
                    res[color][1].append(rul)
                    res[color][2].append(b/maxi)
                t_min_ruls.append(t)
                min_ruls.append(min(ruls))

    fig = figure(fileGZ, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    set_cmap('bone_r') # 0 -> white, 1 -> black
    # add a fake belief to scale colormap
    scatter([0]+res['k'][0]+res['c'][0], [0]+res['k'][1]+res['c'][1], c=[-.1]+res['k'][2]+res['c'][2], s=3, marker='s', edgecolors='face')
    #scatter(res[c][0], res[c][1], c=[1]*len(res[c][2]), s=1, marker='s', edgecolors='face')

    if showEop:
        x = []
        y = []
        for t, k in zip(data.t, data.lpt):
            if not math.isnan(k):
                x.append(t - data.t[0])
                y.append(k - t)
        plot(x,y, 'g--', label = r'$k_{EOP}$')

    if eol is not None:
        plot([0, eol], [eol,0], 'b--', label = 'real EOL', lw=2)
    if eom is not None:
        plot([0, eom], [eom,0], 'b--', label = 'real EOM', lw=2)
    if minimum:
        plot(t_min_ruls, min_ruls, 'k.', label = 'min. RUL', lw=2)

    scatter([0], [0], color='#d5e4e4', s=3, marker='s', edgecolors='face', label='lowest beliefs')
    scatter([0], [0], color='k', s=3, marker='s', edgecolors='face', label='highest beliefs')
    legend(loc=loc, prop={'size':20})

    grid(True)
    #title("Remaining useful life estimation")
    xlabel('time (s)')
    ylabel('RULs (s)')
    rc('font', size=20) # thesis
    #rc('font', size=30) # article
    #left replaces xmin as xmin will be removed from matplotlib 3.2
    xlim(left=0)
    #bottom replaces ymin as ymin will be removed from matplotlib 3.2
    ylim(bottom=0)
    #ylim(ymin=0)
    fig.tight_layout()

    for ax in fig.axes: ax.set_axisbelow(True)
    
    plt.savefig(figName)
    if toshow:
        plt.ion()        
        plt.show()        
        #show()
        #with PdfPages('mode_rul.pdf') as pdf:
        #pdf.savefig(fig)


