# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:04:20 2018

@author: AA_OJ_SM
"""


import os
import pandas
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot 
from pylab import *
import gzip
import itertools as it
from matplotlib.legend_handler import HandlerLine2D
import math
from matplotlib import cm



# se mettre sur le dossier de travail : pour le lancer sur Spyder , sinon le commenter
#os.chdir('C:/Users/abdel_000/Desktop/2018 Projet Long N7/hymu')
file = 'water_tanks/Results scenario_1/2019-03-29_12-07-07/2019-03-29_12-07-07.csv.gz'



def get_possibilities(data, time = None):

    # reconstruct possibilities
    posss = {}
    posssId = data['id']
    posssFid = data['fid']
    timeIter = range(len(posssId))
    if time is not None:
        timeIter = range(time + 1)
    for t in timeIter:
        posssId_at_t = posssId[t].split(';')
        posssId_at_t = list(map(int, posssId_at_t))
        posssFid_at_t = posssFid[t].split(';')
        posssFid_at_t = list(map(int, posssFid_at_t))
        for i, (id, fid) in enumerate(zip(posssId_at_t, posssFid_at_t)):
            if id not in posss:
                if fid == '-1' or t == 0:
                    posss[id] = [None] * len(posssId)
                else:
                    posss[id] = posss[fid].copy()
            posss[id][t] = i

    # check if all possibilities are different
    #l = [str(modes_from_data(p, data)) for p in posss.values()]
    #(len(l), len(set(l)))
    #for i in set(l): print(i, l.count(i))
    #assert(len(posss) == len(set(str(modes_from_data(p, data)) for p in posss.values())))
    return posss



def modes_from_data(p, data):
    try:
        ti = p.index(None)
    except:
        ti = len(data['t'])
    
    ms = []
    for i, t, m in zip(range(ti), data.t, data.m):
        if ms:
            mode = m.split(';')[p[i]]
            if ms[-1][0] != mode:
                # new mode
                ms.append((mode, t))
        else:
            # ms is empty
            mode = m.split(';')[p[i]]
            ms.append((mode, t))
    #ms = data['ms'][ti].split(';')[p[ti]]
    return ms

def modes_from_data_relative_t(p, data):
    try:
        ti = p.index(None)
    except:
        ti = len(data['t'])
    
    ms = []
    for i, t, m in zip(range(ti), data.t, data.m):
        if ms:
            mode = m.split(';')[p[i]]
            if ms[-1][0] != mode:
                # new mode
                ms.append((mode, t-data.t[0]))
        else:
            # ms is empty
            mode = m.split(';')[p[i]]
            ms.append((mode, t-data.t[0]))
    #ms = data['ms'][ti].split(';')[p[ti]]
    return ms


def draw_mode_trajectories(fileGZ,figName, k = None, selectedModes = [], toshow = True, loc='best', future = True):
    data_moni = pandas.read_csv(fileGZ, compression='gzip', header=0, sep='|', quotechar=';', error_bad_lines=False)

    ##afficher les  estimations :
    #for j in (data_moni.iloc[:,3]) :
        #print(j + '\n')        
    data = data_moni

    # correct time
    datat = []
    for t in data.t:
        datat.append(t - data.t[0])

    if k is None:
        k = len(datat)-1

    print('help to choose k: k in [{}, {}]'.format(0, len(datat)-1))
    try :
        d = data.fms # ugly check if there is a column 'fms'
        futureK = {k for k, fms in enumerate(data.fms) if (fms is not None and fms != 'no' and (not isinstance(fms,float) or not math.isnan(fms)))}
        first = max([i for i in futureK if i<k], default=None)
        second = min([i for i in futureK if i>k], default=None)
        print('help to choose future k closed to {}: {}'.format(k, (first, second)))
    except Exception as e:
        futureK = set()
        print(e)
        print('Warning: no prediction in these data')

    posss = get_possibilities(data)

    modes = set(m for ms in data.m for m in ms.split(';'))
    modes = sorted(modes)
    print('help to choose mode: mode in {}'.format(modes))

    if selectedModes == []:
        selectedModes = modes

    # remove double
    posss2 = {}
    for p, pids in posss.items():
        # remove ended before k
        if pids[k] is not None:
            # remove belief == 0
            if float(data.b[k].split(';')[pids[k]])>0:
                # remove not in mode
                if data.m[k].split(';')[pids[k]] in selectedModes:
                    # remove double:
                    double = False
                    for pp in posss2:
                        if posss[pp][0:k+1] == posss[p][0:k+1]:
                            double = True
                    if not double:
                        posss2[p] = pids

    print('remain {} possibilities'.format(len(posss2)))

    # keep all poss that have id in history
    possToKeep = set(str(i) for ids in posss2.values() for i in ids[0:k+1])
    print('{} more possibilities to keep'.format(len(possToKeep)))

    # possibility to work with
    posss = {p:pids for p, pids in posss.items() if p in possToKeep or p in posss2}
    #posss = {p:pids for p, pids in posss.items() if p in posss2}

    print('{} possibilities to work with'.format(len(posss)))

    def modes_it_1(ms):
        return (m[0] for m in ms)

    def modes_it_2(ms):
        ms = ms.split('][')
        ms = [m.strip(' [] ') for m in ms]
        ms = [mm for m in ms for mm in m.split('), ')]
        ms = (m.strip(' ()').split(', ')[0] for m in ms)
        return ms
    modes = set(m for p in posss.values() for m in modes_it_1(modes_from_data(p, data)))
    if future and k in futureK:
        # future mode sequ
        fmss = data.fms[k]
        if fmss != 'no':
            fmss = fmss.split(';')
            modes |= set(m for p, pids in posss.items() if pids[k] is not None for m in modes_it_2(fmss[pids[k]]))
            modes -= {'', 'no'}
    modes = sorted(modes)
    #print(len(modes), 'modes', modes)

    # map categories to y-values
    categories = modes
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))
    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i) for i in categories]))

    print('got {} possibilities'.format(len(posss)))
    poss_modes = {p: [] for p in posss}
    poss_weights = {p: [] for p in posss}
    for i in range(len(datat)):
        bs = data.b[i].split(';')
        ms = data.m[i].split(';')
        for p, pids in posss.items():
            if pids[i] is not None:
                b = float(bs[pids[i]])
                m = cat_dict[ms[pids[i]]]
            else:
                b = None
                m = None
            poss_modes[p].append(m)
            poss_weights[p].append(b)
    
    nbColor = len(modes) 
    colors = cm.jet(np.arange(nbColor)/nbColor)
    modeColors = {m:c for m, c in zip(modes, colors)}
    #modeColors['Sensor BL FL fault'] = 'b'
    #modeColors['Sensor BL FL fault + Parasitic load'] = 'r'
    #fig = figure(2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    #fig = figure(2, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    fig = figure(2, figsize=(12, max(3,round(len(modes)/1.5))), dpi=80, facecolor='w', edgecolor='k')

    maxWeigth = max(poss_weights[p][k] for p in posss)
    count = 0
    for p in posss:
        lw = poss_weights[p][k]/maxWeigth
        c = modeColors[data.m[k].split(';')[posss2[p][k]]]
        alpha = .6
        if lw == 1:
            c='k'
            #alpha=.8
        #print(p, round(lw,2), modes_from_data_relative_t(posss[p], data))
        plot(datat[0:k+1], poss_modes[p][0:k+1], '-', linewidth = lw*4, alpha=alpha, color=c)
        if future and k in futureK:
            pidk = posss[p][k]
            if pidk is not None:
                fms = fmss[pidk]
                #print(fms)
                if fms !='no': # prediction has been made
                    fms = fms.split('][')
                    #for f in fms: print('  ', f, pidk)
                    for f in fms:
                        if f in ['[]', '', '[', ']']:
                            # already in last mode (failure or not, hope failure)
                            ms = []
                        else:
                            ms = f.strip(' [] ').split('), ')
                            ms = [m.strip(' () ') for m in ms]
                            ms = [tuple(m.split(', ')) for m in ms]
                        fx = [datat[k]]
                        fm = [poss_modes[p][k]]
                        #print(f)
                        for f1, f2 in ms:
                            t = float(f2) - data.t[0]
                            if t > fx[0]:
                                fx.append(t)
                                fx.append(fx[-1])
                                fm.append(fm[-1])
                                fm.append(cat_dict[f1])
                        # add last point
                        fx.append(data.lpt[k] - data.t[0])
                        fm.append(fm[-1])
                        #print('    ', list(zip([val_dict[f] for f in fm],fx)))
                        plot(fx, fm, '--', lw = lw*3, alpha=.7, color=c)
                        count += 1

    #print(count)

    if future:
        plot([], [], 'k', alpha=.7, lw=3,label='until {}s'.format(round(datat[k],3)))
        plot([], [], 'k--', alpha=.7, lw=3, label='after {}s'.format(round(datat[k],3)))
        plot([datat[k]]*2, [min(val_dict)-0.3, max(val_dict) + .3], 'k:', alpha=.5, lw=2, label='{}s'.format(round(datat[k],3)))
        plot([datat[k]], [min(val_dict)-0.3], 'k^', alpha=.5, lw=1)
        plot([datat[k]], [max(val_dict)+0.3], 'kv', alpha=.5, lw=1)
        legend(loc=loc, prop={'size':25})
    else:
        plot([], [], 'k', alpha=.7, lw=1,label='lowest beliefs')
        plot([], [], 'k', alpha=.7, lw=5,label='highest beliefs')
        legend(loc=loc, prop={'size':20})
      

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
    ylim(min(val_dict)-0.4, max(val_dict) + .4)
    #ylim(6.8, 8.2)
    #lim(4890, 5089)
    #xlim(570, 760)
    #lim(0, 8200)
    rc('font', size=20)
    fig.tight_layout()
    plt.savefig(figName)
    if toshow:
        plt.ion()        
        plt.show()

        '''show()
        with PdfPages('mode_belief.pdf') as pdf:
            print('write figure...')
            pdf.savefig(fig)'''



