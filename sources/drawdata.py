from pylab import *
import pandas
import numpy
from scenario import Scenario
from monitor import RunnerData
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.backends.backend_pdf import PdfPages
import traceback
import random
import gzip
import math
from equation import softmax
import itertools as it
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

rc('text', usetex=True)
rc('font', family='serif')
ion() # enables interactive mode
#COLORS = ['b', 'c', 'r', 'y', 'g', 'm', 'k'] * 10000

#def draw(filename):
#    draw_health_mode(filename, False)
#    #draw_diagnosis(filename, 1, False)
#    #draw_health_mode_belief(filename, None, False)
#    draw_state_monitoring(filename, False)
#    #draw_degradation_monitoring_3D(filename, False)
#    #draw_degradation_monitoring(filename, False)
#    #draw_degradation_mode(filename, False)
#    draw_health_mode_belief_3D(filename, False)
#
#    show()

def draw_health_mode(filenameScenario, toshow = True, reverse = False):

    scenario = Scenario(filenameScenario)
    
    fig = figure(1, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

    categories = sorted([m for m in set(scenario.m)], key=lambda x:x[1])
    print(categories)
    # map categories to y-values
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i).replace(',',',\n') for i in categories]))

    plotval = [cat_dict[m] for m in scenario.m]
    if reverse :
        plot(plotval, scenario.t, 'k-', lw=4)
    else:
        plot(scenario.t, plotval, 'k-', lw=4)
    
    if reverse :
        gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
        plt.gca().invert_yaxis()
    else :
        gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    grid(True)
    title("Real health mode")
    if reverse :
        ylabel('time (s)')
        xlabel('health mode')
        xlim(.75, len(categories) +0.25)
    else:
        xlabel('time (s)')
        ylabel('health mode')
        ylim(.75, len(categories) +0.25)
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

    if toshow:
        show()

def draw_rul(filename, toshow = True, loc='best', eol = None, eom=None, minimum=False, showEop=False):

    data = read_data(filename)

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

    fig = figure(2, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
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
    xlim(xmin=0)
    ylim(ymin=0)
    #ylim(ymin=0)
    fig.tight_layout()

    for ax in fig.axes: ax.set_axisbelow(True)
    
    if toshow:
        show()
    else:
        with PdfPages('mode_rul.pdf') as pdf:
            pdf.savefig(fig)

def draw_mode_belief_old(filename, minimumB = 0, toshow = True, loc='best', queue_sizes = 0, selectedModes = None):

    data = read_data(filename)

    # correct time
    datat = []
    for t in data.t:
        datat.append(t - data.t[0])

    posss = get_possibilities(data)
    print('got {} possibilities'.format(len(posss)))

    def filter(p):
        #return score_from_data(p,data) > minimumB
        #nb = 120
        try:
            ind = p.index(None)
            ts = [ind - i for i in range(1,queue_sizes+1)]
        except:
            ind = len(p)
            ts = [ind - i for i in range(1,queue_sizes+1)]
        if any([t<0 for t in ts]):
            # tiny possibility
            return False
        else:
            ms = [data['m'][t].split(';')[p[t]] for t in ts]
            if len(set(ms)) > 1:
                # look if the tail (n-1) elts is important enough
                lm = ms[-1]
                bs = [float(data['b'][t].split(';')[p[t]]) for t,m in zip(ts,ms) if m != lm]
                return max(bs) >= minimumB
            else:
                return True

    posss = {p:l for p,l in posss.items() if filter(l)}

    print('remain {} possibilities after filter'.format(len(posss)))

    def modes_it(ms):
        return (m[0] for m in ms)

    # optimized way to find all necessary modes
    modes = set() 
    for i, (t, m) in enumerate(zip(data.t, data.m)):
        ms = m.split(';')
        for p,pids in posss.items():
            if pids[i] is not None:
                modes.add(ms[pids[i]])

    #modes = set(m for p in posss.values() for m in modes_it(modes_from_data(p, data)))
    #modes = {'Sensor BL FL fault + Parasitic load', 'Sensor BL FL fault', 'Sensor BL FL FR fault + Parasitic load',  'Sensor BL FL FR fault','Sensor BL BR FL fault + Parasitic load','Sensor BL BR FL fault','Sensor BL BR FL FR fault + Parasitic load', 'Sensor BL BR FL FR fault'}
    modes = sorted(modes)
    print(len(modes), 'modes')
    
    if selectedModes is not None:
        modes = selectedModes

    if len(modes) == 0:
        raise Exception('no mode (you may want to reduce the possibility queue minimum sizes)')

    # map categories to y-values
    categories = modes
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))
    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [str(i).replace(',',',\n') for i in categories]))

    print('remaining {} possibilities after filtering'.format(len(posss)))

    #poss_modes = {p: [] for p in posss}
    #poss_weights = {p: [] for p in posss}
    #for i, t in enumerate(datat):
    #    bs = data.b[i].split(';')
    #    ms = data.m[i].split(';')
    #    for p, pids in posss.items():
    #        if pids[i] is not None:
    #            b = float(bs[pids[i]])
    #            m = cat_dict[ms[pids[i]]]
    #        else:
    #            b = None
    #            m = None
    #        poss_modes[p].append(m)
    #        poss_weights[p].append(b)
    #
    #for p, ms in poss_modes.items():
    #    plot(data.t, ms, 'k')

    def get_max_modes(i):
        res = {}
        bs = [float(b) for b in data.b[i].split(';')]
        ms = [m for m in data.m[i].split(';')]
        for p, pids in posss.items():
            if pids[i] is not None:
                b = bs[pids[i]]
                mode_str = ms[pids[i]]
                if mode_str in modes:
                    m = cat_dict[ms[pids[i]]]
                    if res.get(m) is not None:
                        if res.get(m) < b:
                            res[m] = b
                    else:
                        res[m] = b
        return res

    fig = figure(2, figsize=(12, max(3,round(len(modes)/1.3))), dpi=80, facecolor='w', edgecolor='k')
    #fig = figure(2, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
    #fig = figure(2, figsize=(26, 12), dpi=80, facecolor='w', edgecolor='k')

    print('got that')

    # plot version
    #for i, t in enumerate(datat):
    #    #print(i)
    #    ms = get_max_modes(i) 
    #    #print(ms)
    #    if len(ms.values())>0:
    #        maxi = max(ms.values())
    #    for m, b in ms.items():
    #        if b != 0:
    #            if b < maxi:
    #                plot(t, m, 'k|', ms=2+18*b)
    #                #color = 'k'
    #            else:
    #                plot(t, m, 'c|', ms=2+18*b)
    #                #color = 'c'

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
                res[color][2].append(b/maxi*200)
    c = 'k'
    scatter(res[c][0], res[c][1], s=res[c][2], c=c, marker='|')
    c = 'c'
    scatter(res[c][0], res[c][1], s=res[c][2], c=c, marker='|')

    l1, = plot([], [], 'k-', lw=10, label='mode belief')
    l2, = plot([], [], 'c-', lw=10, label='higher belief')
    legend(loc=loc, handler_map={line: HandlerLine2D(numpoints=1) for line in [l1,l2]}, prop={'size':20})
    
    val_dict = {v: s.replace(' + ', '\n+ ') for v,s in val_dict.items()}
    val_dict = {v: s.replace('BR FL', 'BR\nFL') for v,s in val_dict.items()}
    val_dict = {v: s.replace('FL FR', 'FL\nFR') for v,s in val_dict.items()}
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    #if modes is None:
    #    interesting = [c for c in data if c.startswith('(') and any([i!=0 for i in data[c]])]
    #    markers = ['-' for c in interesting]
    #    markerssizes = [7 for c in interesting]
    #else:
    #    interesting = ['({})'.format(m) for m in modes]
    #    markers = ['go-', 'rx-', 'k+-', 'cd-', 'b*-']
    #    markerssizes = [9 for m in markers]

    #for c, m, ms  in zip(interesting, markers, markerssizes):
    #    plot(t, data[c], m, ms=ms,label=c)

    #if modes is None:
    #    infer = []
    #    for i in range(len(t)):
    #        infer.append(data[data['real behavioral mode'][i]][i])
    #    plot (t, infer, 'ok', label='real health mode')

    grid(True)
    #title("Belief degrees of modes")
    xlabel('time (s)')
    ylabel('mode')
    ylim(min(val_dict)-0.2, max(val_dict) + .2)
    rc('font', size=20)
    fig.tight_layout()
    gca().axes.get_yaxis().set_ticks(list(range(1, len(val_dict)+1))) # force y axis labels
    print('got this')

    for ax in fig.axes: ax.set_axisbelow(True)
    
    if toshow:
        show()
    else:
        with PdfPages('mode_belief.pdf') as pdf:
            print('write figure...')
            pdf.savefig(fig)

def draw_mode_belief(filename, minimumB = 0, toshow = True, loc='best', selectedModes = None, minimum = 1, blacklist = None):

    data = read_data(filename)

    # correct time
    datat = []
    for t in data.t:
        datat.append(t - data.t[0])

    modes = sorted([m for ms in data['m'] for m in ms.split(';')])
    modes = set(m for m, nb in it.groupby(modes) if len(list(nb)) > minimum)
    modes = sorted(modes)
    print(len(modes), 'modes')
    
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

    fig = figure(2, figsize=(12, max(3,round(len(modes)/1.3))), dpi=80, facecolor='w', edgecolor='k')
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
    
    if toshow:
        show()
    else:
        with PdfPages('mode_belief.pdf') as pdf:
            print('write figure...')
            pdf.savefig(fig)

def draw_scenario_trajectory(filename, toshow = True, loc='best'):

    data = read_data(filename)
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


def draw_mode_trajectories(filename, k = None, selectedModes = [], toshow = True, loc='best', future = True):

    data = read_data(filename)

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
    print(len(modes), 'modes', modes)

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
        print(p, round(lw,2), modes_from_data_relative_t(posss[p], data))
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
                        print(f)
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
                        print('    ', list(zip([val_dict[f] for f in fm],fx)))
                        plot(fx, fm, '--', lw = lw*3, alpha=.7, color=c)
                        count += 1

    print(count)

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
    
    if toshow:
        show()


def draw_health_mode_belief_3D(filenameScenario, filenameRunner, toshow = True):

    scenario = Scenario(filenameScenario)
    runner = RunnerData(filenameRunner)

    fig = figure(3, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    interesting = sorted(runner.m[0], key=lambda x:x[1])

    # map categories to y-values
    cat_dict = dict(zip(interesting, range(1, len(interesting)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(interesting)+1), ['({})'.format(',\n'.join(e for e in m)) for m in interesting]))

    plotval = [cat_dict[i] for i in interesting]

    X = numpy.array([runner.t for _ in interesting])
    Y = numpy.array([[plotval[i] for _ in runner.t] for i in range(len(plotval))])
    Z = numpy.array([[runner.m[i][m] for i in range(len(runner.t))] for m in interesting])

    ax.plot_surface(X, Y, Z, rstride=1, cstride=5, alpha=0.9, antialiased = True, cmap=cm.hot)

    real = [cat_dict[m] for m in scenario.m]
    plot (scenario.t, real, [-0.1 for _ in scenario.t], 'k-', label='real health mode', lw=3)

    best = []
    modes = []
    for i in range(len(runner.t)):
        maxi = max(runner.m[i][m] for m in interesting)
        modes = [cat_dict[m] for m in interesting if runner.m[i][m] == maxi]
        for m in modes : plot([runner.t[i]], [m], [maxi], 'ow')
    #plot([], [], [], 'ow',  label='prefered health mode')

    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    grid(True)
    title("Belief degrees of health modes")
    xlabel('time (s)')
    lt = scenario.t[-1]
    ax.set_xlim(-0.1 * lt, lt + 0.1 * lt)
    ylabel('\n\nhealth mode')
    ax.set_ylim(0, len(interesting) + 1)
    ax.set_zlabel('degré de croyance')
    ax.set_zlim(-0.1, 1.1)
    #legend(loc=4)
    rc('font', size=20)

    #def proj(i) :
    #    x = (t[i-1] + t[i])/2
    #    y = (cat_dict[data['real behavioral mode'][i-1]] + cat_dict[data['real behavioral mode'][i]])/2
    #    x,y,_ = proj3d.proj_transform(x, y, -0.1, ax.get_proj())
    #    return x,y

    #def update_position(e):
    #    for l in labels:
    #        x, y = proj(l)
    #        labels[l].xy = x,y
    #        labels[l].update_positions(fig.canvas.renderer)
    #    fig.canvas.draw()

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
    #        labels[i] = annotate(events[i], xy=(x,y), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    #fig.canvas.mpl_connect('button_release_event', update_position)
    #fig.canvas.mpl_connect('motion_notify_event', update_position)

    if toshow:
        show()

def draw_token_numbers(filenameScenario, filenameRunner, toshow = True):

    scenario = Scenario(filenameScenario)
    runner = RunnerData(filenameRunner)

    fig, axes = subplots(3, sharex=True, sharey=False, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k', squeeze = False)

    axes = [ax[0] for ax in axes]
    axes[0].scatter(runner.t,runner.cn)
    axes[0].set_ylabel('configurations')
    axes[0].set_ylim(min(runner.cn)-1,max(runner.cn)+1)
    axes[0].grid(True)
    axes[1].scatter(runner.t,runner.pn)
    axes[1].set_ylabel('particules')
    axes[1].set_ylim(min(runner.pn)-1,max(runner.pn)+1)
    axes[1].grid(True)
    axes[2].scatter(runner.t,runner.hn)
    axes[2].set_ylabel('hybrid tokens')
    axes[2].set_ylim(min(runner.hn)-1,max(runner.hn)+1)
    axes[2].set_xlim(min(runner.t)-1,max(runner.t)+1)
    axes[2].grid(True)

    axes[0].set_title("Number of tokens")
    axes[2].set_xlabel('time (s)')
    rc('font', size=18)

    if toshow:
        show()

def ploty(filenameRunner, selection, toShow = True):

    fig = figure(4, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

    print('{} data to plot'.format(len(selection)))

    if filenameRunner:
        data = read_data(filenameRunner)#, names = ['t'] + selection)
        for l in selection: 
            if l in ['ppon', 'pn']:
                d = [sum(list(map(int,d.split(';')))) for d in data[l]]
                plot(data.t, d, label=r'{}'.format(l))
            elif l == 'pcput':
                x = []
                y = []
                for t,d in zip(data.t, data[l]):
                    if not math.isnan(d):
                        x.append(t)
                        y.append(d)
                plot(x, y, label=r'{}'.format(l))
            else:
                plot(data.t, data[l], label=r'{}'.format(l))

    grid(True)
    #title("Monitoring of water levels")
    xlabel('time (s)')
    #ylabel('height (m)')
    legend()
    rc('font', size=20)
    if toShow:
        show()

def read_data(filename):
    if filename.endswith('.gz'):
      with gzip.open(filename) as gzipfile:
        #data = pandas.read_csv(gzipfile, sep='|', skiprows = [1], names = names) # ignore first line ('no' values)
        data = pandas.read_csv(gzipfile, sep='|')
    else:
        #data = pandas.read_csv(filename, sep='|', skiprows = [1], names = names) # ignore first line ('no' values)
        data = pandas.read_csv(filename, sep='|')
    print(filename, 'loaded')
    return data

def write_data(data, filename):
    data.to_csv(filename, sep='|', index = False)

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

def score_from_data(p, data, time = None): 
    try:
        if time is None:
            s = float(str(data['b'][len(data['t'])-1]).split(';')[p[-1]])
        else:
            s = float(str(data['b'][time]).split(';')[p[time]])
    except Exception:
        # unfinished possibility
        s = -1
    return s

def num_score_from_data(p, data, time = None): 
    try:
        if time is None:
            s = float(str(data['nb'][len(data['t'])-1]).split(';')[p[-1]])
        else:
            s = float(str(data['nb'][time]).split(';')[p[time]])
    except Exception:
        # unfinished possibility
        s = -1
    return s

def sym_score_from_data(p, data, time = None): 
    try:
        if time is None:
            s = float(str(data['sb'][len(data['t'])-1]).split(';')[p[-1]])
        else:
            s = float(str(data['sb'][time]).split(';')[p[time]])
    except Exception:
        # unfinished possibility
        s = -1
    return s

def part_nb_from_data(p, data, time = None): 
    try:
        if time is None:
            s = float(str(data['pn'][len(data['t'])-1]).split(';')[p[-1]])
        else:
            s = float(str(data['pn'][time]).split(';')[p[time]])
    except Exception:
        # unfinished possibility
        s = -1
    return s

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

def draw_stats(filename):
    fig = figure(4, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    d = pandas.read_csv(filename)
    d = d[(d.psp2==5) & (d.psp3==100)]
    print(d)

    colors = ['g','b','r']

    ax = fig.add_subplot(311)
    ax.plot([],[], '.', c=colors[0], label=r'$\rho^{min}_\Delta$')
    ax.plot([],[], '.', c=colors[1], label=r'$\rho^{max}_\Delta$')
    ax.plot([],[], '.', c=colors[2], label=r'$\rho^{tot}_\Delta$')
    legend(loc='best')
    ax.plot(d.dcput_ave, d.dsp1, '.', c=colors[0])
    ax.plot(d.dcput_ave, d.dsp2, '.', c=colors[1])
    ax.plot(d.dcput_ave, d.dsp3, '.', c=colors[2])
    ax.set_xlabel(r'$\Delta$ CPU time (s)')
    ax.set_ylabel(r'$\rho_\Delta$')

    grid(True)

    ax = fig.add_subplot(312)
    ax.plot(d.pcput_ave, d.dsp1, '.', c=colors[0])
    ax.plot(d.pcput_ave, d.dsp2, '.', c=colors[1])
    ax.plot(d.pcput_ave, d.dsp3, '.', c=colors[2])
    ax.set_xlabel(r'$\Pi$ CPU time (s)')
    ax.set_ylabel(r'$\rho_\Delta$')
    grid(True)

    ax = fig.add_subplot(313)
    ax.plot(d.maxmem, d.dsp1, '.', c=colors[0])
    ax.plot(d.maxmem, d.dsp2, '.', c=colors[1])
    ax.plot(d.maxmem, d.dsp3, '.', c=colors[2])
    ax.set_xlabel(r'maximum RAM (MB)')
    ax.set_ylabel(r'$\rho_\Delta$')

    grid(True)
    rc('font', size=20) # 20 for paper
    fig.tight_layout()

def draw(filenameScenario, filenameRunner, selection, toshow = True, time = None, backToTheFuture = True, extra = None, loc='best'):

    #fig = figure(4, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = figure(4, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

    print('{} data to plot'.format(len(selection)))

    if filenameScenario:
        data = read_data(filenameScenario)
        # correct time
        datat = []
        for t in data.t:
            datat.append(t - data.t[0])

        marks = ['b-', 'g-', 'r-']
        for l, mk in zip(selection, marks):
            try:
                labels = l.split('_')
                label = '$\\textup{' + labels[0] + '}_{' + labels[1] + '}$'
                datal = [float(d) if d != 'no' else None for d in data[l]]
                label = '$y_1$' #'${}$'.format(l)
                ylabel('voltage (V)')
                #plot(datat, datal, mk,label=r'{}'.format(label), alpha=.5, lw = 1.5)
                plot(datat, datal, mk, alpha=.5, lw = 1.5)
                #plot(datat, data[l])
            except:
                print('fail to print {} of {}'.format(l, filenameScenario))

    if filenameRunner:
        psep=';'
        data = read_data(filenameRunner)
        # correct time
        datat = []
        for t in data.t:
            datat.append(t - data.t[0])
        try:
            timePoss = None
            if time is not None and backToTheFuture:
                timePoss = time
            posss = get_possibilities(data, timePoss)

            if time is None:
                time = len(datat)-1

            def score(p):
                return score_from_data(posss[p], data, time)
            def num_score(p):
                return num_score_from_data(posss[p], data, time)
            def sym_score(p):
                return sym_score_from_data(posss[p], data, time)
            def part_nb(p):
                return part_nb_from_data(posss[p], data, time)
            sort = sorted(posss.keys(), key=score, reverse = True)

            ssum = sum(1 for p in sort if score(p) > 0)
            print('{} possibilities with belief > 0 (total: {}) at {}s'.format(ssum, len(sort), datat[time]))

            for l in selection: 
                if '_' in l:
                    smean = next((s for s in data if s.startswith(l) and s.endswith('mean')))
                    smin = next((s for s in data if s.startswith(l) and s.endswith('min')))
                    smax = next((s for s in data if s.startswith(l) and s.endswith('max')))

                # remove useless possibility
                #def filt(p,l):
                #    accepted = l[-1] != None and float(str(data['b'][len(data['t'])-1]).split(';')[l[-1]]) >= minimumB
                #    return accepted
                    #ll = [ll for ll in l if ll is not None]
                    #N = len(l)-1
                    #if len(ll) > N:
                    #    modes = [data['ms'][len(ll)-i].split(';')[ll[-i]] for i in range(1,N+1)]
                    #    return len(set(modes)) == 1

                #posss = {p: posss[p] for p in posss if filt(p,posss[p])}

                #for p in posss:
                #    print('{}, {}, {}'.format(p, sum(1 for pi in posss[p] if pi is not None), posss[p][0]))
                #    print(posss[p])

                #input(len(posss))
                nbColor = 7
                colors = cm.jet(np.arange(nbColor)/nbColor)
                print('total score sym:', sum(sym_score(p) for p in sort if score(p)>0))
                for p in sort: 
                    if score(p)>0: print(sym_score(p))

                sort = sort[:6]
                if 1 not in sort:
                    # add first possibility
                    #sort.append(1)
                    pass
                for i, p in enumerate(sort):
                    pid = posss[p]
                    ms = modes_from_data(pid, data)
                    try: j = next(j-1 for j,p in enumerate(pid) if p==None)
                    except: j = len(pid)-1
                    ms = [(m[0], round(m[1] - data.t[0], 2)) for m in ms]
                    try:
                        # special rover
                        print('{} ({}): ({},{},{}:{}): {}'.format(i, p,round(score(p), 2),round(sym_score(p), 2), round(num_score(p), 2), part_nb(p), ms), round(float(str(data['x_30_mean'][j]).split(';')[pid[j]]),2))
                    except:
                        print('{} ({}): ({},{},{}:{}): {} : {} mode changes'.format(i,p,round(score(p), 2),round(sym_score(p), 2), round(num_score(p), 2), part_nb(p), ms, len(ms)-1))
                    #print(pid)

                    if '_' in l:
                        dmean = []
                        dmin = []
                        dmax = []
                        datat2 = []
                        for t in range(len(data['t'])):
                            if pid[t] is not None:
                                dd = str(data[smean][t]).split(';')[pid[t]]
                                if dd != 'no':
                                    dmean.append(float(dd))
                                    dd = str(data[smin][t]).split(';')[pid[t]]
                                    dmin.append(float(dd))
                                    dd = str(data[smax][t]).split(';')[pid[t]]
                                    dmax.append(float(dd))
                                    datat2.append(datat[t])
                        labels = smean.split('_')
                        #label = '$p_{' + str(i) +'} \\textup{' + labels[0] + '}_{' + labels[1] + labels[2] + '}$'
                        label = '$h_{' + str(i) +'}$'
                        label += ' ({})'.format(round(score(p), 3))
                        if p == 1:
                            dash = '-.'
                            lw = 4
                            c = colors[-1]
                        else:
                            dash = '--'
                            lw = 1 + score(p)
                            c = colors[i]
                        plot(datat2, dmean, dash, linewidth=lw, color=c, alpha=1, label=r'{}'.format(label))
                        dmin = [d if d is not None else 0 for d in dmin]
                        dmax = [d if d is not None else 0 for d in dmax]
                        fill_between(datat2, dmin, dmax, color=c, alpha=0.2, interpolate=True)
                    else:
                        # other values, such pn, ppon
                        label = '$p_{' + str(i) +'}$'
                        label += ' ({})'.format(round(score(p), 3))
                        d = []
                        for t in range(len(data['t'])):
                            d.append(str(data[l][t]).split(';')[pid[t]] if pid[t] is not None else None)
                            d = [float(dd) if dd is not None else None for dd in d]
                        plot(datat, d, '--', linewidth=1+score(p), color=colors[i], alpha=1, label=r'{}'.format(label))

        except Exception as e:
            #raise(e)
            tb = traceback.format_exc()
            for l in selection:
                try:
                    labels = l.split('_')
                    label = '$\\textup{' + labels[0] + '}_{' + labels[1] + '}$'
                    label = 'model'
                    datat2 = []
                    datal = []
                    for i in range(len(datat)):
                        if data[l][i] is not None and data[l][i] != 'no':
                            datal.append(float(data[l][i]))
                            datat2.append(datat[i])

                    #datat2 = [datat2[i] for i in range(0, len(datat2), 30)]
                    #datal = [datal[i] for i in range(0, len(datal), 30)]
                    plot(datat2, datal, '-', label=r'{}'.format(label))
                except Exception as e2:
                    tb2 = traceback.format_exc()
                    print('Exception 1: ' + tb)
                    print('Exception 2: ' + tb2)
                    print('fail to print {} of {}'.format(l, filenameRunner))

        if extra is not None:
            data = read_data(extra)
            for l in selection:
                labels = l.split('_')
                label = '$\\textup{' + labels[0] + '}_{' + labels[1] + '}$'
                label = 'model 2.5 A'
                datat2 = []
                datal = []
                for i in range(len(datat)):
                    if data[l][i] is not None and data[l][i] != 'no':
                        datal.append(float(data[l][i]))
                        datat2.append(datat[i])

                datat2 = [datat2[i] for i in range(0, len(datat2), 30)]
                datal = [datal[i] for i in range(0, len(datal), 30)]
                plot(datat2, datal, '-.', lw=3, label=r'{}'.format(label))

    #columns = ['h1 min correction', 'h2 min correction', 'h3 min correction']
    #markers = ['r--', 'b--', 'g--']
    #for c, m in zip(columns, markers):
    #    plot(t, data[c], m, label=c)

    #columns = ['h1 max correction', 'h2 max correction', 'h3 max correction']
    #markers = ['r--', 'b--', 'g--']
    #for c, m in zip(columns, markers):
    #    plot(t, data[c], m, label=c)
    
    #if filenameRunner:
    #    x = runner.xmin
    #    columns = []
    #    for i in range(len(x[0])):
    #        columns.append([x[j][i] for j in range(len(x))])
    #    x = runner.xmax
    #    columns2 = []
    #    for i in range(len(x[0])):
    #        columns2.append([x[j][i] for j in range(len(x))])
    #    markers = ['r', 'b', 'g']

    #    for c1,c2,m in zip(columns, columns2, markers):
    #        fill_between(runner.t, c1, c2, facecolor=m, alpha=0.2, interpolate=True)

    grid(True)
    #title("Rover battery voltage")
    xlabel('time (s)')
    xlim(datat[0], datat[len(datat)-1])
    #ylabel('voltage (V)')
    legend(loc=loc)
    #rc('font', size=12) # 20 for paper
    rc('font', size=20) # 20 for paper
    fig.tight_layout()

    if toshow:
        show()

def draw_degradation_mode(filename, toshow = True):

    data = pandas.read_csv(filename, sep=';')
    t = data['time']

    figure(5, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

    c = 'real hybrid mode'
    categories = sorted({m for m in data[c]})
    categories = sorted([c for c in categories], key=lambda x:x.split(',')[1])
    # map categories to y-values
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [i.replace(',',',\n') for i in categories]))

    plotval = data[c].apply(cat_dict.get)
    plot(t, plotval, 'k-', lw=4)
    
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    grid(True)
    title("Real degradation mode")
    xlabel('time (s)')
    ylabel('degradation mode')
    ylim(.75, len(categories) +0.25)
    rc('font', size=20)

    def proj(i) :
        x = (t[i-1] + t[i])/2
        y = (cat_dict[data['real hybrid mode'][i-1]] + cat_dict[data['real hybrid mode'][i]])/2
        return x,y

    events = data['events']
    diagevents = data['observed events']
    labels = {}
    for i in range(len(t)):
        if events[i] != 'no event':
            x,y = proj(i)
            style = 'sawtooth,pad=0.3'
            fc = 'red'
            if diagevents[i] == events[i]:
                style = 'round4,pad=0.3'
                fc = 'white'
            labels[i] = annotate(events[i], xy=(x,y), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    if toshow:
        show()

def draw_degradation_monitoring_3D(filenameScenario, filenameRunner, i, toshow = True):

    scenario = Scenario(filenameScenario)
    if filenameRunner:
        runner = RunnerData(filenameRunner)
    
    fig = figure(6, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    interesting = sorted(runner.h[0], key=lambda x:x[1])

    # map categories to y-values
    cat_dict = dict(zip(interesting, range(1, len(interesting)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(interesting)+1), ['({})'.format(',\n'.join(e for e in m)) for m in interesting]))

    plotval = [cat_dict[i] for i in interesting]

    def func(m, i):
        return [float(h[m][i]) if h[m] != None else 0 for h in runner.h]

    faults = range(3)
    if i != None:
        faults = [i]
    for i in faults:
        X = numpy.array([runner.t for _ in interesting])
        Y = numpy.array([[plotval[i] for _ in runner.t] for i in range(len(plotval))])
        Z = numpy.array([func(c, i) for c in interesting])

        ax.plot_surface(X, Y, Z, rstride=1, cstride=5, alpha=0.6, antialiased = True, color=numpy.random.rand(3,1), label = 'f{}'.format(i+1))

    real = [cat_dict[m] for m in scenario.m]
    plot (runner.t, real, [-0.1 for _ in runner.t], 'k-', label='real mode', lw=3)

    #best = []
    #modes = []
    #for i in range(len(t)):
    #    maxi = max([data[m][i] for m in interesting])
    #    modes = [cat_dict[m] for m in interesting if data[m][i] == maxi]
    #    for m in modes : plot([t[i]], [m], [maxi], 'ow')
    #plot([], [], [], 'ow',  label='prefered health mode')

    gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    grid(True)
    if i != None:
        title("Monitoring of fault f{}".format(i+1))
    else:
        title("Monitoring of faults probabilities")
    xlabel('time (s)')
    ylabel('\n\n\nprobability')
    lt = runner.t[-1]
    ax.set_xlim(-0.1 * lt, lt + 0.1 * lt)
    ax.set_ylim(0, len(interesting) + 1)
    ax.set_zlabel('degré de croyance')
    ax.set_zlim(-0.1, 1.1)
    #legend(loc=4)
    rc('font', size=20)

    #def proj(i) :
    #    x = (t[i-1] + t[i])/2
    #    y = (cat_dict[data['real hybrid mode'][i-1]] + cat_dict[data['real hybrid mode'][i]])/2
    #    x,y,_ = proj3d.proj_transform(x, y, -0.1, ax.get_proj())
    #    return x,y

    #def update_position(e):
    #    for l in labels:
    #        x, y = proj(l)
    #        labels[l].xy = x,y
    #        labels[l].update_positions(fig.canvas.renderer)
    #    fig.canvas.draw()

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
    #        labels[i] = annotate(events[i], xy=(x,y), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    #fig.canvas.mpl_connect('button_release_event', update_position)
    ##fig.canvas.mpl_connect('motion_notify_event', update_position)

    if toshow:
        show()


def draw_degradation_monitoring(filename, toshow = True):

    data = pandas.read_csv(filename, sep=';')
    t = data['time']

    figure(7, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')


    columns = [c for c in data if c.startswith('h') and c.endswith('real')]
    columns = [c for c in columns if c.startswith('h0') or c.startswith('h1') or c.startswith('h2')]
    i = 1
    if toshow:
        markers = ['ro-', 'b+-', 'gx-']
        markerssizes = [9 for m in markers]
    else:
        markers = ['ro-', 'bo-', 'go-']
        markerssizes = [7 for m in markers]
    for c, m, ms in zip(columns, markers, markerssizes):
        plot(t, data[c], m, ms=ms, label='real f{}'.format(i))
        i += 1

    def func(tab):
        return [float(tab[i]) if tab[i] != 'no degradation' else 0 for i in range(len(tab))]


    columns = [c for c in data if c.startswith('h') and not c.endswith('real')]
    hybrids = set(c.split(' ')[0] for c in columns)
    modes = set(c.split(' ')[1] for c in columns)
    columns = [c for c in columns if c.startswith('h0') or c.startswith('h1') or c.startswith('h2')]
    i = 1
    if toshow:
        markers = ['ro--', 'b+--', 'gx--']
        markerssizes = [9 for m in markers]
    else:
        markers = ['ro--', 'bo--', 'go--']
        markerssizes = [7 for m in markers]
    for c, m, ms in zip(columns, markers, markerssizes):
        plot(t, func(data[c]), m, ms=ms, alpha=0.5, label='f{}'.format(i))
        i += 1

    #columns = ['h1 min correction', 'h2 min correction', 'h3 min correction']
    #markers = ['r--', 'b--', 'g--']
    #for c, m in zip(columns, markers):
    #    plot(t, data[c], m, label=c)

    #columns = ['h1 max correction', 'h2 max correction', 'h3 max correction']
    #markers = ['r--', 'b--', 'g--']
    #for c, m in zip(columns, markers):
    #    plot(t, data[c], m, label=c)
    
    columns = [c for c in data if c.startswith('h') and c.endswith('min')]
    markers = ['r', 'b', 'g']
    columns2 = [c for c in data if c.startswith('h') and c.endswith('max')]

    for c1,c2,m in zip(columns, columns2, markers):
        fill_between(t,func(data[c1]), func(data[c2]), facecolor=m, alpha=0.2, interpolate=True)

    grid(True)
    title("Monitoring of faults probabilities")
    xlabel('time (s)')
    ylabel('probability')
    rc('font', size=20)
    legend()

    if toshow:
        show()

def draw_diagnosis(filename, loc='best', toshow = True, reverse = False):

    data = pandas.read_csv(filename, sep=';')
    t = data['time']
    
    fig = figure(8, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

    c = 'real behavioral mode'
    #categories = sorted({m for m in data[c]})
    #categories = sorted([c for c in categories], key=lambda x:x.split(',')[1])
    categories = sorted([c for c in data if c.startswith('(')], key=lambda x:x.split(',')[1])
    # map categories to y-values
    cat_dict = dict(zip(categories, range(1, len(categories)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(categories)+1), [i.replace(',',',\n') for i in categories]))

    plotval = data[c].apply(cat_dict.get)
    if reverse :
        plot(plotval, t, 'k-', lw=6)
    else:
        plot(t, plotval, 'k-', lw=6)
    l1, = plot([], [], 'k-', lw=6, label='real health mode')
    
    best = []
    modes = []
    for i in range(len(t)):
        modes = [cat_dict[m] for m in categories if data[m][i] > 0]
        if reverse :
            for m in modes : plot([m], [t[i]], 'oc', ms = 9)
        else :
            for m in modes : plot([t[i]], [m], 'oc', ms = 9)
        maxi = max(data[m][i] for m in categories)
        modes = [cat_dict[m] for m in categories if data[m][i] == maxi]
        if reverse :
            for m in modes : plot([m], [t[i]], 'ow', ms = 6)
        else:
            for m in modes : plot([t[i]], [m], 'ow', ms = 6)
    l2, = plot([], [], 'oc', ms=9,  label='possible health mode')
    l3, = plot([], [], 'ow', ms=6, label='prefered health mode')

    if reverse :
        gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
        plt.gca().invert_yaxis()
    else :
        gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))
    grid(True)
    title("Health mode diagnosis")
    if reverse :
        ylabel('time (s)')
        xlabel('health mode')
        xlim(.75, len(categories) +0.25)
    else:
        xlabel('time (s)')
        ylabel('health mode')
        ylim(.75, len(categories) +0.25)
    legend(loc=loc, handler_map={l1: HandlerLine2D(numpoints=2),l2: HandlerLine2D(numpoints=1),l3: HandlerLine2D(numpoints=1)}, prop={'size':14})
    rc('font', size=20)
    
    def proj(i) :
        x = (t[i-1] + t[i])/2
        y = (cat_dict[data['real behavioral mode'][i-1]] + cat_dict[data['real behavioral mode'][i]])/2
        return x,y

    events = data['events']
    diagevents = data['observed events']
    labels = {}
    for i in range(len(t)):
        if events[i] != 'no event':
            x,y = proj(i)
            style = 'sawtooth,pad=0.3'
            fc = 'red'
            if diagevents[i] == events[i]:
                style = 'round4,pad=0.3'
                fc = 'white'
            if reverse :
                labels[i] = annotate(events[i], xy=(y,x), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            else:
                labels[i] = annotate(events[i], xy=(x,y), xytext=(-30, 30), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = style, fc = fc, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    if toshow:
        show()

def draw_health_mode_belief_at_time(filename, times, loc='best', toshow = True):

    data = pandas.read_csv(filename, sep=';')
    t = data['time']

    #fig, axes = subplots(len(times), sharex=True, sharey=True, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k', squeeze = False)
    fig, axes = subplots(len(times), sharex=True, sharey=True, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', squeeze = False)

    interesting = sorted([c for c in data if c.startswith('(')], key=lambda x:x.split(',')[1])

    # map categories to y-values
    cat_dict = dict(zip(interesting, range(1, len(interesting)+1)))

    # map y-values to categories
    val_dict = dict(zip(range(1, len(interesting)+1), [i.split(' ')[0].replace(',',',\n') for i in interesting]))

    plotval = [cat_dict[i] for i in interesting]

    axes_list = [ax[0] for ax in axes]

    for time, ax in zip(times,axes_list):
        iTime = [i for i in range(len(data['time'])) if data['time'][i] == time][0]
        X = numpy.array(plotval)
        Y = numpy.array([data[c][iTime] for c in interesting])
        X2 = []
        Y2 = []
        for i in range(len(X)):
            if Y[i] > 0 :
                X2.append(X[i])
                Y2.append(Y[i])
    
        ax.bar([x - 0.005 for x in X], Y, width=0.01, color = 'black')
        ax.plot(X2, Y2, 'co', ms=11)

        maxi = max(data[m][iTime] for m in interesting)
        modes = [cat_dict[m] for m in interesting if data[m][iTime] == maxi]
        for m in modes : ax.plot([m], [maxi], 'ow', ms=8)

        ax.plot([cat_dict[data['real behavioral mode'][iTime]]], [-0.05], 'ok', ms =11)
        ax.grid(True)
        ax.set_ylabel('degré de croyance')
        ax.set_xlim(0, len(interesting) + 1)
        ax.set_ylim(-0.1, 1.1)
        ax2 = ax.twinx()
        ax2.set_ylabel('{}s'.format(time))
        ax2.axes.get_yaxis().set_ticks([])
        if time == times[0]:
            l1, = ax.plot([], [], 'ok', ms =11, label='real health mode')
            l2, = ax.plot([], [], 'oc', ms =11,label='possible health mode')
            l3, = ax.plot([], [], 'ow', label='prefered health mode')
            ax.legend(loc=loc, handler_map={line: HandlerLine2D(numpoints=1) for line in [l1,l2,l3]}, prop={'size':12})
            ax.set_title("Belief degrees of health modes")
        if time == times[-1]:
            ax.set_xlabel('health mode')
    gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))

    subplots_adjust(hspace=0)
    rc('font', size=20)

    if toshow:
        show()

def draw_modes(diag, modes, initParticle, command, simuPredictDeltaTime):
    # DEPRECIATED ?

    plt.figure(1)
    nb_row = 2
    nb_col = 2
    for i, p in zip(range(1,len(modes)+1), modes):
        subplot(nb_row, nb_col, i)
        simui = Simulator(model, '{}'.format(p[0]), '{}'.format(p[1]), initParticle)
        times = [simui.time]
        states = [simui.state.copy()]
        for _ in range(1000):
            simui.sense(command, set(), simuPredictDeltaTime)
            times.append(simui.time)
            states.append(simui.state)
        simui2 = Simulator(model, '{}'.format(p[0]), '{}'.format(p[1]), initParticle)
        times2 = [simui2.time]
        states2 = [simui2.state.copy()]
        simui2.sense(command, set(), 1000)
        times2.append(simui2.time)
        states2.append(simui2.state.copy())
    
        for i in range(3):
            plot(times, [s[i] for s in states], marker='.', label='h{}'.format(i+1))
        plot(times2, states2, marker='v')
        title("({}, {}) simulated".format(p[0], p[1]))
        xlabel('time (s)')
        ylabel('height (m)')
        legend()
    tight_layout(pad=0.1, w_pad=5, h_pad=1)
    show()
    
def video(filename, minimumB = 0, toshow = True, loc='best', selectedModes = None, minimum = 1, blacklist = None):


  data = read_data(filename)
  #data = data[:5]

  blacklist = ['Sensor BL BR FL FR fault']

  # correct time
  datat = []
  for t in data.t:
      datat.append(t - data.t[0])

  modes = sorted([m for ms in data['m'] for m in ms.split(';')])
  modes = set(m for m, nb in it.groupby(modes) if len(list(nb)) > minimum)
  modes = sorted(modes)
  print(len(modes), 'modes')
  
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

  #fig, [ax1,ax2] = plt.subplots(2, sharex=True,figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
  fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k') 
  gs = gridspec.GridSpec(2, 3) 
  ax1 = plt.subplot(gs[0,1:])
  ax2 = plt.subplot(gs[1,1:])
  ax3 = plt.subplot(gs[1,0])

  l1, = ax1.plot([], [], 'k-', lw=10, label='mode belief')
  l2, = ax1.plot([], [], 'c-', lw=10, label='higher belief')
  
  val_dict = {v: s.replace(' + ', '\n+ ') for v,s in val_dict.items()}
  val_dict = {v: s.replace('BR FL', 'BR\nFL') for v,s in val_dict.items()}
  val_dict = {v: s.replace('FL FR', 'FL\nFR') for v,s in val_dict.items()}
  ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))

  pointplot1, = ax1.plot([], [])
  ax1.grid(True)
  ax1.set_title("Mode belief distributions")
  #ax1.set_xlabel('time (s)')
  ax1.set_ylabel('mode')
  ax1.set_ylim(min(val_dict)-0.2, max(val_dict) + .2)
  ax1.set_xlim(min(datat), max(datat))
  ax1.get_yaxis().set_ticks(list(range(1, len(val_dict)+1))) # force y axis labels
  ax1.get_xaxis().set_ticklabels([]) # remove x axis labels
  ax1.legend(loc=4, handler_map={line: HandlerLine2D(numpoints=1) for line in [l1,l2]}, prop={'size':13})

  res4 = {'k': ([], [], []), 'c' : ([], [], [])}
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
                  res4[color][0].append(t)
                  res4[color][1].append(rul)
                  res4[color][2].append(b/maxi)
              t_min_ruls.append(t)
              min_ruls.append(min(ruls))

  # add a fake belief to scale colormap
  res5 = [None] * 3
  res5[0] = res4['k'][0]+res4['c'][0]
  res5[1] = res4['k'][1]+res4['c'][1]
  res5[2] = res4['k'][2]+res4['c'][2]
  res4 = res5

  #if eol is not None:
  #    plot([0, eol], [eol,0], 'b--', label = 'real EOL', lw=2)
  #if eom is not None:
  #    plot([0, eom], [eom,0], 'b--', label = 'real EOM', lw=2)
  #if minimum:
  #    plot(t_min_ruls, min_ruls, 'k.', label = 'min. RUL', lw=2)

  #if eol is not None or eom is not None or showEop or minimum:
  #    legend(loc=loc)#, prop={'size':30})

  pointplot2, = ax2.plot([], [])
  ax2.grid(True)
  ax2.set_title("RUL estimations")
  ax2.set_ylim(0, max(res4[1])+100)
  ax2.set_xlim(min(datat), max(datat))
  ax2.set_xlabel('time (s)')
  ax2.set_ylabel('RULs (s)')

  data = pandas.read_csv('rover_lumped/parasitic_load_fault_no_replan.data', header=None)
  x2 = np.array(data[70])
  y2 = np.array(data[71])
  # fix bad values
  x2[3595] = x2[3594]
  y2[3595] = y2[3594]
  # scale value to img (empirically)
  x = (y2-y2[0]) * 100000 * (798.75-64.18)/(563.3-64.18)+ 64.18
  y = -(x2-x2[0]) * 100000 * -(305.7-75.09)/(-124.42)+ 305.70

  ax3.set_title("Rover positions")
  img=mpimg.imread('rover_lumped/faulty-path.png')
  ax3.imshow(img,aspect='equal')
  pointplot3, = ax3.plot([], [])
  ax3.set_ylim(761,0)
  ax3.set_xlim(0, 866)
  ax3.axes.get_xaxis().set_visible(False)
  ax3.axes.get_yaxis().set_visible(False)

  gs.tight_layout(fig)
  rc('font', size=13)
  #subplots_adjust(hspace=.1)
  gs.update(wspace=0.22, hspace=0.1)
  for ax in fig.axes: ax.set_axisbelow(True)

  def animate(i):
    # update plot
    print(i, '/', len(datat))
    c = 'k'
    res1 = []
    res2 = []
    res3 = []
    for e1, e2, e3 in zip(res[c][0], res[c][1], res[c][2]):
      if e1 == i:
        res1.append(e1) 
        res2.append(e2)
        res3.append(e3)
    pointplot1 = ax1.scatter(res1, res2, s=res3, c=c, marker='|')
    c = 'c'
    res1 = []
    res2 = []
    res3 = []
    for e1, e2, e3 in zip(res[c][0], res[c][1], res[c][2]):
      if e1 == i:
        res1.append(e1) 
        res2.append(e2)
        res3.append(e3)
    pointplot1 = ax1.scatter(res1, res2, s=res3, c=c, marker='|')

    # rul
    res1 = []
    res2 = []
    res3 = []
    for e1, e2, e3 in zip(res4[0], res4[1], res4[2]):
      if e1 == i:
        res1.append(e1) 
        res2.append(e2)
        res3.append(e3)
    if len(res1) > 0 :
      res1 = [0] + res1 + [0]
      res2 = [max(res4[1]) + 10000] + res2 + [max(res4[1]) + 10000]
      res3 = [-.1] + res3 + [max(res4[2])]
    cmap = cm.get_cmap('bone_r')
    eol=4571
    if round(i) == eol:
    #if i == datat[-1]:
      pointplot2 = ax2.plot([0, eol], [eol,0], 'b--', label = 'real EOL', lw=2)
      ax2.legend(loc=1)#, prop={'size':30})

    pointplot2 = ax2.scatter(res1, res2, c=res3, s=3, marker='s', edgecolors='face', cmap=cmap)

    ii = datat.index(i)
    X = [x[ii]]
    Y = [y[ii]]
    pointplot3, = ax3.plot(X, Y, 'bo')

    return  [pointplot1, pointplot2, pointplot3]

  anim = animation.FuncAnimation(fig, animate, repeat = False, frames=datat, interval=15, blit=True, repeat_delay=1000)
  anim.save('out.mp4', dpi=200, codec='libx264')
  plt.close(fig)

def video_mode_belief(filename, minimumB = 0, toshow = True, loc='best', selectedModes = None, minimum = 1, blacklist = None):


  data = read_data(filename)
  data = data[:1000]

  blacklist = ['Sensor BL BR FL FR fault']

  # correct time
  datat = []
  for t in data.t:
      datat.append(t - data.t[0])

  modes = sorted([m for ms in data['m'] for m in ms.split(';')])
  modes = set(m for m, nb in it.groupby(modes) if len(list(nb)) > minimum)
  modes = sorted(modes)
  print(len(modes), 'modes')
  
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

  #fig, [ax1,ax2] = plt.subplots(2, sharex=True,figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
  fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k') 
  gs = gridspec.GridSpec(1, 1) 
  ax1 = plt.subplot(gs[0,0])

  l1, = ax1.plot([], [], 'k-', lw=10, label='mode belief')
  l2, = ax1.plot([], [], 'c-', lw=10, label='higher belief')
  
  val_dict = {v: s.replace(' + ', '\n+ ') for v,s in val_dict.items()}
  val_dict = {v: s.replace('BR FL', 'BR\nFL') for v,s in val_dict.items()}
  val_dict = {v: s.replace('FL FR', 'FL\nFR') for v,s in val_dict.items()}
  ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: val_dict.get(x) if val_dict.get(x) is not None  else ''))

  pointplot1, = ax1.plot([], [])
  ax1.grid(True)
  ax1.set_title("Mode belief distributions")
  #ax1.set_xlabel('time (s)')
  ax1.set_ylabel('mode')
  ax1.set_ylim(min(val_dict)-0.2, max(val_dict) + .2)
  ax1.set_xlim(min(datat), max(datat))
  ax1.get_yaxis().set_ticks(list(range(1, len(val_dict)+1))) # force y axis labels
  ax1.get_xaxis().set_ticklabels([]) # remove x axis labels
  ax1.legend(loc=4, handler_map={line: HandlerLine2D(numpoints=1) for line in [l1,l2]}, prop={'size':13})

  gs.tight_layout(fig)
  rc('font', size=13)
  #subplots_adjust(hspace=.1)
  gs.update(wspace=0.22, hspace=0.1)
  for ax in fig.axes: ax.set_axisbelow(True)

  def animate(i):
    # update plot
    print(i, '/', len(datat))
    c = 'k'
    res1 = []
    res2 = []
    res3 = []
    for e1, e2, e3 in zip(res[c][0], res[c][1], res[c][2]):
      if e1 == i:
        res1.append(e1) 
        res2.append(e2)
        res3.append(e3)
    pointplot1 = ax1.scatter(res1, res2, s=res3, c=c, marker='|')
    c = 'c'
    res1 = []
    res2 = []
    res3 = []
    for e1, e2, e3 in zip(res[c][0], res[c][1], res[c][2]):
      if e1 == i:
        res1.append(e1) 
        res2.append(e2)
        res3.append(e3)
    pointplot1 = ax1.scatter(res1, res2, s=res3, c=c, marker='|')

    return  [pointplot1]

  anim = animation.FuncAnimation(fig, animate, repeat = False, frames=datat, interval=15, blit=True, repeat_delay=1000)
  anim.save('out.mp4', dpi=200, codec='libx264')
  plt.close(fig)

def draw_rul_error(filename, eol = None, eom=None, k = None, k2= None):

    data = read_data(filename)

    if not k : k = 0
    if not k2 : k2 = eol

    # color : (times, ruls, weights)
    res = []
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

            if len(ruls) > 0 and t >= k and t <= k2:
                res.append([t, min(ruls), max(ruls)])

    #for r in res :
    #  print(eol, r[1], (r[1]-(eol-r[0]))/abs(eol-r[0]))
    ave_rel_error = np.average([(r[1]-(eol-r[0]))/abs(eol-r[0]) for r in res])
    min_before_eol = sum(1 for r in res if r[1]<= (eol-r[0])) / len(res)
    #print('sum', sum(1 for r in res if r[1] <= (eol-r[0])), len(res))

    for r in res :
      if r[1] > (eol-r[0]):
        print(r[0], r[1])

    print(len(res), 'samples')
    print('minimum rul before eol (%)', min_before_eol)
    print('average relative error (minimum and eol)', ave_rel_error)

