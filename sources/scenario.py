""" This module provides a Scenario object that is used as an interface with system data.

TODO: clean up, add comments, doctest and manage data pipeline
"""

import csv
import gzip
import re
import numpy as np
import pandas
from progressbar import ProgressBar
class Scenario:

    def __init__(self, filename = None):
        self.t = []
        self.uc = []
        self.ud = []
        self.m = []
        self.x = []
        self.h = []
        self.yc = []
        self.yd = []

        self.data = None

        self.name = filename

    def events_map(self, events):
        if events != 'no':
            return set(events.split(','))
        else:
            return set()

    def ts(self, i = None):
        if i:
            return self.data['t'][i]
        else:
            return self.data['t'].values

    def ucs(self, i = None):
        uc = sorted((d for d in self.data if d.startswith('uc')), key=lambda d: int(d.split('_')[1]))
        if i:
            return self.data[uc].values[i]
        else:
            return self.data[uc].values
    
    def uds(self, i = None):
        if i:
            return events_map(self.data['ud'][i])
        else:
            return np.array([self.events_map(e) for e in self.data['ud'].values])
    
    def ms(self, i = None):
        if i:
            return self.data['m'][i]
        else:
            return self.data['m'].values

    def xs(self, i = None):
        x = sorted((d for d in self.data if d.startswith('x')), key=lambda d: int(d.split('_')[1]))
        if i:
            return self.data[x].values[i]
        else:
            return self.data[x].values

    def hs(self, i = None):
        h = sorted((d for d in self.data if d.startswith('h')), key=lambda d: int(d.split('_')[1]))
        if i:
            return self.data[h].values[i]
        else:
            return self.data[h].values

    def ycs(self, i = None):
        yc = sorted((d for d in self.data if d.startswith('yc')), key=lambda d: int(d.split('_')[1]))
        if i:
            return self.data[yc].values[i]
        else:
            return self.data[yc].values

    def yds(self, i = None):
        if i:
            return events_map(self.data['yd'][i])
        else:
            return np.array([self.events_map(e) for e in self.data['yd'].values])

    def write(self, filename):
        with open(filename, "w", newline='') as f:
            data = csv.writer(f, delimiter='|')

            barI = 0
            barL = len(self.t)
            bar = ProgressBar(barL, 10, 'writting scenario')
            bar.update(barI)
            
            if self.uc[-1] is not None : ucNb = len(self.uc[-1]) 
            else : ucNb = 0
            if self.x[-1] is not None : xNb = len(self.x[-1]) 
            else : xNb = 0
            if self.h[-1] is not None : hNb = len(self.h[-1]) 
            else : hNb = 0
            if self.yc[-1] is not None : ycNb = len(self.yc[-1]) 
            else : ycNb = 0

            ul = ['uc_{}'.format(i) for i in range(ucNb)]
            xl = ['x_{}'.format(i) for i in range(xNb)]
            hl = ['h_{}'.format(i) for i in range(hNb)]
            yl = ['yc_{}'.format(i) for i in range(ycNb)]
            line = ['t'] + ul + ['ud', 'm'] + xl + hl + yl + ['yd']
            data.writerow(line)
            for t,uc,ud,m,x,h,yc,yd in zip(self.t, self.uc, self.ud, self.m, self.x, self.h, self.yc, self.yd):
                udl = ','.join(e for e in ud) if ud else 'no'
                ydl = ','.join(e for e in yd) if yd else 'no'
                ml = str(m) if m else 'no'
                ucl = list(uc) if uc is not None else ['no'] * ucNb
                xl = list(x) if x is not None else ['no'] * xNb
                hl = list(h) if h is not None else ['no'] * hNb
                ycl = list(yc) if yc is not None else ['no'] * ycNb
                line = [t] + ucl + [udl,ml] + xl + hl + ycl + [ydl]
                data.writerow(line)
                barI += 1
                bar.update(barI)
            print()
            print('scenario written in {}'.format(filename))

    #def load(self, **args):
    #    """
    #    DEPRECIATED
    #    """

    #    self.read(**args)

    #    barI = 0
    #    barL = len(self.data)
    #    bar = ProgressBar(barL, 10, 'loading scenario')
    #    bar.update(barI)

    #    for t, uc, ud, m, x, h, yc, yd in self.get_data():
    #        self.t.append(t)
    #        self.uc.append(uc)
    #        self.ud.append(ud)
    #        self.m.append(m)
    #        self.x.append(x)
    #        self.h.append(h)
    #        self.yc.append(yc)
    #        self.yd.append(yd)
    #        barI += 1
    #        bar.update(barI)
    #    print()

    def map(self, j, symb, length):
        res = np.zeros(length)
        gen = (float(self.data[d][j]) if self.data[d][j] != 'no' else None for d in self.data if d.startswith(symb))
        for i, e in enumerate(gen): res[i] = e
        return res

    def read(self, **args):
        if self.name.endswith('.gz'):
            with gzip.open(self.name) as gzipfile:
                self.data = pandas.read_csv(gzipfile, sep='|', **args)
        else:
            self.data = pandas.read_csv(self.name, sep='|', **args)

    def get_data(self):

        luc = sum(1 for d in self.data if d.startswith('uc'))
        lx = sum(1 for d in self.data if d.startswith('x'))
        lh = sum(1 for d in self.data if d.startswith('h'))
        lyc = sum(1 for d in self.data if d.startswith('yc'))

        for i in range(len(self.data)):
            t = self.data['t'][i]
            uc = self.map(i, 'uc_', luc)
            ud = set(data['ud'][i].split(',')) if self.data['ud'][i] != 'no' else set() 
            m = self.data['m'][i]
            x = self.map(i, 'x_', lx)
            h = self.map(i, 'h_', lh)
            yc = self.map(i, 'yc_', lyc)
            yd = set(data['yd'][i].split(',')) if self.data['yd'][i] != 'no' else set()
            yield t, uc, ud, m, x, h, yc, yd
