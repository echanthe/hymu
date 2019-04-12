""" This module provides a monitor as a Runner object.
Reads config files, reads scenario data, updated the diagnoser and pronoser and write results.

TODO: clean up, add comments and maybe doctest
"""

import re
import numpy as np
import os
from hppn import *
import csv
import gc
from progressbar import ProgressBar
import pandas
import time
from timer import timed
import log as logging
import system_info
from diagnoser import Diagnoser
from prognoser import Prognoser
from termcolor import colored
import resource
import gzip
import itertools as it

# log and file handler
log = logging.getLogger(__name__[:4])

def info(s):
    if s != '':
        log.info(colored(' ' + s + ' ', attrs=['bold', 'reverse']))
    else:
        log.info('')

class RunnerData :

    def __init__(self, filename = None):
        self.t = [] # diagnoser time

        self.id = [] # diagnosis possibility ids
        self.fid = [] # diagnosis possibility father ids
        self.m = [] # diagnosis possibility current modes
        self.pn = [] # diagnosis particle numbers
        self.sb = [] # diagnosis possibility symbolic belief lists
        self.nb = [] # diagnosis possibility numerical belief lists
        self.b = [] # diagnosis possibility belief lists
        self.xmean = [] # diagnosis x mean
        self.xmin = [] # diagnosis x min
        self.xmax = [] # diagnosis x max
        self.ycmean = [] # diagnosis yc mean
        self.ycmin = [] # diagnosis yc min
        self.ycmax = [] # diagnosis yc max
        self.hmean = [] # diagnosis h mean
        self.hmin = [] # diagnosis h min
        self.hmax = [] # diagnosis h max

        self.dcput = [] # diagnosis cpu time
        self.maxmem = [] # monitor maximum memory used

        self.pe = [] # prognosis enabled

        self.ppon = [] # prognosis possibility numbers
        self.fms = [] # prognosis possibility mode lists
        self.lpt = [] # prognoser last prediction time

        self.pcput = [] # prognosis cpu time

    def write(self, filename, writeHeader):
        #with open(filename, 'a', newline='') as f:
       
        with gzip.open(filename, 'at') as f:
            
            data = csv.writer(f, delimiter='|')
    
            if self.xmean[-1] is not None : xNb = len(self.xmean[-1][0]) 
            else : xNb = 0
            if self.ycmean[-1] is not None : ycNb = len(self.ycmean[-1][0]) 
            else : ycNb = 0
            if self.hmean[-1] is not None : hNb = len(self.hmean[-1][0]) 
            else : hNb = 0

            if writeHeader:
                xmeanl = ['x_{}_mean'.format(j) for j in range(xNb)]
                xminl = ['x_{}_min'.format(j) for j in range(xNb)]
                xmaxl = ['x_{}_max'.format(j) for j in range(xNb)]
                ycmeanl = ['yc_{}_mean'.format(j) for j in range(ycNb)]
                ycminl = ['yc_{}_min'.format(j) for j in range(ycNb)]
                ycmaxl = ['yc_{}_max'.format(j) for j in range(ycNb)]
                hmeanl = ['h_{}_mean'.format(j) for j in range(hNb)]
                hminl = ['h_{}_min'.format(j) for j in range(hNb)]
                hmaxl = ['h_{}_max'.format(j) for j in range(hNb)]
                
                line = ['t', 'id', 'fid', 'm', 'pn', 'sb', 'nb', 'b'] + xmeanl + xmaxl + xminl + ycmeanl + ycmaxl + ycminl + hmeanl + hminl + hmaxl + ['dcput', 'maxmem']
                line += ['ppon', 'fms', 'lpt', 'pcput']

                data.writerow(line)

            psep = ';'

            zipper = zip(self.t,self.id,self.fid,self.m,self.pn,self.sb,self.nb,self.b,self.xmean,self.xmin,self.xmax,self.ycmean,self.ycmin,self.ycmax,self.hmean,self.hmin,self.hmax,self.dcput,self.maxmem,self.ppon,self.fms,self.lpt,self.pcput,self.pe)
            for t,id,fid,m,pn,sb,nb,b,xmean,xmin,xmax,ycmean,ycmin,ycmax,hmean,hmin,hmax,dcput,maxmem,ppon,fms,lpt,pcput,pe in zipper:

                if b : possNb = len(b)
                else : possNb = 0
                id = psep.join(str(i) for i in id) if id else 'no'
                fid = psep.join(str(fi) for fi in fid) if fid else 'no'
                m = psep.join(str(pm) for pm in m) if m else 'no'
                pn = psep.join(str(n) for n in pn) if pn else 'no'
                sb = psep.join(str(pb) for pb in sb) if sb else 'no'
                nb = psep.join(str(pb) for pb in nb) if nb else 'no'
                b = psep.join(str(pb) for pb in b) if b else 'no'
                ppon = psep.join(str(n) if n else 'no' for n in ppon) if ppon else 'no'
                fms = psep.join(''.join(str(m) for m in fm) if fm else 'no' for fm in fms) if fms else 'no'

                xmeanl = []
                for j in range(xNb):
                    xmeanl.append(psep.join(str(xmean[i][j]) for i in range(possNb)))
                xminl = []
                for j in range(xNb):
                    xminl.append(psep.join(str(xmin[i][j]) for i in range(possNb)))
                xmaxl = []
                for j in range(xNb):
                    xmaxl.append(psep.join(str(xmax[i][j]) for i in range(possNb)))
                ycmeanl = []
                for j in range(ycNb):
                    try:
                        ycmeanl.append(psep.join(str(ycmean[i][j]) for i in range(possNb)))
                    except:
                        ycmeanl.append(psep.join('no' for i in range(possNb)))
                ycminl = []
                for j in range(ycNb):
                    try:
                        ycminl.append(psep.join(str(ycmin[i][j]) for i in range(possNb)))
                    except:
                        ycminl.append(psep.join('no' for i in range(possNb)))
                ycmaxl = []
                for j in range(ycNb):
                    try:
                        ycmaxl.append(psep.join(str(ycmax[i][j]) for i in range(possNb)))
                    except:
                        ycmaxl.append(psep.join('no' for i in range(possNb)))
                hmeanl = []
                for j in range(hNb):
                    hmeanl.append(psep.join(str(hmean[i][j]) for i in range(possNb)))
                hminl = []
                for j in range(hNb):
                    hminl.append(psep.join(str(hmin[i][j]) for i in range(possNb)))
                hmaxl = []
                for j in range(hNb):
                    hmaxl.append(psep.join(str(hmax[i][j]) for i in range(possNb)))
                    
                line = [t, id, fid, m, pn, sb, nb, b] + xmeanl + xmaxl + xminl + ycmeanl + ycmaxl + ycminl + hmeanl + hminl + hmaxl + [dcput, maxmem]
                if pe:
                    line += [ppon, fms, lpt] + [pcput]

                data.writerow(line)

            f.close() # force file closure because of some Permission errors


class Runner:

    def __init__(self, model, config):
        
        self.initialNbObjects = len(gc.get_objects())
        self.diag = Diagnoser(model,
                config.DIAGNOSER_MINIMUM_RESOURCE_NUMBER, 
                config.DIAGNOSER_SUFFICIENT_RESOURCE_NUMBER, 
                config.DIAGNOSER_MAXIMUM_RESOURCE_NUMBER,
                config.DIAGNOSER_CONFIDENCE_SYMBOLIC_NUMERICAL)
        self.diag.initialize(config.T0, config.M0, config.X0, config.H0,
                config.DIAGNOSER_INITIAL_RESOURCE_NUMBER, config.UC0)
        self.prognoserEnabled = config.PROGNOSER_ENABLED
        constructPrognoser = True
        # infite cycle
        if hasattr(config.PROGNOSER_ENABLED, "__getitem__"):
            # PROGNOSER_ENABLED is subscriptable
            self.prognoserEnabled = it.cycle(config.PROGNOSER_ENABLED)
        elif isinstance(self.prognoserEnabled, bool):
            # PROGNOSER_ENABLED is a boolean
            self.prognoserEnabled = it.cycle([config.PROGNOSER_ENABLED])
            constructPrognoser = config.PROGNOSER_ENABLED
        else:
            # wrong value TODO: check config param
            self.prognoserEnabled = it.cycle([False])
            constructPrognoser = False
            print("wrong value ?")

        if constructPrognoser:
            print("construction prog")
            self.prog = Prognoser(model,
                    config.PROGNOSER_MINIMUM_RESOURCE_NUMBER,
                    config.PROGNOSER_SUFFICIENT_RESOURCE_NUMBER,
                    config.PROGNOSER_MAXIMUM_RESOURCE_NUMBER,
                    config.PROGNOSER_PREDICTION_HORIZON)
        else:
            self.prog = Prognoser(HPPN('empty'))
        self.data = RunnerData()
        self.nbObjects = len(gc.get_objects()) - self.initialNbObjects
        self.maxMemoryUsed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._record(0, None, 0, 0, False)

        n = model.name
        badChar = " ()'\/[]{},;:!?."
        for i in badChar: n = n.replace(i, '_')
        self.name = '{}_monitor'.format(n)

    def _record(self, dt, uc, diagCpuTime, progCpuTime, prognoserEnabled):
        self.data.t.append(self.diag.k)

        diags = sorted(self.diag.diagnosis(dt, uc), key=lambda d: d.belief)

        self.data.id.append([d.id for d in diags])
        self.data.fid.append([d.fatherId for d in diags])
        self.data.m.append([d.mode for d in diags])
        self.data.pn.append([d.pn for d in diags])
        self.data.sb.append([d.symbolicBelief for d in diags])
        self.data.nb.append([d.numericalBelief for d in diags])
        self.data.b.append([d.belief for d in diags])
        self.data.xmean.append([d.xmean for d in diags])
        self.data.xmin.append([d.xmin for d in diags])
        self.data.xmax.append([d.xmax for d in diags])
        self.data.ycmean.append([d.ycmean for d in diags])
        self.data.ycmin.append([d.ycmin for d in diags])
        self.data.ycmax.append([d.ycmax for d in diags])
        self.data.hmean.append([d.hmean for d in diags])
        self.data.hmin.append([d.hmin for d in diags])
        self.data.hmax.append([d.hmax for d in diags])

        self.data.dcput.append(diagCpuTime)
        self.data.maxmem.append(self.maxMemoryUsed)

        self.data.pe.append(prognoserEnabled)

        if prognoserEnabled:

            progs = {pr.id : pr for pr in self.prog.prognosis()}
            progs = [progs.get(d.id) for d in diags]

            # check if the None Prognosis are the last ones of the progs list
            # because diags are sorted by belief and possibility with too small belief are not prognosed
            # FALSE with the stochastic assignation
            #try:
            #    iNone = progs.index(None)
            #except:
            #    iNone = None
            #    assert(len(progs) == len(diags))
            #if iNone is not None:
            #    nbNone = progs.count(None)
            #    assert(len(progs) == iNone + nbNone)
            #    # remove None Prognosis
            #    progs = progs[:iNone]

            self.data.ppon.append([p.pon if p else None for p in progs])
            self.data.fms.append([p.modes if p else None for p in progs])
            self.data.lpt.append(self.prog.k)

            self.data.pcput.append(progCpuTime)

        else: 
            # fill with None
            self.data.ppon.append(None)
            self.data.fms.append(None)
            self.data.lpt.append(None)
            self.data.pcput.append(None)
        
        
        
        
        filepath = os.path.dirname(__file__)            
        # write file header
        writeHeader = not system_info.file_exist(filepath + os.path.sep + 'tmp')
        if writeHeader:
            write = len(self.data.t) > 2
        else:
            write = True

        if write:
            self.data.write(filepath + os.path.sep + 'tmp', writeHeader)
            # empty saved data
            for attr in self.data.__dict__.keys():
                del getattr(self.data, attr)[:]

    @timed
    def diagnose(self, *args, **kargs):

        self.diag.update(*args, **kargs)
 

    @timed
    def prognose(self, *args, **kargs):
        logging.disable(logging.INFO)
        self.prog.prognose(*args, **kargs)
        logging.disable(logging.NOTSET)

    def run(self, scenario):
        s = scenario
        #s.read(skiprows = [1], nrows = 2)
        #s.read(skiprows = [1], nrows = 100)
        s.read(skiprows = [1])

        ts = s.ts()
        ucs = s.ucs()
        uds = s.uds()
        #ms = s.ms()
        ycs = s.ycs()
        yds = s.yds()
        
        bar = ProgressBar(ts[-1], 60, 'moni', self.diag.k)
        bar.update(self.diag.k)
        print()    
        if logging.ENABLED:
            print()

        realTime0 = time.time()
        realTime = (realTime0, ts[-1] - ts[0])

        block = False

        #for i, (t, uc, ud, m, yc, yd) in enumerate(zip(ts, ucs, uds, ms, ycs, yds)):
        for i, (t, uc, ud, yc, yd) in enumerate(zip(ts, ucs, uds, ycs, yds)):
            #info('')
            #info('DIAG UPDATE...')
            _, diagTimer = self.diagnose(t, uc, ud, yc, yd)
            # print diag result
            #info('DIAG time: {}s'.format(diagTimer))
            #info('diag belief superior to {} (belief, modes):'.format(0))
            #diags = self.diag.diagnosis(1, None, 0, False)
            #for d in diags:
            #    modes = [(m[0], '{} (+ {})'.format(m[1], m[1]-ts[0])) for m in d.modes]
            #    info('  {}, {}'.format(d.belief, modes))
            #    if d.mode.name.endswith('load'):
            #        info('{}'.format(d.xmean[30]))
            #mergedDiags = self.diag.diagnosis(1, None, 0, True)
            #if len(mergedDiags) < len(diags):
            #    info('diag belief superior to {} (belief, modes) [MERGED]:'.format(0))
            #    for d in mergedDiags:
            #        info('  {}, {}'.format(d.belief, d.modes))

            if block:
                diags = self.diag.diagnosis(1, None, 0, False)
                if all([len(d.modes) > 1 for d in diags]):
                    # debug: draw cause of the non match
                    import equation as eq
                    ids = []
                    posss = self.diag.possibilities()
                    posss = (p for p in posss if len(p.modes()) == 1 and p.mode().name == 'Sensor BL FL fault')
                    dt = ts[i] - ts[i-1]
                    for p in posss:
                        particles = list(p.particles())
                        xs = [pi.x for pi in particles]
                        eycs = [pi.place.ssr.output(pi.x, dt, uc) for pi in particles]
                        mus = [pi.place.ssr.outputNoiseEquation.mu for pi in particles]
                        sigmas = [pi.place.ssr.outputNoiseEquation.sigma for pi in particles]
                        diff = [eq.gaussian(mu, eyc-yc, sigma) for mu, eyc, sigma in zip(mus, eycs, sigmas)]
                        weights = [np.multiply.reduce(d) for d in diff]
                        weightAdd = np.add.reduce(weights)
                        weightMul = np.multiply.reduce(weights)
                        diffAdd = np.add.reduce(diff)
                        diffMul = np.multiply.reduce(diff)
                        diffAve = np.average(diff, axis=0)
                        eycAve = np.average(eycs, axis=0)
                        xAve = np.average(xs, axis=0)
                        ids.append('{}, +{}, *{}, +{}, *{}'.format(p.modes(), weightAdd, weightMul, [i for i, d in enumerate(diffAdd) if d == 0], [i for i, d in enumerate(diffMul) if d == 0]))
                    print('Blocked because \n{}'.format('\n'.join(i for i in ids)))
                    print('truth lost')
                    #input()
                    block = False
            #if m != 'no':
            #    self.check(m)

            #info('')
            #info('PROG UPDATE...')
            
            prognoserEnabled = next(self.prognoserEnabled)
            if prognoserEnabled:
     
                # prog init
                self.prog.initialize(self.diag.tokens(), self.diag.k, uc, ud)
                futureTs = ts[i+1:]
                futureUcs = ucs[i+1:]
                futureUds = uds[i+1:]
                _, progTimer = self.prognose(futureTs, futureUcs, futureUds, display=True)
                # print prog result
                #info('PROG time: {}s'.format(progTimer))
                #info('prog belief superior to {} (belief, RUL/EOL, modes):'.format(self.minimumBelief))
                #progs = self.prog.prognosis(False)
                #for pr in progs:
                #    info('  {}, {}/{}, {}'.format(pr.belief, pr.rul, pr.eol, pr.modes))
                #mergedProgs = self.prog.prognosis(True)
                #if len(mergedProgs) < len(progs):
                #    info('prog belief superior to {} (belief, RUL/EOL, modes) [MERGED]:'.format(self.minimumBelief))
                #    for pr in mergedProgs:
                #        info('  {}, {}/{}, {}'.format(pr.belief, pr.rul, pr.eol, pr.modes))
                #input('check')
            else:
                progTimer = 0
            
            self._record(ts[i] - ts[i-1], uc, diagTimer, progTimer, prognoserEnabled)
            #oldRealTime = realTime
            #realTime = (time.time(), ts[-1] - t)

            ## time since simu started
            #tPast = realTime[0] - realTime0
            #sTPast = '{}'.format(time.strftime('%Hh%Mm%Ss',time.gmtime(tPast)))
            #
            ## remaining simu time
            #tRemain = - oldRealTime[1] / ((realTime[1] - oldRealTime[1]) / (realTime[0] - oldRealTime[0])) + realTime[0]
            #if abs(tRemain) != float('inf'):
            #    sTRemain = '{}'.format(time.strftime('%Hh%Mm%Ss',time.gmtime(tRemain)))
            #else:
            #    sTRemain = 'inf'
            #info('SIMU time: {}, remaining: {}'.format(sTPast, sTRemain))

            bar.update(self.diag.k)
            if logging.ENABLED or prognoserEnabled:
                print()
            old1 = self.nbObjects
            old2 = self.maxMemoryUsed
            self.nbObjects = len(gc.get_objects()) - self.initialNbObjects
            self.maxMemoryUsed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #info('GC: {} -> {} objects in memory, maximum memory used : {} -> {}'.format(old1, self.nbObjects, old2, self.maxMemoryUsed))
            #print(time.time() - realTime0)    
        #print(time.time() - realTime0)    
        return time.time() - realTime0
 
    #def check(self, simuMode):
    #    # DEPRECIATED
    #    b = {m:s for m,s in self.data.m[-1].items() if s > 0}
    #    maxi = max(b.values())
    #    bests = [best for best in b if b[best] == maxi]
    #    still_true = simuMode in b 
    #    if still_true:
    #        if simuMode in bests:
    #            if len(bests) == 1:
    #                s = "  DIAGNOSER GOT THE TRUTH AND IS THE BEST"
    #            else:
    #                s = "  DIAGNOSER GOT THE TRUTH AND IS THE BEST WITH AMBIGUITY"
    #        else:
    #            s = "  DIAGNOSER GOT THE TRUTH BUT IS NOT THE BEST"
    #        s += ' (real = {})'.format(simuMode)
    #        info(s)
    #    else:
    #        s = "  DIAGNOSER LOST THE TRUTH"
    #        s += ' (real = {})'.format(simuMode)
    #        info('')
    #        info(s)
    #        info('')

    def write(self, filename):
        filepath = os.path.dirname(__file__)
        import shutil
        filename += '.gz'
        if system_info.file_exist(filepath + os.path.sep + 'tmp'):
            shutil.move(filepath + os.path.sep + 'tmp', filename)
            info('scenario written in {}'.format(filename))
