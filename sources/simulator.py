""" This module provides a Runner object that is a simulator of the system.
Inits the simulator, reads input data to update the simulator state, and record simulated output data.

TODO: clean up, add comments and maybe doctest
"""

import log as logging
import numpy as np
import time
log = logging.getLogger(__name__)
from hppn import *
from progressbar import ProgressBar
from scenario import Scenario

class Runner(HPPN):

    def __init__(self, model, config):
        """ Create the simulator

        >>> simu = test.simple_simulator()
        """ 
        super().__init__('{} simulator'.format(model.name))
        #log.info('{} generation'.format(self.name))
        self.fill(model)
        self.initialize(config.T0, config.M0, config.X0, config.H0)
        self.pi = next(self.particles())
        self.c = next(self.configurations())
        self.ht = next(self.hybrid_tokens())
        self.poss = self.c.possibility

        self.scenario = Scenario()
        self.scenario.t.append(self.k)
        self.scenario.uc.append(None)
        self.scenario.ud.append(set())
        self.scenario.m.append(config.M0)
        self.scenario.x.append(config.X0)
        self.scenario.h.append(config.H0)
        self.scenario.yc.append(None)
        self.scenario.yd.append(set())

        #log.info('  {} generated with {} places and {} transitions'.format(self.name, len(list(self.place())), len(list(self.transition()))))

    @HPPN.info()
    def initialize(self, t0, m0, x0, h0, uc0=None, ud0=None):
        """ Initialize the simulator

        >>> simu = test.simple_simulator()
        """
        mode = self.get_mode_by_name(m0)
        p = Possibility.create(mode, t0, x0, h0, 1)
        marking = list(p.tokens())
        super().initialize(marking, t0)
        self.uc_k = uc0
        self.ud_k = ud0

    def is_possibility_accepted(self, t, poss):
        """ Change conditions

        >>> simu = test.simple_simulator()
        >>> poss = next(simu.possibilities())
        >>> simu.is_possibility_accepted(simu.transition('t0'), poss)
        False
        """
        # test behavioral conditions
        sym = t.cs(poss.configuration())
        num = sum(1 for pi in poss.particles() if t.cn(pi))
        num = num >= len(list(poss.particles()))/2
        hyb = sum(1 for ht in poss.hybrid_tokens() if t.ch(ht))
        hyb = hyb >= len(list(poss.hybrid_tokens()))/2
        if t.cn == true or t.cn == false or t.cs == true or t.cs == false:
            accepted = sym and num or hyb
        else:
            accepted = sym or num or hyb
        return accepted

    def update(self, t, uc, ud, yd):
        """ Update the simulator with the given commands
          - fire according to the discrete command
          - update tokens values
          - generate simulator output

        >>> simu = test.simple_simulator()
        >>> yc, yd = simu.update(1, np.array([4,4,4]), set('e1'), set())
        >>> len(yc)
        2
        >>> type(yd)
        <class 'set'>
        """
        dt = t - self.k
        assert(dt > 0)

        # use old commands if there is some
        # only usefull for k=0
        oldUd = self.ud_k
        if oldUd is None: oldUd = ud
        oldUc = self.uc_k
        if oldUc is None: oldUc = uc

        self.c.update_events(oldUd, self.k)
        self.fire()
        self.update_hybrid_token_modes()
        self.update_hybrid_token_values(dt)
        self.update_particle_values(dt, oldUc)

        # update old commands
        self.ud_k = ud | yd # include current discrete output
        self.uc_k = uc
        self.k = t

        #assert(len(list(self.particles())) == 1)
        #assert(len(list(self.configurations())) == 1)
        #assert(len(list(self.hybrid_tokens())) == 1)

        # use current commands
        yc = self.pi.place.ssr.noisy_output(self.pi.x, dt, uc, self.k)

        # DO NOT GENERATE AUTO OUTPUT -> think about it first
        #yd = set()
        #events = sorted(self.c.events)
        #if len(events) > 0 and events[-1][1] == self.k:
        #    yd = {events[-1][0]} | yd

        # record
        self.scenario.t.append(self.k)
        self.scenario.uc.append(uc)
        self.scenario.ud.append(ud)
        self.scenario.m.append(self.poss.mode())
        self.scenario.x.append(self.pi.x.copy())
        self.scenario.h.append(self.ht.h.copy())
        self.scenario.yc.append(yc)
        self.scenario.yd.append(yd)

        #log.debug('## {} ## SIMULATOR IN {}'.format(self.k, self.poss.mode()))
        #log.debug('## {} ## SIMULATOR UC {}'.format(self.k, uc))
        #log.debug('## {} ## SIMULATOR UD {}'.format(self.k, ud))
        #log.debug('## {} ## SIMULATOR X {}'.format(self.k, self.pi.x))
        #log.debug('## {} ## SIMULATOR H {}'.format(self.k, self.ht.h))
        #log.debug('## {} ## SIMULATOR YC {}'.format(self.k, yc))
        #log.debug('## {} ## SIMULATOR YD {}'.format(self.k, yd))
        return yc, yd

    @HPPN.info()
    def run(self, scenario) :
        """ Running the simulator consists in firing the net and 
        udpating token values according to the given commands 
        Then output data is record in a Scenario
        Return the name of the simulator

        >>> simu = test.simple_simulator()
        """
        s = scenario
        s.read(skiprows = [1])

        ts = s.ts()
        ucs = s.ucs()
        uds = s.uds()

        bar = ProgressBar(ts[-1], 60, 'simu', self.k)
        bar.update(self.k)

        logging.disable(logging.INFO)

        timer = time.time()

        for t, uc, ud in zip(ts, ucs, uds):
            yc, yd = self.update(t, uc, ud, set())

            bar.update(self.k)

        timer = time.time() - timer

        logging.disable(logging.NOTSET)
        print()

        return timer

    def write(self, filename):
        """ Write simulator scenario to given filename
        """
        self.scenario.write(filename)

if __name__ == "__main__":
    import doctest
    import test
    logging.disable(logging.INFO)
    doctest.testmod()
