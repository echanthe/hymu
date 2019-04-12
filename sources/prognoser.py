""" This module contains the implementation of a HPPN prognoser
"""

from hppn import HPPN, Transition, Configuration, Particle, HybridToken, Possibility, api_update_particle_values
from equation import NoiselessEquation, true, false, stochastic_resource_assignation, sure_resource_assignation
import log as logging
from progressbar import ProgressBar
import numpy as np
import itertools as it
import random
from termcolor import colored

# log and file handler
log = logging.getLogger(__name__[:4])

class Prognoser (HPPN) :
    """ This claas implement a prognoser based on a HPPN formalism
    A prognoser is generated from a given HPPN model
    """

    def __init__(self, model, minResources=None, suffResources=None, maxResources=None, predictionHorizon=None):
        """ Create a Prognoser object from a given HPPN model
        Prognoser features are generated during the creation :
            - remove noises in model dynamics
        
        >>> prog = test.simple_prognoser()
        """
        super().__init__('{} prognoser'.format(model.name))
        self.fill(model)
        self.remove_numerical_place_noises() 
        self.remove_unsatisfiable_symbolic_conditions()
        self.transform_conditions()
        self.delete_uninformative_transitions()
        self.groups = {}
        self.minResources = minResources
        self.suffResources = suffResources
        self.maxResources = maxResources
        self.predictionHorizon = predictionHorizon


        # create end mode list
        self.end_modes = self.get_end_modes()# + self.get_locker_modes()

        #log.info('  {} generated with {} places and {} transitions'.format(self.name, len(list(self.place())), len(list(self.transition()))))
        #log.info('  {} end modes'.format(len(self.end_modes)))
        #log.debug('  end modes: {}'.format([m.name for m in self.end_modes]))

    def remove_numerical_place_noises(self):
        """ Remove noises from continuous dynamics associated to NumericalPlace

        >>> prog = test.simple_prognoser()
        """
        for pn in self.numerical_places():
            ne = pn.ssr.stateNoiseEquation
            pn.ssr.stateNoiseEquation = NoiselessEquation(ne)
            ne = pn.ssr.outputNoiseEquation
            pn.ssr.outputNoiseEquation = NoiselessEquation(ne)

    def remove_unsatisfiable_symbolic_conditions(self):
        """  Put all unsatisfiable symbolic conditions to false()

        >>> prog = test.simple_prognoser()
        >>> sum(1 for t in prog.transition() if t.cs == false)
        2
        """
        for t in self.transition():
            if t.cs != true and t.cs != false:
                event = t.cs.event
                if event not in self.udo:
                    t.cs = false

    def transform_conditions(self):
        """ Transform transition conditions with the following rules:
         - if no hybid condition: set hybrid condition to false
         - if no observable condition: set both to false
         - if only one observable condition: set the other one to true

        """
        # check if a function is a 'condition'
        def cond(func):
            return not (func == true or func == false)

        for t in self.transition():
            if not cond(t.ch): t.ch = false
            if not cond(t.cs) and not cond(t.cn):
                t.cs = false
                t.cn = false
            else:
                if not cond(t.cs): t.cs = true
                if not cond(t.cn): t.cn = true

    def delete_uninformative_transitions(self):
        """  Delete all uninformative Transition
        A Transition is uninformative if its conditions are reductible to true of false

        >>> prog = test.simple_prognoser()
        >>> len(list(prog.transition()))
        7
        """
        # check if a function is a 'condition'
        def cond(func):
            return not (func == true or func == false)

        toDel = []
        for t in self.transition():
            if all([not cond(c) for c in (t.cs, t.cn, t.ch)]):
                toDel.append(t)
        for t in toDel:
            try:
                self.remove_transition(t.name)
            except:
                raise Exception('{} not in {}'.format(t.name, [tr.name for tr in self.transition()]))

    def divide(self, poss, nb):
        """ Divide the given Possibility

        >>> prog = test.simple_prognoser()
        >>> c = Configuration()
        >>> particles = [Particle(np.array([0])) for _ in range(100)]
        >>> hybridTokens = [HybridToken(np.array([0]), [], c, pi) for pi in particles]
        >>> poss = Possibility(c)
        >>> possibilities = prog.divide(poss, 100)
        >>> len(possibilities)
        100
        >>> from random import random
        >>> c = Configuration()
        >>> particles = [Particle(np.array([random(), random()])) for _ in range(100)]
        >>> hybridTokens = [HybridToken(np.array([random(), random()]), [], c, pi) for pi in particles]
        >>> poss = Possibility(c)
        >>> possibilities = prog.divide(poss, 80)
        >>> len(possibilities)
        80
        """
        try:
            choosen = random.sample(list(poss.hybrid_tokens()), nb)
        except ValueError as e:
            n = len(list(poss.hybrid_tokens()))
            log.warning(str(e) + ', len(list(poss.hybrid_tokens()))={}, nb={}'.format(n, nb))
            choosen = random.sample(list(poss.hybrid_tokens()), min((nb,n)))


        # no need with random.sample
        ## remove doubloons
        #choosen = set(choosen)

        # NOT ACCURATE : couple value (pi.x, ht.h) -> TOO EXPENSIVE
        #values = []
        #inter = []
        #for ht in interrestingH:
        #    found = False
        #    for val in values:
        #        if np.all(ht.h == val):
        #            found = True
        #    if not found:
        #        values.append(ht.h)
        #        inter.append(ht)

        # create new possibilities
        newPossibilities = []
        for ht in choosen:
            cCopy = poss._configuration.copy()
            pCopy = ht.particle.copy()
            # create new mode sequences to include the modes of Prognoser
            # instead of keeping previous HPPN modes
            mCopy = [(self.get_mode_by_name(m.name), t) for (m,t) in ht.modes]
            hCopy = HybridToken(ht.h, mCopy, cCopy, pCopy, ht.place)
            newPossibilities.append(Possibility(cCopy))

        return newPossibilities

    @HPPN.info()
    def initialize(self, marking, t0, uc0 = None, ud0 = None):
        """ Initializes the Prognoser:
            - reset cluster sets from the given marking
            - reset net marking with cluster

        >>> model = test.simple_model()
        >>> prog = Prognoser(model, 10, 100, 1000, 10000)
        >>> prog.initialize(model.tokens(), 0)
        >>> len(list(prog.tokens()))
        300
        """
        self.t0 = t0
        self.uck = uc0
        self.udk = ud0

        possibilities = [c.possibility for c in marking if isinstance(c, Configuration)]
        
        # do not recompute prognosis
        # false if the continuous state influence the prognosis
        #computed = {p.id:p for p in self.groups}
        #groups = {}
        #newMarking = []
        #for p in possibilities:
        #    if p._belief >= minimumBelief:
        #        if p.id in computed:
        #            poss = computed[p.modes]
        #            groups[poss] = self.groups[poss]
        #        else:
        #            groups[p] = self.divide(p, compactionCoefficient)
        #            # do not added tokens of possibilities already prognosed
        #            newMarking.extend(t for poss in groups[p] for t in poss.tokens())

        groups = {}
        scores = {p:p._belief for p in possibilities}

        assignations, matchButNotEnought = stochastic_resource_assignation(scores, self.minResources, self.suffResources, self.maxResources)
        # remove not choosen possibilities
        assignations = {a: nb for a, nb in assignations.items() if nb > 0}
        if len(assignations) == 0:
            if matchButNotEnought:
                print(colored(' TOO FEW RESOURCES', 'red', attrs=['bold', 'reverse']))
                # assign a new particle number for each cluster
                assignations = sure_resource_assignation(scores, self.minResources, self.suffResources, self.maxResources)
                # remove not choosen possibilities
                assignations = {a: nb for a, nb in assignations.items() if nb > 0}

        # assign
        for p,nb in assignations.items():
            if nb > 0: # avoid possibilities with no prediction to end in prognosis results
                groups[p] = self.divide(p, nb)
        newMarking = [t for poss in groups.values() for p in poss for t in p.tokens()]
        super().initialize(newMarking, t0)

        # recover possibilities created during initializing
        # possibilities created in divide() still point on configuration
        groups = {p: [poss.configuration().possibility for poss in posss]for p, posss in groups.items()}

        self.groups = groups

    def update(self, t, uc, ud, particles, hybridTokens):
        """ Update the Prognoser with the given commands
          - fire according to the discrete command
          - update tokens values
        Only update given tokens to gain computation time

        >>> prog = test.simple_prognoser()
        >>> len(list(prog.tokens()))
        300
        >>> prog.update(1, np.array([4,4,4]), set('e1'), list(prog.particles()), list(prog.hybrid_tokens()))
        >>> len(list(prog.tokens()))
        300
        >>> len(prog.place('OK1').tokens)
        100
        >>> len(prog.place('ON1').tokens)
        100
        >>> len(prog.place('OK1_ON1').tokens)
        100
        """
        dt = t - self.k
        assert(dt >= 0)

        # use old commands
        oldUc = self.uck
        if oldUc is None: oldUc = uc
        oldUd = self.udk
        if oldUd is None: oldUd = ud

        for c in self.configurations():
            c.update_events(oldUd, self.k)
        self.fire()
        self.update_hybrid_token_modes()
        api_update_particle_values(particles, self.k, dt, oldUc)
        HPPN.api_update_hybrid_token_values(hybridTokens, dt)

        self.k = t
        # update old commands
        self.uck = uc
        self.udk = ud

    @HPPN.info()
    def prognose(self, ts, ucs, uds, display = False):
        """ Running the prognoser consist in firing the net and 
        udpating token values according to the given commands 
        Ones command scenario is over, use the last command and timestamp to
        continues prediction until all tokens are in an end mode or the prediction 
        horizon is reached

        >>> prog = test.simple_prognoser()
        >>> dt = 60
        >>> uc = np.array([1., 1., 1.])
        >>> prog.predictionHorizon = 0
        >>> prog.prognose([dt, dt], [uc, uc], [set(), set()])
        """
        # construct future times and inputs
        toFill = self.predictionHorizon - len(ts)
        if toFill < 0:
            # reduce inputs
            ts = ts[:self.predictionHorizon]
            ucs = ucs[:self.predictionHorizon]
            uds = uds[:self.predictionHorizon]
        elif toFill > 0:
            # fill inputs with last continuous inputs

            # find last input and sampling time
            lastUc = ucs[-1]
            try:
                lastDt = ts[-1] - ts[-2]
            except IndexError:
                # not enought future input
                lastDt = ts[-1] - self.k

            # construct missing future inputs
            ucsToFill = np.array([lastUc for _ in range(toFill)])
            udsToFill = np.array([set() for _ in range(toFill)])
            tsToFill = np.array([ts[-1] + i*lastDt for i in range(1, toFill +1)])
            
            # concatenate given inputs with constructed inputs
            ts = np.concatenate((ts, tsToFill))
            ucs = np.concatenate((ucs, ucsToFill))
            uds = np.concatenate((uds, udsToFill))

        # FIXME unknow future input
        uds = [set()] * self.predictionHorizon
        #ucs = [np.array([100.] * 4)] * self.predictionHorizon
        # FIXME random future input in [0, 50, 51, ..., 100]
        #pucs = [float(0)] + [float(i) for i in range(50,101)]
        pucs = [float(0), float(50), float(100)]
        ucs = []
        for _ in range(self.predictionHorizon):
          a = np.random.choice(pucs)
          b = np.random.choice(pucs)
          ucs.append(np.array([a,b,a,b])) # same value on the left, same value on the right

        if display:
            bar = ProgressBar(len(ts), 60, 'prog', 0)
            bar.update(0)

        over = False
        piToUpdate = list(self.particles())
        htToUpdate = list(self.hybrid_tokens())

        i = 0
        # 'while loop' is prefered to 'for loop' with a break
        while not over and i < len(ts):
            self.update(ts[i], ucs[i], uds[i], piToUpdate, htToUpdate)
            piToUpdate, htToUpdate = self.tokens_not_in_end_mode()
            #assert(len(piToUpdate) == len(htToUpdate))
            over = len(piToUpdate) == 0
            if display:
                bar.update(i)
            i += 1

        # to complete bar
        if display:
            if not over:
                bar.update(len(ts))
            print()
            #if next(self.possibilities(), None) is not None:
            #    if over:
            #        end = 'tokens in end modes'
            #    else:
            #        end = 'max iteration reached'
            #    log.info('prognoser end with: {}'.format(end))
            #else:
            #    log.info('prognoser did not run because: no tokens')

    def tokens_not_in_end_mode(self):
        """ Return the particles and the hybrid tokens that are not in a end mode

        >>> prog = test.simple_prognoser()
        >>> pi, ht = prog.tokens_not_in_end_mode()
        >>> len(pi)
        100
        >>> len(ht)
        100
        >>> c = list(prog.configurations())
        >>> pi = list(prog.particles())
        >>> ht = list(prog.hybrid_tokens())
        >>> len(c)
        100
        >>> len(pi)
        100
        >>> len(ht)
        100
        >>> prog.empty()
        >>> for t in c: t.place = prog.place('STOPPED')
        >>> for t in pi : t.place = prog.place('OFF')
        >>> for t in ht : 
        ...     t.place = prog.place('STOPPED_OFF')
        ...     t.modes = [(prog.get_mode_by_name('STOPPED-OFF'), 0)]
        >>> prog.initialize(c + pi + ht, 1)
        >>> pi, ht = prog.tokens_not_in_end_mode()
        >>> len(pi)
        0
        >>> len(ht)
        0
        """
        possibilities = [p for p in self.possibilities() if p.mode() not in self.end_modes]
        particles = [pi for p in possibilities for pi in p.particles()]
        hybridTokens = [ht for p in possibilities for ht in p.hybrid_tokens()]
        return particles, hybridTokens

    def get_end_modes(self):
        """ Return the list of end modes of the Prognoser

        >>> prog = test.simple_prognoser()
        >>> sorted(m.name for m in prog.get_end_modes())
        ['KO1-OFF', 'KO12-OFF', 'STOPPED-OFF']
        """
        return [m for m in self.modes() if all(m != self.get_transition_input_mode(t) for t in self.transition())]

    #def get_locker_modes(self):
    #    """ Return the list of locker modes of the Prognoser
    #    that are not end modes

    #    >>> prog = test.simple_prognoser()
    #    >>> sorted(m.name for m in prog.get_locker_modes())
    #    []
    #    """
    #    possibles = [m for m in self.modes() if m not in self.get_end_modes()]
    #    lockers = []
    #    for m in possibles:
    #        outputTransitions = [t for t in self.transition() if m == self.get_transition_input_mode(t)]
    #        if all((t.cs == false or t.cn == false) and t.ch == false for t in outputTransitions):
    #            lockers.append(m)
    #    return lockers

    def possibility_to_prognosis(self, p, possibilities):
        """ Transform the given Possibility into Prognosis
        
        >>> prog = test.simple_prognoser()
        >>> p = list(prog.groups.keys())[0]
        >>> pr = prog.possibility_to_prognosis(p, prog.groups[p])
        """
        Prognosis = type('Prognosis', (), {})
        # create a Prognosis
        pr = Prognosis()

        # id
        pr.id = p.id

        # add future modes only
        lModes = len(p.modes())
        pr.modes = [poss.modes()[lModes:] for poss in possibilities]

        # token numbers
        pr.pon = len(possibilities)

        return pr

    #def merge_prognosis(self, prognosis):
    #    """ Merge the given Prognosis
    #    DEPRECIATED
    #    
    #    >>> prog = test.simple_prognoser()
    #    >>> prognosis = [prog.possibility_to_prognosis(p) for p in prog.possibilities()]
    #    >>> len(prog.merge_prognosis(prognosis))
    #    1
    #    """
    #    Prognosis = type('Prognosis', (), {})
    #    def modeKey(pr):
    #        return tuple(m[0] for m in pr.modes)

    #    clusters = sorted(prognosis, key=modeKey)
    #    clusters = it.groupby(clusters, key=modeKey)
    #    prognosis = []
    #    for mode, progs in clusters:
    #        progs = list(progs)

    #        # create a Prognosis
    #        pr = Prognosis()

    #        # modes
    #        pr.modes = []
    #        for i, m in enumerate(mode):
    #            mini = min(pr.modes[i][1] for pr in progs)
    #            maxi = max(pr.modes[i][1] for pr in progs)
    #            pr.modes.append((m, (mini,maxi)))

    #        # mode
    #        pr.mode = pr.modes[-1][0]

    #        # events
    #        pr.events = []
    #        events = progs[0].events
    #        events = tuple(e[0] for e in events)
    #        for i, e in enumerate(events):
    #            mini = min(pr.events[i][1] for pr in progs)
    #            maxi = max(pr.events[i][1] for pr in progs)
    #            pr.events.append((e, (mini,maxi)))
    #        
    #        # belief
    #        pr.belief = progs[0].belief

    #        # rul/eol
    #        if pr.mode in self.get_end_modes():
    #            mini = min([pr.eol[0] for pr in progs])
    #            maxi = max([pr.eol[1] for pr in progs])
    #            pr.rul = (max(mini - self.t0, 0), max(maxi - self.t0, 0))
    #        else:
    #            pr.eol = None
    #            pr.rul = None

    #        # token numbers
    #        pr.cn = sum(pr.cn for pr in progs)
    #        pr.pn = sum(pr.pn for pr in progs)
    #        pr.hn = sum(pr.hn for pr in progs)
    #        pr.pon = sum(pr.pon for pr in progs)

    #        prognosis.append(pr)

    #    return prognosis

    def prognosis(self):
        """ Return the distribution of prognosis
        computed by the Prognoser (after a run)

        >>> prog = test.simple_prognoser()
        >>> len(list(prog.prognosis()))
        1
        """
        #progs = [self.possibility_to_prognosis(p) for p in self.possibilities()]
        # take into account prognosis already computed
        progs = (self.possibility_to_prognosis(p, pp) for p, pp in self.groups.items())
        #keyfunc = lambda pr: pr.belief
        #progs = sorted(progs, reverse=True, key=keyfunc)
        #if merge:
        #    groups = it.groupby(progs, key=keyfunc)
        #    progs = []
        #    for b, g in groups:
        #        progs += self.merge_prognosis(list(g))
        return progs

if __name__ == "__main__":
    import doctest
    import test
    logging.disable(logging.INFO)
    doctest.testmod()
