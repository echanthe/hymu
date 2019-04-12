""" This module contains the implementation of a HPPN diagnoser
"""

from hppn import HPPN, SymbolicPlace, NumericalPlace, HybridPlace, Transition, Configuration, Particle, HybridToken, Possibility, GroupToken, api_update_particle_values
from equation import true, false, alpha_weighted_function, WeightedDistribution, gaussian, sure_resource_assignation, weird_resource_assignation, stochastic_resource_assignation
from termcolor import colored
import itertools as it
import log as logging
from functools import wraps
from optimization import optimization
import numpy as np

# log and file handler
log = logging.getLogger(__name__[:4])

# exception in this module
class ParticleMatchException(Exception): pass

class Diagnoser (HPPN) :
    """ This claas implement a diagnoser based on a HPPN formalism
    A diagnoser is generated from a given HPPN model
    """

    def __init__(self, model, minResources=None, suffResources=None, maxResources=None, globalConfidenceInSymbolicPart=None, postTreatement = True):
        """ Create a Diagnoser object from a given HPPN model
        Diagnoser features are generated during the creation :
            - remove condition on unobservable events because they are unsatisfiable (unobservable)
            - merge transitions modeling the same change of dynamics into a single one
        
        >>> diag = Diagnoser(test.simple_model())
        >>> diag = Diagnoser(test.simple_model(), False)
        """
        super().__init__('{} diagnoser'.format(model.name))

        self.fill(model)
        #self.add_hybrid_transitions(model)
        self.minResources = minResources
        self.suffResources = suffResources
        self.maxResources = maxResources
        self.globalConfidenceInSymbolicPart = globalConfidenceInSymbolicPart
        self.observedEvents = set()

        if postTreatement:
            self.remove_symbolic_and_hybrid_conditions()
            self.merge_transitions()

        #log.info('  {} generated with {} places and {} transitions'.format(self.name, len(list(self.place())), len(list(self.transition()))))

    def add_hybrid_transitions(self, hppn):
        trI = len(list(self.transition())) + 1
        for t in hppn.transition():
            ip = [p for p,_ in t.input() if isinstance(p,HybridPlace)]
            op = [p for p,_ in t.output() if isinstance(p, HybridPlace)]
            if ip[0] != op[0]:
                t = Transition('t{}'.format(trI), None, None, None)
                trI += 1
                super().add_transition(ip, t, op)

    def info(string = None):
        """ Logging decorator generator
    
        >>> deco = Diagnoser.info()
        """
        def info_deco(function):
            """ Logging decorator
            """
            if string is None:
                s = function.__name__
            @wraps(function)
            def info_func(*args, **kwargs):
                diag = args[0]
                name = diag.name
                name = name[:7]
                #log.info("{} {}|{} {}".format(name, colored(diag.k, attrs=['bold']), colored(diag.k, attrs=['bold']), "START " + s))
                res = function(*args, **kwargs)
                #log.info("{} {}|{} {}".format(name, colored(diag.k, attrs=['bold']), colored(diag.k, attrs=['bold']), "END " + s))
                return res
            return info_func
        return info_deco

    #@info()
    def initialize(self, t0, m0, x0, h0, initResources, uc0=None, ud0=None):
        """ Initialize the Diagnoser

        >>> diag = Diagnoser(test.simple_model())
        >>> diag.initialize(0, None, [0., 0.], [8000., 5000.], 100)
        >>> len(list(diag.configurations()))
        8
        >>> len(list(diag.particles()))
        300
        >>> len(list(diag.hybrid_tokens()))
        800
        """
        self.k = t0
        self.uck = uc0
        if ud0 is not None:
            self.observedEvents.update((ud0, self.k) for e in ud0)

        if m0 is not None:
            mode = self.get_mode_by_name(m0)
            p = Possibility.create(mode, t0, x0, h0, initResources)
            marking = list(p.tokens())
        else:
            poss = []
            pns = {}
            marking = []
            if x0 == []:
                x0 = [0]
            if h0 == []:
                h0 = [0]
            for mode in self.modes():
                ps = mode.ps
                pn = mode.pn
                ph = mode.ph
                c = Configuration(place = ps)
                p = Possibility(c)
                marking.append(c)
                if pn not in pns:
                    pns[pn] = [Particle(np.array(x0), place = pn) for _ in range(initResources)]
                    marking += pns[pn]

                hybridTokens = [HybridToken(np.array(h0), [(mode, t0)], c, pi, place = ph) for pi in pns[pn]]
                marking += hybridTokens
                poss.append(p)
        super().initialize(marking, t0)

    def distribute(self,x, pn):
        #TODO
        noise = pn.ssr.stateNoiseEquation.random() 
        for xi, n in zip(x, noise):
            print(xi, n)
        return x + noise 

    def add_transition(self, inputs, transition, outputs):
        """ Overwrite to associate events from symbolic conditions with arcs
        Events are extracted from symbolic condition names
        Do not add HybridPlace in the Transition

        >>> diag = test.simple_model()
        >>> np.all([a.events == set() for t in diag.transition() for _,a in t.output()])
        True
        >>> diag = test.simple_diagnoser()
        >>> np.all([a.events == [] for t in diag.transition() for _,a in t.output()])
        False
        >>> next(a for p, a in diag.transition('t6').output() if type(p) == SymbolicPlace).events
        {'start'}
        """
        if len(inputs) > 1 :
            ip = [p for p in inputs if type(p) != HybridPlace]
            op = [p for p in outputs if type(p) != HybridPlace]
            super().add_transition(ip, transition, op)
            if transition.cs != true and transition.cs != false:
                events = {transition.cs.event}
                for p,a in transition.output():
                    if type(p) == SymbolicPlace:
                        a.events = events.copy()
        else:
            # special case for merged hybrid transitions
            ip = [p for p in inputs if type(p) == HybridPlace]
            op = [p for p in outputs if type(p) == HybridPlace]
            super().add_transition(ip, transition, op)

    #@info()
    def remove_symbolic_and_hybrid_conditions(self):
        """ Transform symbolic conditions of all transition in true conditions
        and hybrid conditions in false conditions

        >>> diag = test.simple_model()
        >>> np.all([t.cs == true for t in diag.transition()])
        False
        >>> np.all([t.ch == false for t in diag.transition()])
        False
        >>> diag = test.simple_diagnoser()
        >>> np.all([t.cs == true for t in diag.transition()])
        True
        >>> np.all([t.ch == false for t in diag.transition()])
        True
        """
        for t in self.transition():
            t.cs = true
            t.ch = false
            if t.cn == false:
                t.cn = true

    def are_mergeable(self, t1, t2):
        """ Return True if the two given Transition are mergeable, false otherwise
        Two Transition are mergeable if they have the same symbolic and numerical condition,
        the same set of input places, and the same NumericalPlace (input and output)

        >>> diag = Diagnoser(test.simple_model(), postTreatement = False)
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t3'))
        False
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t1'))
        False
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t0'))
        False
        >>> diag.are_mergeable(diag.transition('t7'), diag.transition('t1'))
        False
        >>> diag.are_mergeable(diag.transition('t4'), diag.transition('t5'))
        False
        >>> diag.remove_symbolic_and_hybrid_conditions()
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t3'))
        False
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t1'))
        False
        >>> diag.are_mergeable(diag.transition('t2'), diag.transition('t0'))
        False
        >>> diag.are_mergeable(diag.transition('t3'), diag.transition('t0'))
        True
        >>> diag.are_mergeable(diag.transition('t7'), diag.transition('t1'))
        True
        >>> diag.are_mergeable(diag.transition('t4'), diag.transition('t5'))
        False
        """
        areMergeable = False
        if t1.cn == t2.cn:
            if t1.cs == t2.cs:
                if set(p for p,_ in t1.input()) == set(p for p,_ in t2.input()):
                    pn1 = set(p for p,_ in t1.output() if type(p) == NumericalPlace)
                    pn2 = set(p for p,_ in t2.output() if type(p) == NumericalPlace)
                    ph1 = set(p for p,_ in t1.output() if type(p) == HybridPlace)
                    ph2 = set(p for p,_ in t2.output() if type(p) == HybridPlace)
                    if pn1 == pn2 and ph1 == ph2:
                        areMergeable = True
        return areMergeable

    def find_mergeable_transitions(self, transition):
        """ Find all Transition that are mergeable with the given Transition and return the set
        The resulting set will at least have one Transition (the given one) if the given Transition is in the Diagnoser 

        >>> diag = Diagnoser(test.simple_model(), postTreatement=False)
        >>> diag.remove_symbolic_and_hybrid_conditions()
        >>> g = diag.find_mergeable_transitions(diag.transition('t7'))
        >>> sorted([t.name for t in g])
        ['t1', 't7']
        >>> g = diag.find_mergeable_transitions(diag.transition('t2'))
        >>> sorted([t.name for t in g])
        ['t2']
        >>> g = diag.find_mergeable_transitions(diag.transition('t0'))
        >>> sorted([t.name for t in g])
        ['t0', 't3']
        >>> g = diag.find_mergeable_transitions(diag.transition('t3'))
        >>> sorted([t.name for t in g])
        ['t0', 't3']
        >>> g = diag.find_mergeable_transitions(diag.transition('t4'))
        >>> sorted([t.name for t in g])
        ['t4']
        """
        return {t for t in self.transition() if self.are_mergeable(transition, t)}

    def merge_transition_group(self, transitions):
        """ Merge all the given mergeable transitions into a single one

        >>> diag = Diagnoser(test.simple_model(), postTreatement=False)
        >>> sorted([t.name for t in diag.transition()])
        ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7']
        >>> diag.remove_symbolic_and_hybrid_conditions()
        >>> diag.merge_transition_group([diag.transition('t1'), diag.transition('t7')])
        >>> sorted([t.name for t in diag.transition()])
        ['t0', 't1-t7', 't2', 't3', 't4', 't5', 't6']
        >>> t = diag.transition('t1-t7')
        >>> sorted([sorted(a.events) for p,a in t.output() if type(p) == SymbolicPlace])
        [['f1'], ['stop']]
        """
        newTransition = list(transitions)[0]
        inputs = {p for p,_ in newTransition.input()}
        outputs = {p for t in transitions for p,_ in t.output()}
        outputEvents = {}
        for t in transitions:
            p, arc = next(((p, a) for p,a in t.output() if type(p) == SymbolicPlace), (None, None))
            if p is not None:
                if p in outputEvents:
                    outputEvents[p] |= arc.events
                else:
                    outputEvents[p] = arc.events
            self.remove_transition(t.name)
        newTransition.name = '-'.join(name for name in sorted([t.name for t in transitions]))
        self.add_transition(inputs, newTransition, outputs)
        # update arc events
        for p, a in newTransition.output():
            if p in outputEvents:
                a.events = outputEvents[p]
        

    #@info()
    def merge_transitions(self) :
        """ Merge all the  Transition in the Diagnoser that model the same change of continuous dynamics into one single Transition
        Mergeable SpecialTransition have the same numerical condition, the same set of input places, and the same NumericalPlace
        The resulting Transition has as ouput places the jonction of all the sets of output places of the merged Transition

        >>> diag = Diagnoser(test.simple_model(), postTreatement=False)
        >>> sorted([p.name for p in diag.transition()])
        ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7']
        >>> diag.remove_symbolic_and_hybrid_conditions()
        >>> diag.merge_transitions()
        >>> sorted([p.name for p in diag.transition()])
        ['t0-t3', 't1-t7', 't2', 't4', 't5', 't6']
        >>> sorted([p.name for p,a in diag.transition('t1-t7').output()])
        ['KO1', 'OFF', 'STOPPED']
        >>> sorted([p.name for p,a in diag.transition('t0-t3').output()])
        ['KO2', 'OK2', 'ON2']
        """
        transitions = {t for t in self.transition()}
        trNb = len(transitions)

        while transitions != set():
            t = transitions.pop()
            group = self.find_mergeable_transitions(t)

            #assert len(group) >= 1

            if len(group) > 1:
                # merge
                names = sorted([t.name for t in group])
                self.merge_transition_group(group)
                #log.info("  {} are merged in {}".format(" and ".join(n for n in names), '-'.join(n for n in names)))
        
            transitions -= group

        trNb2 = len(list(self.transition())) - trNb
        #log.info('  {} transitions have been removed ({}% gain)'.format(trNb2, trNb2/trNb))

    def dupplicate(self, p, n):
        """ Create special possibility copies for accept()

        >>> diag = test.simple_diagnoser()
        >>> p = list(diag.place('OK1').tokens)[0].possibility
        >>> poss = diag.dupplicate(p, 5)
        >>> l = list(poss[0].particles())
        >>> len(poss)
        5
        >>> np.all([list(p.particles()) == l for p in poss])
        True
        """
        configurations = [p.configuration().copy() for _ in range(n)]
        
        for h in p.hybrid_tokens():
            piCopy = h.particle.copy()
            for cCopy in configurations:
                h.__class__(h.h, h.modes, cCopy, piCopy, h.place)
        return [p.__class__(c, p.id) for c in configurations]

    def dupplicate_and_delete(self, p, n):
        """ Create special possibility copies for accept()
        Include the given Possibility in the copies

        Warning -> this is tricky -> read the code

        >>> diag = test.simple_diagnoser()
        >>> p = list(diag.place('OK1').tokens)[0].possibility
        >>> poss = diag.dupplicate_and_delete(p, 5)
        >>> l = list(poss[0].particles())
        >>> len(poss)
        5
        >>> np.all([list(p.particles()) == l for p in poss])
        True
        """
        # check if several configurations are linked to set of particle of p
        nominal = len(next(p.hybrid_tokens()).particle.hybridTokens) == 1

        if nominal:
            # nominal case
            # the set of particle is only use by p
            # -> all tokens are move and the configuration is dupplicate for the n-1 symbolic places
            configurations = [p.configuration().copy() for _ in range(n-1)]
            
            for h in p.hybrid_tokens():
                for cCopy in configurations:
                    h.__class__(h.h, h.modes, cCopy, h.particle, h.place)
            
            # add the initial configuration to move it with copies
            return [p.__class__(c, p.id) for c in configurations] + [p]
        else:
            # case where the set of particle of p is use by others possibilities
            # -> the set of particles is copied but the configuration is moved
            configurations = [p.configuration().copy() for _ in range(n-1)]

            # add the initial configuration to associated it with the new set
            configurations.append(p.configuration())

            hybridTokens = list(p.hybrid_tokens()) # save the hybrid tokens of p
            for h in hybridTokens:
                piCopy = h.particle.copy()
                h.configuration = None # break the link
                for cCopy in configurations:
                    h.__class__(h.h, h.modes, cCopy, piCopy, h.place)
            
            return [p.__class__(c, p.id) for c in configurations]

    def is_enabled(self, t):
        """ Return True if the given Transition is enabled
        Transition is enabled if at leat one Possibility in
        its input places is accepted

        >>> diag = test.simple_diagnoser()
        >>> poss = next(diag.possibilities())
        >>> sum(1 for t in diag.transition() if diag.is_enabled(t))
        3
        """
        # get input mode
        sip, sia = next((p,a) for p,a in t.input() if type(p) == SymbolicPlace)
        nip, nia = next((p,a) for p,a in t.input() if type(p) == NumericalPlace)
        input_mode = self.get_mode_with_places(sip, nip)

        possibilities = (c.possibility for c in sip if type(c) == Configuration and c.possibility.mode() == input_mode)
        return any(self.is_possibility_accepted(t, poss) for poss in possibilities)

    def accept(self, t):
        """ Select Tokens that have to be move throught the given Transition during the firing
        Firing is pseudo-firing
        This function admit that the transition is enabled

        >>> diag = test.simple_diagnoser()
        >>> diag.accept(diag.transition('t0-t3'))
        >>> len(diag.place('OK1').tokens)
        3
        >>> len(diag.place('ON1').tokens)
        101
        >>> sum(1 for g in diag.place('OK1').tokens if type(g) == GroupToken and g.tag == 'OK1_OK2 ' + GroupToken.accepted_tag)
        1
        >>> gt = next(g for g in diag.place('OK1').tokens if type(g) == GroupToken and g.tag == 'OK1_OK2 ' + GroupToken.accepted_tag)
        >>> list(gt.tokens[0].events)
        [('e1', 0)]
        >>> sum(1 for g in diag.place('ON1').tokens if type(g) == GroupToken and g.tag == 'ON1_ON2 ' + GroupToken.accepted_tag)
        1
        """

        sip, sia = next((p,a) for p,a in t.input() if type(p) == SymbolicPlace)
        nip, nia = next((p,a) for p,a in t.input() if type(p) == NumericalPlace)
        input_mode = self.get_mode_with_places(sip, nip)
        nv = next(v for v in nia.vars())

        events = {v:set.union(*[ao.events for po,ao in t.output() if v in ao.vars()]) for v in sia.vars()}

        confToGroup = {v: [] for v in sia.vars()}
        partToGroup = {nv: []}

        possibilities = (c.possibility for c in sip if isinstance(c, Configuration) and (c.possibility.mode() == input_mode) and self.is_possibility_accepted(t, c.possibility))

        for poss in possibilities:

            # TODO: try without
            #if self.is_numerical_transition(t) and all(t.cn(pi) for pi in poss.particles()):
            #    # trust the model
            #    # sure firing / pseudo firing
            #    newPossibilities = self.dupplicate_and_delete(poss, len(events))

            #else:
            #    # unsure firing / pseudo firing
            #    newPossibilities = self.dupplicate(poss, len(events))
            newPossibilities = self.dupplicate(poss, len(events))

            partToGroup[nv] += list(newPossibilities[0].particles())

            # add event to configurations
            for v, poss in zip(events, newPossibilities):
                poss.configuration().update_events(events[v], self.k)
                confToGroup[v].append(poss.configuration())

        sip.group(confToGroup)
        nip.group(partToGroup)

    #@info()
    def fire(self):
        """ Fire transitions in the HPPN-diagnoser

        >>> diag = test.simple_diagnoser()
        >>> sorted(t.name for t in diag.firing_order())
        ['t0-t3', 't5', 't6']
        >>> for pi in diag.place('ON2').tokens: pi.x[0] = 25
        >>> sorted(t.name for t in diag.firing_order())
        ['t0-t3', 't4', 't5', 't6']
        >>> diag.k = 1
        >>> diag.fire()
        >>> len(diag.place('OK1').tokens)
        2
        >>> len(diag.place('OK2').tokens)
        2
        >>> len(diag.place('KO1').tokens)
        2
        >>> len(diag.place('KO2').tokens)
        2
        >>> len(diag.place('KO12').tokens)
        3
        >>> len(diag.place('READY').tokens)
        1
        >>> len(diag.place('STOPPED').tokens)
        1
        >>> len(diag.place('ON1').tokens)
        200
        >>> len(diag.place('ON2').tokens)
        200
        >>> len(diag.place('OFF').tokens)
        300
        """
        super().fire() 
        # fire hybrid tokens
        for h in self.hybrid_tokens():
            mode = self.get_mode_with_places(h.configuration.place, h.particle.place)
            place = mode.ph
            if place:
                h.place = place
                if len(h.modes) == 0 or h.modes[-1][0] != mode:
                    h.modes.append((mode, self.k))
            else:
                input('error to investigate' + mode)
        #assert(sum(len(list(p.hybrid_tokens())) for p in self.possibilities()) == sum(1 for _ in self.hybrid_tokens()))

    #@info()
    def predict(self, dt, uc):
        """ Perform the prediction step of the HPPN-diagnoser online process
        The prediction step is composed of two steps:
            - the firing of all the fire-enabled transition
            - the update of all the Particle x
            - the update of all the HybridToken h
            - incrementing the t by the given time

        >>> diag = test.simple_diagnoser()
        >>> diag.predict(1, np.array([4,4,4]))
        """
        assert(dt > 0)
        self.fire()     
        self.update_particle_values(dt, uc)
    #@info()
    def update_particle_weights(self, dt, uc, yc):
        """ Update Particle weights

        >>> diag = test.simple_diagnoser()
        >>> np.sum(pi.weight for pi in diag.particles())
        0
        >>> diag.update_particle_weights(2, np.array([4,4,4]), np.array([5,5]))
        >>> np.sum(pi.weight for pi in diag.particles()) > 0
        True
        """
        pis = list(self.particles())
        Diagnoser.api_update_particle_weights(pis, self.k, dt, uc, yc)
    @optimization(50, 'weight')
    def api_update_particle_weights(particles, k, dt, uc, yc):
        """ Update the weight value of all the Particle in the Diagnoser

        >>> diag = test.simple_diagnoser()
        >>> np.sum(pi.weight for pi in diag.particles())
        0
        >>> Diagnoser.api_update_particle_weights(list(diag.particles()), 0, 2, np.array([4,4,4]), np.array([5,5]))
        >>> np.sum(pi.weight for pi in diag.particles()) > 0
        True
        """
        #log.debug("  updating {} particle weights...".format(len(particles)))

        return [Diagnoser.api_update_particle_weight(pi, k, dt, uc, yc) for pi in particles]
 
    def api_update_particle_weight(pi, k, dt, uc, yc):
        """ Update the weight of the given Particle, computed with the weight fonction associated to the NumericalPlace it belong

        >>> diag = test.simple_diagnoser()
        >>> pi = next(diag.particles())
        >>> w1 = pi.weight
        >>> w2 = Diagnoser.api_update_particle_weight(pi, 0, 100, np.array([4, 4, 4]), np.array([5, 5]))
        >>> w1 != w2
        True
        """
        
        pi.weight = pi.place.ssr.weight(yc, pi.x, dt, uc, k)
        
        #print("après pi.weight")
        #pi.error = pi.place.ssr.outputNoiseEquation.error(diff)
        #if pi.weight == 0:
        #    for i in range(60,61):
        #        print(eyc[i], yc[i])
        #return pi (?)
        return pi.weight

    #@info()
    def normalize_particle_weights(self):
        """ Normalize Particle weights relatively to the cluster of Particle they belong
        DEPRECIATED

        >>> diag = test.simple_diagnoser()
        >>> for pi in diag.particles(): pi.weight = 1
        >>> diag.normalize_particle_weights()
        >>> np.sum(pi.weight for pi in diag.particles()) # should be 3 but approximation
        2.99999999999998
        """
        # global weight
        s = np.sum(pi.weight for pi in self.particles())
        if s != 0:
            for pi in self.particles(): pi.globalWeight = pi.weight/s
        else:
            for pi in self.particles(): pi.globalWeight = 0

        assert(sum([pi.globalWeight for pi in self.particles()])<1.1)

        # local weight
        for cluster in self.particle_clusters():
            s = np.sum(pi.weight for pi in cluster)
            if s != 0:
                for pi in cluster: pi.weight /= s
            else:
                for pi in cluster: pi.weight = 0

    #@info()
    def update_configuration_weights(self, do):
        """ Update Configuration weights with their values and the set of observed event until now

        >>> diag = test.simple_diagnoser()
        >>> np.sum(c.weight for c in diag.configurations())
        0
        >>> diag.update_configuration_weights({'e1'})
        >>> np.sum(c.weight for c in diag.configurations()) > 0
        True
        """
        confs = list(self.configurations())
        Diagnoser.api_update_configuration_weights(confs, self.symbolic_weight_function, self.observedEvents)
        #print('')
        #print(self.observedEvents)
        #for c in self.configurations():
        #    print('{} : {}'.format(c.weight, c.events))
        #input('tot')

        # the rest is DEPRECIATED

        # compute weights
        # old way
        #distances = {}
        #for c in self.configurations():
        #    diff =  do ^ set(c.events)
        #    distO = 0
        #    distUO = 0
        #    for e in diff:
        #        # check if event is observable
        #        if e[0] in self.o:
        #            distO += 1
        #        else:
        #            distUO += .5
        #    distances[c] = np.sqrt(distO**2 + distUO**2)
        #
        #closest = min(distances.values())
        #for c, dist in distances.items():
        #    # compute the weight relative to the closest event sequence
        #    dist -= closest
        #    c.weight = 1/np.exp(dist)
        #    #c.weight = 1/np.exp(2 * dist)
        #    #c.weight = gaussian(dist, 0, .5)

        #for c in self.configurations():
        #    # project configuration event set to observable event set
        #    proj = set(e for e in c.events if e[0] in self.o)

        #    # Hamming distance
        #    dist =  len(self.observedEvents ^ proj)

        #    # compute weight with the model specification
        #    c.weight = gaussian(dist, 0, 1)
    
    @optimization(100, 'weight')
    def api_update_configuration_weights(configurations, weight_function, do):
        """ Update the weight value of all the Configuration in the Diagnoser

        >>> diag = test.simple_diagnoser()
        >>> np.sum(c.weight for c in diag.configurations())
        0
        >>> Diagnoser.api_update_configuration_weights(list(diag.configurations()), diag.symbolic_weight_function, {'e1'})
        >>> np.sum(c.weight for c in diag.configurations()) > 0
        True
        """
        #log.debug("  updating {} configuration weights...".format(len(configurations)))
        return [Diagnoser.api_update_configuration_weight(weight_function, c, do) for c in configurations]

    def api_update_configuration_weight(weight_function, c, do):
        """ Update the weight of the given Configuration, computed with the symbolic weight function associated to the model

        >>> diag = test.simple_diagnoser()
        >>> c = next(diag.configurations())
        >>> w1 = c.weight
        >>> w2 = Diagnoser.api_update_configuration_weight(diag.symbolic_weight_function, c, {'e1'})
        >>> w1 != w2
        True
        """
        c.weight = weight_function(c.events, do)
        return c.weight

    def compute_possibility_scores(self):
        """ Compute the Possibility scores with a weighted function between the weight of their Configuration and the weight of their particle cluster

            - configuration weights are normalized
              with the sum of all configuration weights
            - a particle cluster weight is the average of 
              all the weights of the particles in the cluster
            - particle cluster weights are normalized
              with the sum of all particle cluster weights
            - the weighted function uses the globalConfidenceInSymbolicPart coefficient

        >>> diag = test.simple_diagnoser()
        >>> diag.compute_possibility_scores()
        """
        # configuration weights are normalized with the sum of all configuration weights
        symScores = {}
        s = sum(c.weight for c in self.configurations())
        if s != 0:
            symScores = {c: c.weight/s for c in self.configurations()}
        else:
            symScores = {c: 0 for c in self.configurations()}

        # a particle cluster weight is the average of all the weights of the particles in the cluster
        numScores = {}

        # compute particle cluster weights
        #numScores = {cl: np.sum(pi.weight for pi in cl) for cl in self.particle_clusters()}
        numScores = {cl: np.average([pi.weight for pi in cl]) for cl in self.particle_clusters()}

        # normalized particle cluster weigth with all particle cluster weights
        s = sum(numScores.values())
        if s != 0:
            numScores = {cl: sc/s for cl,sc in numScores.items()}
        else:
            numScores = {cl: 0 for cl in numScores}

        #assert(len(list(symScores)) == len(list(self.possibilities())))
        for p in self.possibilities():
            sym = symScores[p.configuration()]
            p._particleCluster = frozenset(p.particles())
            num = numScores[p._particleCluster]
            # debug
            if sym < 0 or sym > 1.01:
                print('sym', sym)
            #assert(sym>= 0 and sym <= 1.1)
            if num < 0 or num > 1.01:
                print('num', num)
                print('nb', len(list(p.particles())), len(list(self.particles())))
                print('nbweight', len({pi.globalWeight for pi in p.particles()}), len({pi.globalWeight for pi in self.particles()}))
                print('min', min([pi.globalWeight for pi in p.particles()]), min([pi.globalWeight for pi in self.particles()]))
                print('max', max([pi.globalWeight for pi in p.particles()]), max([pi.globalWeight for pi in self.particles()]))
                print('weight', sum(pi.globalWeight for pi in p.particles()), sum([pi.globalWeight for pi in self.particles()]))
            
            #assert(num>= 0 and num <= 1.1)

            p._symbolicBelief = sym
            p._numericalBelief = num
            # compute score
            p._belief = alpha_weighted_function(sym, num, self.globalConfidenceInSymbolicPart)
        
    def compute_particle_cluster_scores(self):
        """ Compute the particle cluster scores
        The score of a cluster is the maximum belief of the Possibility the cluster is part of

        >>> diag = test.simple_diagnoser()
        >>> for c in diag.configurations(): c.weight = 1
        >>> for pi in diag.particles(): pi.weight = 1
        >>> s = diag.compute_particle_cluster_scores()
        """
        self.compute_possibility_scores()
        scores = {}
        for p in self.possibilities():
            cl = p._particleCluster
            if np.sum(pi.weight for pi in cl) > 0 : # FIXME: what's the use ?
                if cl not in scores:
                    scores[cl] = p._belief
                else:
                    if p._belief > scores[cl]:
                        scores[cl] = p._belief
            else:
                scores[cl] = 0
        return scores

    def assign_particle_number(self, scores, method) :
        """ Assign a new particle number to each particle cluster depending 
        on its scores given as a dict
        
        >>> diag = test.simple_diagnoser()
        >>> diag.assign_particle_number({1: 1}, 'stochastic')
        ({1: 100}, False)
        """
        if method == 'stochastic':
            return stochastic_resource_assignation(scores, self.minResources, self.suffResources, self.maxResources)
        elif method == 'sure':
            return sure_resource_assignation(scores, self.minResources, self.suffResources, self.maxResources), False
        elif method == 'weird':
            return weird_resource_assignation(scores, self.minResources, self.suffResources, self.maxResources), False
        else :
            raise Exception('method {} does not exist'.format(method))

    def dupplicate_particle(self, pi):
        piCopy = pi.copy()
        # dupplicate hybrid tokens
        htCopies = [HybridToken(ht.h, ht.modes, ht.configuration, piCopy, ht.place) for ht in pi.hybridTokens if ht.particle is not None and ht.configuration is not None]
        # FIXME: if h.particle is not None useful ??
        return piCopy

    #@info()
    def resample(self):
        """ Resample the Possibility of the net

        >>> diag = test.simple_diagnoser()
        >>> try: 
        ...     diag.resample()
        ... except Exception as e:
        ...     str(e) == colored(' NO PARTICLE MATCHES ', 'red', attrs=['bold', 'reverse'])
        True
        >>> sum(1 for _ in diag.tokens()) # get 1108 instead of 0 cause it exit()
        1108
        """
        # compute particle cluster scores
        scores = self.compute_particle_cluster_scores()
        #print([(len(cl), next((pi for pi in cl)).place.name, s) for cl,s in scores.items()])

        #log.info("  {}/{} clusters of particle are improblable".format(np.sum(1 for cl, s in scores.items() if s == 0), len(scores)))

        # remove improbable cluster FIXME: only useful for debug ?
        scores = {cl: s for cl, s in scores.items() if s > 0}

        # assign a new particle number for each cluster
        piAssignation, matchButNotEnought = self.assign_particle_number(scores, 'stochastic')

        debug = sorted([r for r in piAssignation.values()])
        debug = it.groupby(debug)
        debug = {k: len(list(p)) for k, p in debug}
        #log.info('  particle assignation (particle nb: cluster nb): {}'.format(debug))
        #log.info('  remaining {} particles'.format(self.maxResources - sum(nb for nb in piAssignation.values())))

        #log.info("  {}/{} clusters of particle got 0 particle assigned".format(np.sum(1 for cl, nb in piAssignation.items() if nb == 0), len(piAssignation)))

        # remove improbable cluster FIXME: only useful for debug ?
        piAssignation = {cl: nb for cl, nb in piAssignation.items() if nb > 0}

        # recovery 
        if len(piAssignation) == 0:
            if matchButNotEnought:
                #raise Exception(colored(' PARTICLE MATCH BUT TOO FEW ', 'red', attrs=['bold', 'reverse']))
                print(colored(' PARTICLE MATCH BUT TOO FEW ', 'red', attrs=['bold', 'reverse']))
                # assigne a new particle number for each cluster
                piAssignation, matchButNotEnought = self.assign_particle_number(scores, 'sure')
                # remove improbable cluster
                piAssignation = {cl: nb for cl, nb in piAssignation.items() if nb > 0}

            else:
                raise ParticleMatchException(colored(' NO PARTICLE MATCHES ', 'red', attrs=['bold', 'reverse']))

        ## SIR resampling
        newParticles = []

        for cl, nb in piAssignation.items():
            particles = list(cl)
            weights = [pi.weight for pi in particles]
            distribution = WeightedDistribution(particles, weights)
            for _ in range(nb):
                newParticle = distribution.pick()
                newParticle = self.dupplicate_particle(newParticle)
                newParticles.append(newParticle)

                #log.debug("new resampled particle {}".format(newParticles[-1]))

        # replace old particles by new in the net
        for old in self.particles():
            old.unlink()
        for pn in self.numerical_places():
            pn.empty()
        for pi in newParticles:
            pi.place.add_tokens([pi])
        
        # TODO: same with hybrid tokens ?

        # configuration assignation
        # find configuration that lost numerical support
        toDel = [p.configuration() for p in self.possibilities() if next(p.particles(),None) is None]

        # remove configurations
        #log.info("  {} configurations removed in {}".format(len(toDel), ", ".join(n for n in set(c.place.name for c in toDel))))
        for c in toDel:
            c.unlink() # TODO: useful ? yes while 1 c == 1 poss
            c.place.remove_tokens([c])
  
        # clean hybrid token list USEFUL for conf that are not copied nor deleted
        for c in self.configurations():
            c.hybridTokens[:] = [ht for ht in c.hybridTokens if ht.configuration is not None and ht.particle is not None]
    
    #@info()
    def correct(self, dt, uc, do, yc):
        # FIXME comments
        """ Perform the correction step of the HPPN-diagnoser online process
        The correction step is composed of:
         - a numerical correction
         - a symbolic correction
         - a hybrid correction

        >>> diag = test.simple_diagnoser()
        >>> diag.correct(1, np.array([4, 4, 4]), {'e1'}, np.array([26, 14]))
        """
        
        self.update_particle_weights(dt, uc, yc)
        
        self.update_configuration_weights(do)
        

        # record net events after to not consider current discrete observation
        self.observedEvents.update((e, self.k) for e in do)
        #print("before try")
        try:
            
            self.resample()
            
            
        except ParticleMatchException as e:
            print("On rentre dans l'except")
            # debug: draw cause of the non match
            import equation as eq
            ids = []
            posss = self.possibilities()
            #posss = (p for p in posss if len(p.modes()) == 1 and p.mode().name == 'Sensor BL FL fault')
            for p in posss:
                particles = list(p.particles())
                xs = [pi.x for pi in particles]
                eycs = [pi.place.ssr.output(pi.x, dt, uc, self.k) for pi in particles]
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
                #ids.append('{}, +{}, *{}, +{}, *{}\n{}\n{}\n{}'.format(p.modes(), weightAdd, weightMul, [i for i, d in enumerate(diffAdd) if d == 0], [i for i, d in enumerate(diffMul) if d == 0], '\n'.join(['{}: {}, {}, {}, {}, {}'.format(i, ey0, ey, y, s, d) for i,(d, ey0, ey, s, y) in enumerate(zip(diffAve, eycs[0], eycAve, sigmas[0], yc))]), np.multiply.reduce(diffAve), '\n'.join(['{}: {}, {}'.format(i, x0, x) for i,(x0, x) in enumerate(zip(xs[0], xAve))])))
                ids.append('{} ({}, {}, {}), +{}, *{}, +{}, *{}'.format(p.modes(), round(p._symbolicBelief,2), round(p._numericalBelief,2), round(p._belief,2), weightAdd, weightMul, [i for i, d in enumerate(diffAdd) if d == 0], [i for i, d in enumerate(diffMul) if d == 0]))
            raise Exception(str(e) + ' because \n{}'.format('\n'.join(i for i in ids)))
                

        # update hybrid tokens after resampling to minimize calculation
        # it uses x_k : BAD -> should used x_k-1 but it minimize calculation
        # yet because hybrid_token are probabilities I don't care about a one sampling period gap
        # The wright solution is to save x_k-1 values in Particle objects 
        self.update_hybrid_token_values(dt)
        

    #@info()
    def update(self, t, uc, ud, yc, yd):
        """ Update the HPPN-diagnoser according to the symbolic and numerical informations
        It consists in:
            - predicting the future state of the Diagnoser with the given command during the given time
            - correcting the Diagnoser according to numerical and symbolic observations

        >>> diag = test.simple_diagnoser()
        >>> diag.update(1, np.array([4,4,4]), {'e1'}, np.array([26, 14]), set())
        """   
        
        dt = t - self.k
        #print(dt)
        # use old commands
        oldUc = self.uck
        
        if oldUc is None : oldUc = uc
        '''TEST =list(self.particles())
        print("----------Avant predict------------")
        print(vars(TEST[0])) 
        print("OldUc avant : " +str(oldUc))
        print("dt avant: " + str(dt))'''
        self.predict(dt, oldUc)
        '''TEST =list(self.particles())
        print("----------Apres predict------------")
        print(vars(TEST[0])) 
        print("oldUc après : " + str(oldUc))
        print("dt après: " + str(dt))'''
        self.k += dt
        # update old commands
        self.uck = uc
        #self.log_marking()
        #l = [pi.x[30] for pi in self.particles() if pi.x[30] >.5]
        #print()
        #print(len(l), round(min(l),2), round(max(l),2))
        self.correct(dt, uc, ud | yd, yc)
        #l = [pi.x[30] for pi in self.particles() if pi.x[30] >.5]
        #try: 
        #    print('  ', len(l), round(min(l),2), round(max(l),2))
        #except:
        #    pass
        #self.log_marking()

    def get_transition_output_mode(self, t):
        """ Return the output mode of the given Transition

        >>> diag = test.simple_diagnoser()
        >>> sorted(diag.get_transition_output_mode(diag.transition('t0-t3')))
        [KO2-ON2, OK2-ON2]
        >>> sorted(diag.get_transition_output_mode(diag.transition('t1-t7')))
        [KO1-OFF, STOPPED-OFF]
        >>> diag.get_transition_output_mode(diag.transition('t2'))
        [KO1-ON2]
        """
        nip = next(p for p,_ in t.output() if type(p) == NumericalPlace)
        sip = (p for p,_ in t.output() if type(p) == SymbolicPlace)
        return [self.get_mode_with_places(ps, nip) for ps in sip]

    def get_start_modes(self):
        """ Return the list of end modes of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(m.name for m in hppn.get_start_modes())
        ['READY-OFF']
        """
        return [m for m in self.modes() if all(m not in self.get_transition_output_mode(t) for t in self.transition())]

    def mode_belief(self):
        """ Return current mode belief of the Diagnoser
        The belief of a Mode is the best belief of the Possibility
        that are in that Mode
        
        >>> diag = test.simple_diagnoser()
        >>> diag.compute_possibility_scores()
        >>> sorted(diag.mode_belief().items())
        [('KO1-OFF', 0.0), ('KO1-ON2', 0.0), ('KO12-OFF', 0.0), ('KO2-ON2', 0.0), ('OK1-ON1', 0.0), ('OK2-ON2', 0.0), ('READY-OFF', 0.0), ('STOPPED-OFF', 0.0)]
        """
        keyfunc = lambda poss: poss.mode().name
        posss = sorted(self.possibilities(), key=keyfunc)
        groups = it.groupby(posss, key=keyfunc)
        return {m: max(p._belief for p in g) for m,g in groups}

    def possibility_to_diagnosis(self, p, dt, uc):
        """ Transform a Possibility into Diagnosis 
        dt/uc needed to compute output
        
        >>> diag = test.simple_diagnoser()
        >>> diag.compute_possibility_scores()
        >>> d = diag.possibility_to_diagnosis(next(diag.possibilities()), 1, None)
        """
        Diagnosis = type('Diagnosis', (), {})
        # create a diagnosis
        d = Diagnosis()

        # ids
        d.id = p.id
        d.fatherId = p.fatherId

        # modes
        d.modes = p.modes().copy()

        # mode
        d.mode = p.mode()

        # events
        d.events = p.configuration().events.copy()
        
        # beliefs
        d.symbolicBelief = p._symbolicBelief
        d.numericalBelief = p._numericalBelief
        d.belief = p._belief

        # continuous state
        xl = [pi.x for pi in p.particles()]
        d.xmean = np.mean(xl, axis=0)
        d.xmin = np.min(xl, axis=0)
        d.xmax = np.max(xl, axis=0)

        # hybrid state
        hl = [pi.h for pi in p.hybrid_tokens()]
        d.hmean = np.mean(hl, axis=0)
        d.hmin = np.min(hl, axis=0)
        d.hmax = np.max(hl, axis=0)

        # output
        if uc is not None:
            ycl = [pi.place.ssr.output(pi.x, dt, uc, self.k) for pi in p.particles()]
            d.ycmean = np.mean(ycl, axis=0)
            d.ycmin = np.min(ycl, axis=0)
            d.ycmax = np.max(ycl, axis=0)
        else:
            d.ycmean = None
            d.ycmin = None
            d.ycmax = None

        # token numbers
        d.cn = 1 #sum(1 for p in posss for _ in p.configurations())
        d.pn = sum(1 for _ in p.particles())
        d.hn = sum(1 for _ in p.hybrid_tokens())
        d.pon = 1 #len(posss)

        return d

    def merge_diagnosis(self, diagnosis):
        """ Merge the given Diagnosis
        DEPRECIATED
        
        >>> diag = test.simple_diagnoser()
        >>> diag.compute_possibility_scores()
        >>> diags = [diag.possibility_to_diagnosis(p, 1, None) for p in diag.possibilities()]
        >>> len(diag.merge_diagnosis(diags))
        8
        """
        Diagnosis = type('Diagnosis', (), {})
        def modeKey(d):
            return tuple(m[0] for m in d.modes)

        clusters = sorted(diagnosis, key=modeKey)
        clusters = it.groupby(clusters, key=modeKey)
        diagnosis = []
        for mode, diags in clusters:
            diags = list(diags)

            # create a diagnosis
            d = Diagnosis()

            # modes
            d.modes = []
            for i, m in enumerate(mode):
                mini = min(d.modes[i][1] for d in diags)
                maxi = max(d.modes[i][1] for d in diags)
                d.modes.append((m, (mini,maxi)))

            # mode
            d.mode = d.modes[-1][0]

            # events
            d.events = []
            events = diags[0].events
            events = tuple(e[0] for e in events)
            for i, e in enumerate(events):
                mini = min(d.events[i][1] for d in diags)
                maxi = max(d.events[i][1] for d in diags)
                d.events.append((e, (mini,maxi)))
            
            # belief
            d.belief = diags[0].belief

            # continuous state
            d.xmean = np.mean([d.xmean for d in diags], axis=0)
            d.xmin = np.min([d.xmin for d in diags], axis=0)
            d.xmax = np.max([d.xmax for d in diags], axis=0)

            # hybrid state
            d.hmean = np.mean([d.hmean for d in diags], axis=0)
            d.hmin = np.min([d.hmin for d in diags], axis=0)
            d.hmax = np.max([d.hmax for d in diags], axis=0)

            # output
            if diags[0].ycmean is not None:
                d.ycmean = np.mean([d.ycmean for d in diags], axis=0)
                d.ycmin = np.min([d.ycmin for d in diags], axis=0)
                d.ycmax = np.max([d.ycmax for d in diags], axis=0)
            else:
                d.ycmean = None
                d.ycmin = None
                d.ycmax = None

            # token numbers
            d.cn = sum(d.cn for d in diags)
            d.pn = sum(d.pn for d in diags)
            d.hn = sum(d.hn for d in diags)
            d.pon = sum(d.pon for d in diags)

            diagnosis.append(d)

        return diagnosis

    def diagnosis(self, dt, uc):
        """ Return the distribution of diagnosis
        computed by the Diagnoser
        dt/uc needed to compute output

        >>> diag = test.simple_diagnoser()
        >>> diag.compute_possibility_scores()
        >>> len(list(diag.diagnosis(1, None)))
        8
        """
        diags = (self.possibility_to_diagnosis(p, dt, uc) for p in self.possibilities())
        #keyfunc = lambda d: d.belief
        #diags = sorted(diags, reverse=True, key=keyfunc)
        #if merge:
        #    groups = it.groupby(diags, key=keyfunc)
        #    diags = []
        #    for b, g in groups:
        #        if b >= minimumBelief :
        #            diags += self.merge_diagnosis(list(g))
        return diags

    def log_marking(self):

        super().log_marking()
        
        #log.info('    diag events: {}'.format(self.events))
        #self.compute_possibility_scores()
        mode_belief = self.mode_belief()
        #log.info('  mode beliefs: {}'.format(mode_belief))

        #log.info('  belief:')
        belief = self.diagnosis(1, None, 0, False)
        #for d in belief: log.info('    {}, {}'.format(d.belief, d.modes))

if __name__ == "__main__":
    import doctest
    import test
    logging.disable(logging.INFO)
    doctest.testmod()
