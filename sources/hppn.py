
""" This module contains a library to manipulate Hybrid Particle Petri Nets
"""

from apn import SpecialToken, SpecialPlace, SpecialTransition, Expression, PetriNet, Variable, MultiArc, GroupToken
from equation import StateSpaceRepresentation, true, false, discrete_gaussian
from optimization import optimization
#import parallel2
import itertools as it
import log as logging
import numpy as np
from termcolor import colored
from functools import wraps

# log and file handler
log = logging.getLogger(__name__[:4])

class HPPNExpression(Expression):
    """ Extension of Expression to keep easylly access event sequences
    """
    def __init__(self, exp, events = set()):
        super().__init__(exp)
        self.events = events.copy()

class Configuration(SpecialToken):
    """ This class represents symbolic tokens in HPPN
    """

    def __init__(self, events = set(), place = None):
        """ Create a Configuration 
        Be careful, give the place attribute do not add the token to the place
        
        >>> c = Configuration()
        >>> c.events == set()
        True
        """
        super().__init__(place = place)
        self.events = set(events)
        self.hybridTokens = []
        self.weight = 0
        self.possibility = None

    def update_events(self, events, t):
        """ update the events vector values of the Configuration with the given list of events
        Each events in the list are set to True
        
        >>> c = Configuration([('e1',3)])
        >>> c.update_events(('e1', 'e2'), 13)
        >>> sorted(c.events)
        [('e1', 3), ('e1', 13), ('e2', 13)]
        
        """
        self.events.update((e, t) for e in events)

       
    def unlink(self):
        """ Destroy links between the Configuration and its hybridTokens
        Then remove the HybridToken from their HybridPlace

        >>> c = Configuration()
        >>> pi = Particle(np.array(3.))
        >>> ht = HybridToken(np.array(0.), [], c, pi)
        >>> hp = HybridPlace('foo', StateSpaceRepresentation())
        >>> hp.add_tokens([ht])
        >>> ht in c.hybridTokens and ht in pi.hybridTokens and ht.configuration == c and ht.particle == pi and ht in hp
        True
        >>> c.unlink()
        >>> ht not in c.hybridTokens and ht in pi.hybridTokens
        True
        >>> ht.configuration == c or ht.particle == pi
        False
        >>> ht in hp
        True
        """
        for h in self.hybridTokens:
            h.configuration = None
            h.particle = None
            h.place = None
            #if h.place is not None:
            #    h.place.remove_tokens(h)
        self.hybridTokens = []

    #def copy_for_accept(self):
    #    # TODO: to remove when accept() totally rewriten
    #    """ Copy a Configuration and duplicate all the HybridToken linked with the Configuration
    #    and link them to the copy
    #    
    #    >>> c1 = Configuration([('e1',1)])
    #    >>> pi1 = Particle(np.array(3.))
    #    >>> pi2 = Particle(np.array(5.))
    #    >>> h1 = HybridToken(np.array(0.), [], c1, pi1, None)
    #    >>> h2 = HybridToken(np.array(1.), [], c1, pi2, None)
    #    >>> c2 = c1.copy_for_accept()
    #    >>> type(c2) == Configuration
    #    True
    #    >>> l1 = {h for h in c1.hybridTokens}
    #    >>> l2 = {h for h in c2.hybridTokens}
    #    >>> l1 & l2 == set()
    #    True
    #    >>> len(l1) == len(l2)
    #    True
    #    >>> all([any([h.particle in [ht.particle for ht in l2]]) for h in l1])
    #    True
    #    """
    #    copy = self.copy()
    #    copy.hybridTokens = [h.copy_for_configuration(copy) for h in self.hybridTokens if h.configuration is not None]
    #    return copy

    def copy(self):
        """ Copy the Configuration
        Create a new Configuration object and return it
        DO NOT CONSERVE LINKS WITH HYBRID TOKENS

        >>> c1 = Configuration([('e1', 1)])
        >>> c1.place = SpecialPlace('foo')
        >>> c2 = c1.copy()
        >>> c2 != c1
        True
        >>> c2.place == c1.place
        True
        >>> list(c2.events)
        [('e1', 1)]
        >>> c1.events == c2.events
        True
        >>> id(c2.events) == id(c1.events)
        False
        """
        c = self.__class__(self.events, self.place)
        c.weight = self.weight
        return c

    def __repr__(self):
        """ Return the string representation of a Configuration

        >>> r = repr(Configuration([('e1',1)]))
        """
        return "({}, {})".format(id(self), self.events)

class Particle (SpecialToken):
    """ This class represents numerical tokens in HPPN
    """

    def __init__(self, x, place = None):
        """ Create a Particle with the given initial value 
        Be careful, give the place attribute do not add the token to the place

        >>> pi = Particle(np.array([0, 1, 0.]))
        >>> np.all(pi.x == [0, 1, 0])
        True
        >>> pi2 = Particle(np.array('[0 1 0.]'))
        >>> pi == pi2
        False
        """
        super().__init__(place = place)
        self.x = x.copy() # estimation value
        self.weight = 0
        self.globalWeight = 0
        self.hybridTokens = []

    def __repr__(self):
        """ Return the string representation of a Particle

        >>> r = repr(Particle(np.array([0, 1, 0.])))
        """
        return "({}, {})".format(id(self), self.x)

    def unlink(self):
        """ Destroy links between the Particle and its hybridTokens
        Then remove the HybridToken from their HybridPlace

        >>> c = Configuration([('e1',0)])
        >>> pi = Particle(np.array(3.))
        >>> h = HybridToken(np.array(0.), [], c, pi)
        >>> hp = HybridPlace('foo', StateSpaceRepresentation())
        >>> hp.add_tokens([h])
        >>> h in c.hybridTokens and h in pi.hybridTokens and h.configuration == c and h.particle == pi
        True
        >>> h in hp
        True
        >>> pi.unlink()
        >>> h in c.hybridTokens and h not in pi.hybridTokens 
        True
        >>> h.configuration == c or h.particle == pi
        False
        >>> h in hp
        True
        """
        for h in self.hybridTokens:
            h.particle = None
            h.configuration = None
            h.place = None
            #if h.place is not None:
            #    h.place.remove_tokens(h)
        self.hybridTokens = []

    #def copy_for_resampling(self):
    #    """ Special copy of the Particle
    #    Needed during the resampling step of the particle filter
    #    The copy has the same creationTime than the origin
    #    The copy has the same place than the origin

    #    >>> pi = Particle(np.array('[0 1 0.]'), creationTime = 4)
    #    >>> pi2 = pi.copy_for_resampling(5)
    #    >>> pi2.creationTime == 5
    #    True
    #    """
    #    new = self.copy_for_accept()
    #    return new

    #def copy_with_hybrid_tokens(self):
    #    """ Copy a Particle and duplicate all the HybridToken linked with the Particle 
    #    and link them to the copy
    #    
    #    >>> c1 = Configuration([('e1', 0)])
    #    >>> c2 = Configuration()
    #    >>> pi1 = Particle(np.array(3.))
    #    >>> h1 = HybridToken(np.array(0.), [], c1, pi1, None)
    #    >>> h2 = HybridToken(np.array(1.), [], c2, pi1, None)
    #    >>> pi2 = pi1.copy_for_accept()
    #    >>> type(pi2) == Particle
    #    True
    #    >>> l1 = {h for h in pi1.hybridTokens}
    #    >>> l2 = {h for h in pi2.hybridTokens}
    #    >>> l1 & l2 == set()
    #    True
    #    >>> len(l1) == len(l2)
    #    True
    #    >>> all([any([h.configuration in [ht.configuration for ht in l2]]) for h in l1])
    #    True
    #    """
    #    copy = self.copy()
    #    copy.hybridTokens = [h.copy_for_particle(copy) for h in self.hybridTokens if h.particle is not None]
    #    return copy

    def copy(self):
        """ Copy the Particle 
        Create a new Particle object and return it
        DO NOT CONSERVE LINKS WITH HYBRID TOKENS

        >>> pi1 = Particle(np.array([0, 1, 0.]))
        >>> pi2 = pi1.copy()
        >>> pi1 == pi2
        False
        >>> np.all(pi1.x == pi2.x)
        True
        >>> id(pi1.x) == id(pi2.x)
        False
        """
        copy = self.__class__(self.x, self.place)
        copy.weight = self.weight
        copy.globalWeight = self.globalWeight
        return copy

class HybridToken (SpecialToken):
    """ This class represents hybrid tokens in HPPN
    """

    def __init__(self, h, modes, configuration, particle, place = None):
        """ Create a HybridToken with the given initial value and Configuration and Particle that the HybridToken follows
        Be careful, give the place attribute do not add the token to the place

        >>> c = Configuration()
        >>> pi = Particle(np.array(5))
        >>> ht = HybridToken(np.array([0, 1, 0.]), [], c, pi)
        >>> ht.configuration == c and ht in c.hybridTokens
        True
        >>> ht.particle and ht in pi.hybridTokens
        True
        """
        super().__init__(place = place)
        self.configuration = configuration
        self.particle = particle
        self.h = h.copy() # hybrid value
        self.place = place
        self.modes = modes.copy()

        # update configuration and particle links
        self.particle.hybridTokens.append(self)
        self.configuration.hybridTokens.append(self)

    def __repr__(self):
        """ Return the string representation of a Particle

        >>> c = Configuration()
        >>> pi = Particle(np.array(5))
        >>> r = repr(HybridToken(np.array([0, 1, 0.]), [], c, pi))
        """
        return "({}, [{}, {}], {})".format(id(self), id(self.configuration), id(self.particle), self.h)
    
    #def copy_for_configuration(self, configuration):
    #    """ Copy the HybridToken
    #    Particle link: the copy is linked with the Particle linked with the original HybridToken
    #    Configuration link: the copy is linked with the given Configuration
    #    
    #    >>> c1 = Configuration([('e1', 0)])
    #    >>> c2 = Configuration()
    #    >>> pi1 = Particle(3.)
    #    >>> h1 = HybridToken(np.array(5.), [], c1, pi1)
    #    >>> h2 = h1.copy_for_configuration(c2)
    #    >>> h2.particle == pi1
    #    True
    #    >>> h2.configuration == c2
    #    True
    #    >>> id(h1.h) == id(h2.h)
    #    False
    #    """
    #    copy = self.__class__(self.h.copy(), self.modes.copy(), configuration, self.particle, self.place)
    #    #if self.place:
    #    #    self.place.add_tokens([copy])
    #    return copy
    #    
    #def copy_for_particle(self, particle):
    #    """ Copy the HybridToken
    #    Configuration link: the copy is linked with the Configuration linked with the original HybridToken
    #    Particle link: the copy is linked with the given Particle
    #    
    #    >>> c1 = Configuration([('e1', 0)])
    #    >>> pi1 = Particle(3.)
    #    >>> pi2 = Particle(3.)
    #    >>> h1 = HybridToken(np.array(5.), [], c1, pi1)
    #    >>> h2 = h1.copy_for_particle(pi2)
    #    >>> h2.particle == pi2
    #    True
    #    >>> h2.configuration == c1
    #    True
    #    >>> id(h1.h) == id(h2.h)
    #    False
    #    """
    #    copy = self.__class__(self.h.copy(), self.modes.copy(), self.configuration, particle, self.place)
    #    #if self.place:
    #    #    self.place.add_tokens([copy])
    #    return copy

    def copy(self):
        """ Copy the HybridToken 
        Create a new HybridToken object and return it
        DO NOT CONSERVE LINKS WITH PARTICLE AND CONFIGURATION

        >>> h = HybridToken(np.array([0, 1, 0.]), [], Configuration(), Particle(np.array(0)))
        >>> h2 = h.copy()
        >>> id(h.h) == id(h2.h)
        False
        >>> h.configuration == h2.configuration
        True
        >>> h.particle == h2.particle
        True
        >>> h in h.configuration.hybridTokens and h2 in h.configuration.hybridTokens
        True
        >>> h in h.particle.hybridTokens and h2 in h.particle.hybridTokens
        True
        """
        return __class__(self.h, self.modes, self.configuration, self.particle, self.place)

class SymbolicPlace(SpecialPlace):
    """ Class representing symbolic places in HPPN
    """

    def __init__(self, name):
        """ Create a SymbolicPlace with the given name

        >>> p1 = SymbolicPlace('foo')
        >>> p2 = SymbolicPlace('foo')
        >>> p1 == p2
        False
        """
        SpecialPlace.__init__(self, name)

    def copy(self):
        """ Copy the SymbolicPlace
        
        >>> p1 = SymbolicPlace('foo')
        >>> p2 = p1.copy() 
        >>> p1 == p2
        False
        """
        copy = self.__class__(self.name)
        if hasattr(self, 'color'): copy.color = self.color
        return copy
        
class NumericalPlace(SpecialPlace):
    """ Class representing numerical places in HPPN
    """

    def __init__(self, name, stateSpaceRepresentation):
        """ Create a NumericalPlace with the given name
        Given dynamic and output equations as well a process noises and sense 
        noise are associated to the NumericalPlace

        >>> ssr = StateSpaceRepresentation() 
        >>> p1 = NumericalPlace('foo', ssr)
        >>> p2 = NumericalPlace('foo', ssr)
        >>> p1 == p2
        False
        >>> p1.ssr == p2.ssr
        False
        """
        SpecialPlace.__init__(self, name)
        self.ssr = stateSpaceRepresentation.copy()
      
    def copy(self):
        """ Copy the NumericalPlace

        >>> ssr = StateSpaceRepresentation() 
        >>> p1 = NumericalPlace('foo', ssr)
        >>> p2 = p1.copy()
        >>> p1 == p2
        False
        >>> p1.ssr == p2.ssr
        False
        """
        new = self.__class__(self.name, self.ssr)
        return new

class HybridPlace(SpecialPlace):
    """ Class representing hybrid places in HPPN
    """

    def __init__(self, name, stateSpaceRepresentation):
        """ Create a HybridPlace with the given StateSpaceRepresentation

        >>> ssr = StateSpaceRepresentation() 
        >>> p1 = HybridPlace('foo', ssr)
        >>> p2 = HybridPlace('foo', ssr)
        >>> p1 == p2
        False
        >>> p1.ssr == p2.ssr
        False
        """
        self.ssr = stateSpaceRepresentation.copy()
        SpecialPlace.__init__(self, name)

    def copy(self):
        """ Copy the HybridPlace

        >>> ssr = StateSpaceRepresentation() 
        >>> p1 = HybridPlace('foo', StateSpaceRepresentation())
        >>> p2 = p1.copy() 
        >>> p1 == p2
        False
        >>> p1.ssr == p2.ssr
        False
        """
        new = self.__class__(self.name, self.ssr)
        return new

class Transition(SpecialTransition):
    """ This class represents a transition in HPPN
    """

    def __init__(self, name, symCond, numCond, hybCond):
        """ Create a Transition with the given name and conditions

        >>> t = Transition('foo', true, true, false)
        """
        super().__init__(name)
        if not symCond:
            symCond = true
        if not numCond:
            numCond = true
        if not hybCond:
            hybCond = false
        self.cs = symCond
        self.cn = numCond
        self.ch = hybCond

                    
    def copy(self):
        """ Copy the Transition

        >>> t = Transition('t', None, None, None)
        >>> c = t.copy()
        """
        return self.__class__(self.name, self.cs, self.cn, self.ch)
    
class Possibility:
    """ This class allows to manipulate easily the tokens of the HPPN.
    This class represents a possibility in an HPPN.
    A possibility is a cluster of a Configuration and its associated
    Particle and HybridToken.
    Basically, a possibility is a pointer to the Configuration.
    """
    
    def create(mode, time, x, h, particleNumber):
        """ Create a Possibility and all the tokens with the given initial
        parameters

        >>> ps = SymbolicPlace('sp')
        >>> pn = NumericalPlace('np', StateSpaceRepresentation())
        >>> ph = HybridPlace('hp', StateSpaceRepresentation())
        >>> m = Mode('foo', ps, pn ,ph)
        >>> p = Possibility.create(m, 0, [8000,5000], [0], 10)
        >>> v = [pi.x for pi in p.particles()]
        >>> np.all(v == np.array([8000, 5000]))
        True
        >>> len(v)
        10
        >>> v = [h.h for h in p.hybrid_tokens()]
        >>> len(v)
        10
        >>> np.all(v == np.array([0]))
        True
        """
        ps = mode.ps
        pn = mode.pn
        ph = mode.ph
        c = Configuration(place = ps)
        if x == []:
            x = [0]
        particles = [Particle(np.array(x), place = pn) for _ in range(particleNumber)]
        if h == []:
            h = [0]
        hybridTokens = [HybridToken(np.array(h), [(mode, time)], c, pi, place = ph) for pi in particles]
        p = Possibility(c)
        return p

    # iterator returning the ID of a new Possibility
    new_id = it.count()

    def __init__(self, configuration, fatherId = -1):
        """ Initialize a Possibility by creating the link to the Configuration
        Updates the Configuration "possibility" attribute.
        
        >>> c = Configuration()
        >>> p = Possibility(c)
        >>> p.id == 0
        True
        >>> p = Possibility(c)
        >>> p.id == 1
        True
        """
        self.id = next(Possibility.new_id)
        self.fatherId = fatherId
        self._configuration = configuration
        self._configuration.possibility = self
        self._belief = 0
        self._symbolicBelief = 0
        self._numericalBelief = 0

    def configuration(self):
        """ Return the Configuration of the Possibility
        
        >>> c = Configuration()
        >>> p = Possibility(c)
        >>> c == p.configuration()
        True
        """
        return self._configuration

    def particles(self):
        """ Return a generator over the Particle of the Possibility

        >>> c = Configuration()
        >>> particles = [Particle(np.array([0., 0.])) for i in range(100)]
        >>> hybridTokens = [HybridToken(np.array([8000,5000]), [], c, pi) for pi in particles]
        >>> p = Possibility(c)
        >>> list(p.particles()) == particles
        True
        """

        return (h.particle for h in self.hybrid_tokens()) 

    def hybrid_tokens(self):
        """ Return a generator over the HybridToken of the Possibility

        >>> c = Configuration()
        >>> particles = [Particle(np.array([0., 0.])) for i in range(100)]
        >>> hybridTokens = [HybridToken(np.array([8000,5000]), [], c, pi) for pi in particles]
        >>> p = Possibility(c)
        >>> list(p.hybrid_tokens()) == hybridTokens
        True
        """
        return (h for h in self._configuration.hybridTokens if h.configuration is not None and h.particle is not None)

    def tokens(self):
        """ Return a generator over all the tokens of the Possibility

        >>> c = Configuration()
        >>> particles = [Particle(np.array([0., 0.])) for i in range(100)]
        >>> hybridTokens = [HybridToken(np.array([8000,5000]), [], c, pi) for pi in particles]
        >>> p = Possibility(c)
        >>> list(p.tokens()) == [c] + particles + hybridTokens
        True
        """
        return it.chain([self.configuration()], self.particles(), self.hybrid_tokens())

    def mode(self):
        """ Return the current mode of the Possibility.
        Basically, it returns the last mode of the mode sequence of its HybridToken.

        >>> p = Possibility.create(Mode('foo', None, None, None) , 0, [0., 0.], [8000,5000], 10)
        >>> p.mode()
        foo
        """
        return next(self.hybrid_tokens()).modes[-1][0]

    def modes(self):
        """ Return the mode sequence of the Possibility.
        Basically, it returns the mode sequence of its HybridToken.

        >>> p = Possibility.create(Mode('foo', None, None, None), 0, [0., 0.], [8000,5000], 10)
        >>> p.modes()
        [(foo, 0)]
        """
        return next(self.hybrid_tokens()).modes

    def copy(self):
        cCopy = self.configuration().copy()
        for h in self.hybrid_tokens():
            pCopy = h.particle.copy()
            hCopy = h.copy(cCopy, pCopy)
        copy = self.__class__(cCopy)
        copy._belief = self._belief
        copy._symbolicBelief = self._symbolicBelief
        copy._numericalBelief = self._numericalBelief
        return copy

    def get_mode_sequences(possibilities):
        clusters = {}
        for poss in possibilities:
            # debug
            l = [h.modes for h in poss.hybrid_tokens()]
            #assert(l.count(l[0]) == len(l))

            histo = next(poss.hybrid_tokens()).modes
            hist = tuple(m[0] for m in histo)
            if not hist in clusters:
                clusters[hist] = [poss]
            else:
                clusters[hist].append(poss)
        seqs = []
        for hist, posss in clusters.items():
            seq = []
            for i, h in enumerate(hist):
                mini = min(poss.modes()[i][1] for poss in posss)
                maxi = max(poss.modes()[i][1] for poss in posss)
                #time = "{}-{}".format(mini,maxi)
                #if mini == maxi:
                #    time = str(mini)
                seq.append((h, (mini,maxi)))
            seqs.append(seq)
        return seqs

class HPPN(PetriNet):
    """ This class represent a HPPN object
    """

    def __init__(self, name):
        """ Creates an empty HPPN

        >>> hppn = HPPN('foo')
        """
        super().__init__(name)
        self.k = 0
        self.modesTable = {}
        self.stateSpace = []
        self.udo = set() # observable command event label set
        self.ydo = set() # observable sensor event label set
        self.o = set() # observable event label set
        self.uo = set() # unobservable event label set
        self.symbolic_weight_function = discrete_gaussian # function that computes configuration weights
 
    def info(string = None):
        """ Logging decorator generator
    
        >>> deco = HPPN.info()
        """
        def info_deco(function):
            """ Logging decorator
            """
            if string is None:
                s = function.__name__
            else:
                s = string
            @wraps(function)
            def info_func(*args, **kwargs):
                hppn = args[0]
                name = hppn.name
                name = name[:7]
                #log.info('{} {} {}'.format(name, colored(hppn.k, attrs=['bold']), "START " + s))
                res = function(*args, **kwargs)
                #log.info("{} {} {}".format(name, colored(hppn.k, attrs=['bold']), "END " + s))
                return res
            return info_func
        return info_deco
       
    def fill(self, hppn):
        """ Fills the current HPPN with a copy of the given HPPN
        The HPPN names remain different

        >>> hppn = HPPN('foo')
        >>> hppn.add_place(SymbolicPlace('sp'))
        >>> hppn.add_place(NumericalPlace('np', StateSpaceRepresentation()))
        >>> hp = HybridPlace('hp', StateSpaceRepresentation())
        >>> hppn.add_place(hp)
        >>> hppn.add_mode('fun', 'sp', 'np', 'hp')
        >>> hppn2 = HPPN('foo22')
        >>> hppn2.fill(hppn)
        >>> hppn == hppn2
        False
        >>> hppn.name == hppn2.name
        False
        >>> ids = {id(hppn)}
        >>> ids2 = {id(hppn2)}
        >>> ids |= {id(p) for p in hppn.place()}
        >>> ids2 |= {id(p) for p in hppn2.place()}
        >>> len(ids) == len(ids2)
        True
        >>> ids |= {id(t) for t in hppn.transition()}
        >>> ids2 |= {id(t) for t in hppn2.transition()}
        >>> len(ids) == len(ids2)
        True
        >>> for p in hppn.place(): ids |= {id(t) for t in p}
        >>> for p in hppn2.place(): ids2 |= {id(t) for t in p}
        >>> len(ids) == len(ids2)
        True
        >>> ids & ids2 == set()
        True
        >>> len(hppn.modesTable) == len(hppn2.modesTable)
        True
        """
        self.k = hppn.k

        self.udo = hppn.udo.copy()
        self.ydo = hppn.ydo.copy()
        self.o = hppn.o.copy()
        self.uo = hppn.uo.copy()
        self.symbolic_weight_function = hppn.symbolic_weight_function
        
        self.add_places([p.copy() for p in hppn.place()])
        for t in hppn.transition():
            inputs = [p for p,a in t.input()]
            outputs = [p for p,a in t.output()]

            # ok cause add_transition only use place.name
            self.add_transition(inputs, t.copy(), outputs)
        
        self.modesTable = {}
        for (ps, pn), m in hppn.modesTable.items():
            ph = m.ph
            self.add_mode(m.name, ps.name, pn.name, ph.name)

    def add_places(self, places):
        """ Add given places in the HPPN
        
        >>> hppn = HPPN('foo')
        >>> places = [SymbolicPlace("{}".format(i)) for i in range(5)]
        >>> hppn.add_places(places)
        >>> all((hppn.has_place(p.name) for p in places))
        True
        """
        for p in places: self.add_place(p)

    def initialize(self, initialMarking, t0):
        """ Initialize the marking of the HPPN with the given list of SpecialToken (Configuration, Particle and HybridToken)
        A SpecialToken is add to the SpecialPlace specified in its place attribut.
        SpecialToken in the list should already be linked between them
        
        >>> hppn = test.simple_model()
        >>> sum((len(p.tokens) for p in hppn.place()))
        201
        >>> from hppn import * # avoid type error
        >>> c = Configuration(place = hppn.place('OK2'))
        >>> particles = [Particle(np.array([0., 0.]), place = hppn.place('ON2')) for i in range(1000)]
        >>> hybridTokens = [HybridToken(np.array([8000,5000]), [], c, pi, place = hppn.place('OK2_ON2')) for pi in particles]
        >>> initialMarking = [c] + particles + hybridTokens
        >>> hppn.initialize(initialMarking, 10)
        >>> hppn.k
        10
        >>> len(list(hppn.possibilities()))
        1
        >>> sum((len(p.tokens) for p in hppn.place()))
        2001
        >>> sum((len(p.tokens) for p in hppn.place() if type(p) == SymbolicPlace))
        1
        >>> sum((len(p.tokens) for p in hppn.place() if type(p) == NumericalPlace))
        1000
        >>> c in hppn.place('OK2')
        True
        >>> len(hppn.place('ON2').tokens)
        1000
        >>> len(hppn.place('OK2_ON2').tokens)
        1000
        """
        self.empty()
        self.k = t0

        # add tokens to places
        for t in initialMarking : 
            if t.place:
                #if type(t) != HybridToken:
                self.place(t.place.name).add_tokens([t])
                #else:
                #    t.place = self.place(t.place.name)
                if type(t) == Configuration:
                    Possibility(t)
        #log.info("{} initialized with {} configurations, {} particles and {} hybrid tokens".format(self.name, len(list(self.configurations())), len(list(self.particles())), len(list(self.particles()))))
        #log.info('  continuous state number: {}'.format(np.average([len(pi.x) for pi in self.particles()])))
        #log.info('  hybrid state number: {}'.format(np.average([len(h.h) for h in self.hybrid_tokens()])))
        #for p in self.place():
            #log.debug('{} initialize with {} tokens'.format(p.name, len(p.tokens)))

    def _gen_input_vars_text(place, places):
        """ Generate the variables liste in an HPPN.

        >>> spi1 = SymbolicPlace('spi1')
        >>> spi1.add_tokens([Configuration()])
        >>> spo1 = SymbolicPlace('spo1')
        >>> spo2 = SymbolicPlace('spo2')
        >>> res = set(['spi1_spo1', 'spi1_spo2'])
        >>> set(HPPN._gen_input_vars_text(spi1, [spo1, spo2])) == res
        True
        """
        return ["{}_{}".format(place.varname, p.varname) for p in places]

    def _gen_output_vars_text(place, places):
        """ Generate the variables liste in an HPPN.

        >>> spi1 = SymbolicPlace('spi1')
        >>> spi1.add_tokens([Configuration()])
        >>> spi2 = SymbolicPlace('spi2')
        >>> spi2.add_tokens([Configuration()])
        >>> spo1 = SymbolicPlace('spo1')
        >>> res = set(['spi1_spo1', 'spi2_spo1'])
        >>> set(HPPN._gen_output_vars_text(spo1, [spi1, spi2])) == res
        True
        """
        return ["{}_{}".format(p.varname, place.varname) for p in places]

    def _add_type_transition(self, places1, transition, places2, varType, gen_func, add_func):
        """
        """
        for i in places1:
            vars = [varType(v) for v in gen_func(i, places2)]
            if len(vars) == 1:
                annotation = list(vars)[0]
            else:
                annotation = MultiArc(vars)
            add_func(i.name, transition.name, annotation)

    def _add_transition(self, inputs, transition, outputs, placeType):
        """
        """
        type_inputs = [p for p in inputs if type(p) == placeType]
        type_outputs = [p for p in outputs if type(p) == placeType]
        self._add_type_transition(type_inputs, transition, type_outputs, Variable, HPPN._gen_input_vars_text, self.add_input)
        self._add_type_transition(type_outputs, transition, type_inputs, HPPNExpression, HPPN._gen_output_vars_text, self.add_output)

    def add_transition(self, inputs, transition, outputs):
        """ Add a transition to the HPPN
        
        >>> hppn = HPPN('foo')
        >>> spi1 = SymbolicPlace('spi1')
        >>> spi1.add_tokens([Configuration()])
        >>> spi2 = SymbolicPlace('spi2')
        >>> spi2.add_tokens([Configuration()])
        >>> npi = NumericalPlace('npi', StateSpaceRepresentation())
        >>> npi.add_tokens([Particle(np.array(i)) for i in range(5)])
        >>> spo1 = SymbolicPlace('spo1')
        >>> spo2 = SymbolicPlace('spo2')
        >>> npo1 = NumericalPlace('npo1', StateSpaceRepresentation())
        >>> npo2 = NumericalPlace('npo2', StateSpaceRepresentation())
        >>> hppn.add_places([spi1, spi2, npi, spo1, spo2, npo1, npo2])
        >>> t = Transition('t', lambda c: c.events[-1][0] == 'e1', lambda pi: pi.x > 1, None)
        >>> hppn.add_transition([spi1, spi2, npi], t, [spo1, spo2, npo1, npo2])
        >>> all((len(a.vars()) == 2 for p, a in t.input()))
        True
        >>> all((len(a.vars()) == 2 for p, a in t.input() if type(p) == NumericalPlace))
        True
        >>> all((len(a.vars()) == 1 for p, a in t.output() if type(p) == NumericalPlace))
        True
        """
        super().add_transition(transition)
        self._add_transition(inputs, transition, outputs, SymbolicPlace)
        self._add_transition(inputs, transition, outputs, NumericalPlace)
        self._add_transition(inputs, transition, outputs, HybridPlace)
        transition._createConditionExpression()

    
    def symbolic_places(self):
        """ Return a generator object that browse all the SymbolicPlace of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(t.name for t in hppn.symbolic_places())
        ['KO1', 'KO12', 'KO2', 'OK1', 'OK2', 'READY', 'STOPPED']
        """
        return (p for p in self.place() if type(p) == SymbolicPlace)

    def configurations(self):
        """ Return a generator object that browse all the Configuration of the HPPN

        >>> hppn = test.simple_model()
        >>> len(list(hppn.configurations()))
        1
        >>> hppn.empty()
        >>> len(list(hppn.configurations()))
        0
        """
        return (c for p in self.symbolic_places() for c in p.tokens if type(c) == Configuration)
    
    def numerical_places(self):
        """ Return a generator object that browse all the NumericalPlace of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(t.name for t in hppn.numerical_places())
        ['OFF', 'ON1', 'ON2']
        """
        return (p for p in self.place() if type(p) == NumericalPlace)

    def particles(self):
        """ Return a generator object that browse all the Particle of the HPPN

        >>> hppn = test.simple_model()
        >>> len(list(hppn.particles()))
        100
        >>> hppn.empty()
        >>> len(list(hppn.particles()))
        0
        """

        return (pi for p in self.numerical_places() for pi in p.tokens if type(pi) == Particle)
    
    def hybrid_places(self):
        """ Return a generator object that browse all the HybridPlace of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(t.name for t in hppn.hybrid_places())
        ['KO12_OFF', 'KO1_OFF', 'KO1_ON2', 'KO2_ON2', 'OK1_ON1', 'OK2_ON2', 'READY_OFF', 'STOPPED_OFF']
        """
        return (p for p in self.place() if type(p) == HybridPlace)

    def hybrid_tokens(self):
        """ Return a generator object that browse all the HybridToken of the HPPN

        >>> hppn = test.simple_model()
        >>> len(list(hppn.hybrid_tokens()))
        100
        >>> hppn.empty()
        >>> len(list(hppn.hybrid_tokens()))
        0
        """
        return (h for c in self.configurations() for h in c.hybridTokens if h.configuration is not None and h.particle is not None)

    def tokens(self):
        """ Return a generator object that browse all the SpecialToken in the HPPN

        >>> hppn = test.simple_model()
        >>> len(list(hppn.tokens()))
        201
        >>> hppn.empty()
        >>> len(list(hppn.tokens()))
        0
        """
        return it.chain(self.configurations(), self.particles(), self.hybrid_tokens())

    def possibilities(self):
        """ Return a generator object that browse all the Possibility monitored in the HPPN

        >>> hppn = test.simple_model()
        >>> len(list(hppn.possibilities()))
        1
        >>> hppn.empty()
        >>> len(list(hppn.possibilities()))
        0
        """
        return (c.possibility for c in self.configurations())

    def particle_clusters(self):
        """ Compute the set of all particle cluster of the HPPN

        >>> hppn = test.simple_model()
        >>> len(hppn.particle_clusters())
        1
        """
        return set(frozenset(p.particles()) for p in self.possibilities())
       
    #@info()
    def update_particle_values(self, dt, uc):
        """ Update the particle of the HPPN

        >>> hppn = test.simple_model()
        >>> hppn.update_particle_values(3, np.array([1, 1, 1]))
        """
        pis = list(self.particles())
        
        #log.info('  updating {} particle values during {}...'.format(len(pis), dt))
        api_update_particle_values(pis, self.k, dt, uc)
        #log.info('  {} particle values updated'.format(len(pis)))
        

    #@info()
    def update_hybrid_token_values(self, dt):
        """ Update the hybridTokens of the HPPN

        >>> hppn = test.simple_model()
        >>> hppn.update_hybrid_token_values(3)
        """
        hts = list(self.hybrid_tokens())
        #log.info('  updating {} hybrid token values...'.format(len(hts)))
        HPPN.api_update_hybrid_token_values(hts, dt)
        #log.info('  {} hybrid token values updated'.format(len(hts)))

    @optimization(50, 'h')
    def api_update_hybrid_token_values(hybridTokens, dt):
        """ Update the h of all the given HybridToken
        
        >>> hppn = test.simple_model()
        >>> sum1 = np.sum(np.sum(ht.h) for ht in hppn.hybrid_tokens())
        >>> HPPN.api_update_hybrid_token_values(list(hppn.hybrid_tokens()), 5000)
        >>> len(list(hppn.hybrid_tokens())) == 100
        True
        >>> sum2 = np.sum(np.sum(ht.h) for ht in hppn.hybrid_tokens())
        >>> sum1 != sum2
        True
        """
        #log.debug("  updating {} hybrid tokens values...".format(len(hybridTokens)))
        return [HPPN.api_update_hybrid_token_value(ht, dt) for ht in hybridTokens]

    def api_update_hybrid_token_value(ht, dt):
        """ Update the h of all the given HybridToken according the the state equation associated to the HybridPlace they belong
        
        >>> hppn = test.simple_model()
        >>> ht = next(hppn.hybrid_tokens())
        >>> v = HPPN.api_update_hybrid_token_value(ht, 5000)
        >>> np.all(v != [0, 0])
        True
        
        """
        ht.h = ht.place.ssr.state(ht.particle.x, ht.configuration.events, ht.h, ht.modes, dt) 
        return ht.h

    def update_hybrid_token_modes(self):
        """ Update the HybridToken mode sequences according
        to the Mode they belong

        >>> hppn = test.simple_model()
        >>> hppn.update_hybrid_token_modes()
        >>> all(len(ht.modes) == 1 for ht in hppn.hybrid_tokens())
        True
        >>> next(hppn.configurations()).update_events({'e1'},0)
        >>> hppn.fire()
        >>> hppn.update_hybrid_token_modes()
        >>> all(ht.modes[-1][0].name == 'OK2-ON2' for ht in hppn.hybrid_tokens())
        True
        """
        for ht in self.hybrid_tokens():
            mode = self.get_mode_with_places(ht.configuration.place, ht.particle.place)
            if len(ht.modes) == 0 or ht.modes[-1][0] != mode:
                ht.modes.append((mode, self.k))

    def is_possibility_accepted(self, t, poss):
        """ Return True if the given Possibility satisfied the given Transition conditions
        Assume the Possibility is in the same mode than the Transition input mode

        >>> hppn = test.simple_model()
        >>> poss = next(hppn.possibilities())
        >>> hppn.is_possibility_accepted(hppn.transition('t0'), poss)
        False
        >>> poss.configuration().update_events({'e1'},0)
        >>> hppn.is_possibility_accepted(hppn.transition('t0'), poss)
        True
        >>> hppn.is_possibility_accepted(hppn.transition('t2'), poss)
        False
        >>> for ht in poss.hybrid_tokens(): ht.h[0] = .9
        >>> hppn.is_possibility_accepted(hppn.transition('t2'), poss)
        True
        """
        # test behavioral conditions
        sym = t.cs(poss.configuration())
        num = any(1 for pi in poss.particles() if t.cn(pi))
        accepted = sym and num
        if not accepted:
            # test hybrid condition
            accepted = any(1 for ht in poss.hybrid_tokens() if t.ch(ht))
            if not accepted:
                log = 'test error'
            else:
                log = 'hyb statisfied'
        else:
            log = 'beh statisfied'
        return accepted

    def is_enabled(self, t):
        """ Return True if the given Transition is enabled
        Transition is enabled if at leat one Possibility in
        its input places is accepted

        >>> hppn = test.simple_model()
        >>> poss = next(hppn.possibilities())
        >>> hppn.is_enabled(hppn.transition('t0'))
        False
        >>> poss.configuration().update_events({'e1'},0)
        >>> hppn.is_enabled(hppn.transition('t0'))
        True
        """
        # get input mode
        sip, sia = next((p,a) for p,a in t.input() if type(p) == SymbolicPlace)
        nip, nia = next((p,a) for p,a in t.input() if type(p) == NumericalPlace)
        hip, hia = next((p,a) for p,a in t.input() if type(p) == HybridPlace)
        input_mode = self.get_mode_with_places(sip, nip)

        possibilities = (c.possibility for c in sip if type(c) == Configuration and c.possibility.mode() == input_mode)
        return any(self.is_possibility_accepted(t, poss) for poss in possibilities)

    def is_numerical_transition(self, t):
        """ Check if the numerical condition of the given Transition is a 'real' condition, i.e. not true or false condition
        """
        return not (t.cn == true or t.cn == false)

    def firing_order(self):
        """ Return the list of enabled transitions with 'Numerical transition' first.

        >>> hppn = test.simple_model()
        >>> poss = next(hppn.possibilities())
        >>> len(list(hppn.firing_order()))
        0
        >>> poss.configuration().update_events({'e1'},0)
        >>> list(t.name for t in hppn.firing_order())
        ['t0']
        >>> hppn.empty()
        >>> len(list(hppn.firing_order()))
        0
        """
        numerical = []
        others = []
        for t in self.transition():
            if self.is_enabled(t):
                if self.is_numerical_transition(t):
                    numerical.append(t)
                else:
                    others.append(t)

        return numerical + others

    def accept(self, t):
        """ Select Tokens that have to be move throught the given Transition during the firing
        Firing is conditionned

        >>> hppn = test.simple_model()
        >>> len(hppn.place('OK1').tokens)
        1
        >>> len(hppn.place('ON1').tokens)
        100
        >>> len(hppn.place('OK1_ON1').tokens)
        100
        >>> hppn.accept(hppn.transition('t0'))
        >>> len(hppn.place('OK1').tokens)
        2
        >>> len(hppn.place('ON1').tokens)
        101
        >>> len(hppn.place('OK1_ON1').tokens)
        101
        >>> hppn.transition('t0').flat()
        >>> hppn = test.simple_model()
        >>> next(hppn.configurations()).update_events({'e1'}, 1)
        >>> hppn.accept(hppn.transition('t0'))
        >>> len(hppn.place('OK1').tokens)
        1
        >>> len(hppn.place('ON1').tokens)
        1
        >>> len(hppn.place('OK1_ON1').tokens)
        1
        >>> sum(1 for g in hppn.place('OK1').tokens if type(g) == GroupToken and g.tag == 'OK1_OK2 ' + GroupToken.accepted_tag)
        1
        >>> gt = next(g for g in hppn.place('OK1').tokens if type(g) == GroupToken and g.tag == 'OK1_OK2 ' + GroupToken.accepted_tag)
        >>> sorted(gt.tokens[0].events)
        [('e1', 1)]
        >>> sum(1 for g in hppn.place('ON1').tokens if type(g) == GroupToken and g.tag == 'ON1_ON2 ' + GroupToken.accepted_tag)
        1
        >>> sum(1 for g in hppn.place('OK1_ON1').tokens if type(g) == GroupToken and g.tag == 'OK1_ON1_OK2_ON2 ' + GroupToken.accepted_tag)
        1
        """
        # get input mode
        sip, sia = next((p,a) for p,a in t.input() if type(p) == SymbolicPlace)
        nip, nia = next((p,a) for p,a in t.input() if type(p) == NumericalPlace)
        hip, hia = next((p,a) for p,a in t.input() if type(p) == HybridPlace)
        input_mode = self.get_mode_with_places(sip, nip)
        sv = next(v for v in sia.vars())
        nv = next(v for v in nia.vars())
        hv = next(v for v in hia.vars())

        events = {v:set.union(*[ao.events for po,ao in t.output() if v in ao.vars()]) for v in sia.vars()}

        confToGroup = {sv: []}
        partToGroup = {nv: []}
        hybTToGroup = {hv: []}

        possibilities = (c.possibility for c in sip if type(c) == Configuration and (c.possibility.mode() == input_mode))

        # sure firing / classical firing
        for poss in possibilities:
            # test conditions
            if self.is_possibility_accepted(t, poss):
                confToGroup[sv].append(poss.configuration())
                partToGroup[nv] += list(poss.particles())
                hybTToGroup[hv] += list(poss.hybrid_tokens())

        sip.group(confToGroup)
        nip.group(partToGroup)
        hip.group(hybTToGroup)

    #@info()
    def fire(self):
        """ Fire transitions in the HPPN

        >>> hppn = test.simple_model()
        >>> hppn.fire()
        >>> len(list(hppn.tokens()))
        201
        >>> len(hppn.place('OK1').tokens)
        1
        >>> len(hppn.place('ON1').tokens)
        100
        >>> len(hppn.place('OK1_ON1').tokens)
        100
        >>> next(hppn.configurations()).update_events({'e1'},0)
        >>> hppn.fire()
        >>> len(list(hppn.tokens()))
        201
        >>> len(hppn.place('OK2').tokens)
        1
        >>> len(hppn.place('ON2').tokens)
        100
        >>> len(hppn.place('OK2_ON2').tokens)
        100
        """
        toFire = list(t for t in self.firing_order())
        #log.info("  {} enabled transitions".format(len(toFire)))
        firedNb = 0
        for t in toFire:
            self.accept(t)
            #log.debug("  {} accepted".format(t))
            t.fire()
            #log.debug("  {} fired".format(t))
        for t in toFire: t.flat()
        #log.info("  {} transitions fired".format(len(toFire)))

    def add_mode(self, modeName, symbolicName, numericalName, hybridName):
        """ Add a new mode to the HPPN
        
        >>> hppn = HPPN('foo')
        >>> hppn.add_place(SymbolicPlace('sp'))
        >>> hppn.add_place(NumericalPlace('np', StateSpaceRepresentation()))
        >>> hp = HybridPlace('hp', StateSpaceRepresentation())
        >>> hppn.add_place(hp)
        >>> hppn.add_mode("fun", 'sp', 'np', 'hp')
        >>> hppn.modesTable.get((hppn.place('sp'), hppn.place('np'))).name
        'fun'
        """
        if modeName in [m.name for m in self.modes()]:
            raise AssertionError('mode {} already in {}'.format(modeName, self.name))
        symbolic = self.place(symbolicName)
        numerical = self.place(numericalName)
        hybrid = self.place(hybridName)
        mode = Mode(modeName, symbolic, numerical, hybrid)
        self.modesTable[(symbolic, numerical)] = mode 
        #log.info('{} ({}, {}, {}) added'.format(modeName, symbolicName, numericalName, hybridName))

    def get_mode_with_places(self, ps, pn):
        return next((m for (pps,ppn), m in self.modesTable.items() if ps == pps and pn == ppn))

    def modes(self):
        return (m for m in self.modesTable.values())

    def get_mode_by_name(self, name):
        try:
            return next(m for m in self.modes() if m.name == name)
        except:
            raise Exception('no mode {} in {}'.format(name, [m.name for m in self.modes()]))

    def get_transition_input_mode(self, t):
        """ Return the input mode of the given Transition

        >>> hppn = test.simple_model()
        >>> hppn.get_transition_input_mode(hppn.transition('t0'))
        OK1-ON1
        >>> hppn.get_transition_input_mode(hppn.transition('t1'))
        OK2-ON2
        >>> hppn.get_transition_input_mode(hppn.transition('t2'))
        OK1-ON1
        """
        sip = next(p for p,_ in t.input() if type(p) == SymbolicPlace)
        nip = next(p for p,_ in t.input() if type(p) == NumericalPlace)
        return self.get_mode_with_places(sip, nip)

    def get_transition_output_mode(self, t):
        """ Return the output mode of the given Transition

        >>> hppn = test.simple_model()
        >>> hppn.get_transition_output_mode(hppn.transition('t0'))
        OK2-ON2
        >>> hppn.get_transition_output_mode(hppn.transition('t1'))
        KO1-OFF
        >>> hppn.get_transition_output_mode(hppn.transition('t2'))
        KO1-ON2
        """
        sip = next(p for p,_ in t.output() if type(p) == SymbolicPlace)
        nip = next(p for p,_ in t.output() if type(p) == NumericalPlace)
        return self.get_mode_with_places(sip, nip)

    def get_end_modes(self):
        """ Return the list of end modes of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(m.name for m in hppn.get_end_modes())
        ['KO1-OFF', 'KO12-OFF', 'STOPPED-OFF']
        """
        return [m for m in self.modes() if all(m != self.get_transition_input_mode(t) for t in self.transition())]

    def get_start_modes(self):
        """ Return the list of end modes of the HPPN

        >>> hppn = test.simple_model()
        >>> sorted(m.name for m in hppn.get_start_modes())
        ['READY-OFF']
        """
        return [m for m in self.modes() if all(m != self.get_transition_output_mode(t) for t in self.transition())]

    def copy(self):
        """ Copy the HPPN

        >>> hppn = HPPN('foo')
        >>> hppn.add_place(SymbolicPlace('sp'))
        >>> hppn.add_place(NumericalPlace('np', StateSpaceRepresentation()))
        >>> hp = HybridPlace('hp', StateSpaceRepresentation())
        >>> hppn.add_place(hp)
        >>> hppn.add_mode('fun', 'sp', 'np', 'hp')
        >>> hppn2 = hppn.copy()
        >>> hppn == hppn2
        False
        """
        copy = self.__class__(self.name)
        copy.fill(self)
        return copy

    def log_marking(self):

        # global
        conf = sum(1 for _ in self.configurations())
        part = sum(1 for _ in self.particles())
        hybT = sum(1 for _ in self.hybrid_tokens())
        #log.info("  global marking : {} configurations, {} particles, {} hybrid tokens".format(conf, part, hybT)) 
        
        # zombie
        average = sum(sum(1 for h in c.hybridTokens if h.configuration is not None) for c in self.configurations()) / conf
        zombieAverage = sum(len(c.hybridTokens) for c in self.configurations()) / conf
        #log.info("  average hybrid tokens per configuration : {}, with zombies : {}".format(average, zombieAverage)) 
        average = sum(sum(1 for h in pi.hybridTokens if h.particle is not None) for pi in self.particles()) / part
        zombieAverage = sum(len(pi.hybridTokens) for pi in self.particles()) / part
        #log.info("  average hybrid tokens per particle : {}, with zombies : {}".format(average, zombieAverage)) 

        # symbolic
        sym_marking = {p.name : (len(p.tokens), len(p.tokens)/conf) for p in self.symbolic_places() if len(p.tokens) > 0}
        #log.info("  symbolic marking (place: (nb, %)): {}".format(sym_marking)) 

        # numerical
        num_marking = {p.name : (len(p.tokens), len(p.tokens)/part) for p in self.numerical_places() if len(p.tokens) > 0}
        #log.info("  numerical marking (place: (nb, %)): {}".format(num_marking)) 

        # hybrid
        pl = [h.place.name for h in self.hybrid_tokens()]
        pl.sort()
        groupMarking = it.groupby(pl)
        names = []
        numbers = []
        for n, g in groupMarking:
            names.append(n)
            numbers.append(len(list(g)))
        hyb_marking = {n : (nb, nb / sum(numbers)) for n, nb in zip(names, numbers) if nb > 0}
        #log.info("  hybrid marking (place: (nb, %)): {}".format(hyb_marking)) 

        # possibility
        posss = list(self.possibilities())
        keyfunc = lambda poss: poss.mode().name
        posss = sorted(posss, key=keyfunc)
        group = it.groupby(posss, key=keyfunc)
        poss_group = {m: list(g) for m,g in group}
        poss_marking = {m: (len(g), len(g)/len(posss)) for m,g in poss_group.items()}
        #log.info('  possibility marking (mode: (nb, %): {}'.format(poss_marking))

        
    def draw(self, filename = None):

        # specify arc input and output
        for t in super().transition(): # super to condider diagnoser hybrid transitions
            for po,ao in t.output():
                ao.place = po
            for po,ao in t.input():
                ao.place = po

        # forced to call super here
        import drawhppn
        draw_place = drawhppn.draw_place(self)
        draw_transition = drawhppn.draw_transition(self)
        draw_arc = drawhppn.draw_arc(self)
        g = super().draw(None, place_attr=draw_place, trans_attr=draw_transition, arc_attr=draw_arc)
        drawhppn.draw(self, g, filename)

    def empty(self):
        """ Remove all tokens and reset k to 0

        >>> hppn = test.simple_model()
        >>> np.sum(len(p.tokens) for p in hppn.place())
        201
        >>> hppn.empty()
        >>> np.sum(len(p.tokens) for p in hppn.place())
        0
        """
        self.k = 0
        for p in self.place(): p.empty()

    def random_state(self):
        return np.random.uniform(self.stateSpace[0], self.stateSpace[1])

    def hppn_stats(self):
        """ Return statistics on HPPN structure
        >>> hppn = test.simple_model()
        >>> hppn.hppn_stats() == {'hybrid place number': 8, 'symbolic place number': 7, 'numerical place number': 3, 'transition number': 8, 'place number': 18, 'mode number': 8}
        True
        """
        s = {}
        s['place number'] = len(list(self.place()))
        s['symbolic place number'] = len(list(self.symbolic_places()))
        s['numerical place number'] = len(list(self.numerical_places()))
        s['hybrid place number'] = len(list(self.hybrid_places()))
        s['transition number'] = len(list(self.transition()))
        s['mode number'] = len(self.modesTable)

        return s

    
@optimization(50, 'x')
def api_update_particle_values(particles, k, dt, uc):
    """ Update the x of the given Particle list

    >>> hppn = test.simple_model()
    >>> sum1 = np.sum(np.sum(pi.x) for pi in hppn.particles())
    >>> api_update_particle_values(list(hppn.particles()), 0, 100, np.array([4, 4, 4]))
    >>> len(list(hppn.particles())) == 100
    True
    >>> sum2 = np.sum(np.sum(pi.x) for pi in hppn.particles())
    >>> sum1 != sum2
    True
    """
    #log.debug("  updating {} particle values...".format(len(particles)))
    #functions = []
    #xs = []
    #for pi in particles:
    #    functions.append(pi.place.ssr.noisy_state)
    #    xs.append((k-pi.t0, pi.x))
    #newXs = parallel2.distribute(functions, xs, (dt, uc))
    #for pi, x in zip(particles, newXs):
    #    pi.x = x
    return [api_update_particle_value(pi, k, dt, uc) for pi in particles]
  
def api_update_particle_value(pi, k,  dt, uc):
    """ Update the x of the given Particle according to the dynamic 
    equation and the given command associated to the NumericalPlace it belong
    A random noise is computed with a random normal gaussian parametrized 
    with the process noise parameters associated to the NumericalPlace and added to the x 

    >>> hppn = test.simple_model()
    >>> pi = next(hppn.particles())
    >>> v = api_update_particle_value(pi, 0, 100, np.array([4, 4, 4]))
    >>> np.all(v != [0, 0])
    True
    """
    #print(pi.place.name, pi.x[30], pi.x[31])
    #print("-------------pi.x avant new----------------:")
    #print(pi.x)
    pi.x = pi.place.ssr.noisy_state(pi.x, dt, uc, k)
    #print(pi.place.name, pi.x[30], pi.x[31])
    #print("-------------pi.x apres new----------------:")
    #print(pi.x)
    #pi.x
    return pi.x    

class Mode:

    def __init__(self, name, ps, pn, ph):
        self.name = name
        self.ps = ps
        self.pn = pn
        self.ph = ph

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name
 
if __name__ == "__main__":
    import doctest
    import test
    logging.disable(logging.INFO)
    doctest.testmod()
    #parallel2.shutdown()
