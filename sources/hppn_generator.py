from hppn import *
from unicode_converter import to_sub

""" This class implement a generator that automate the creation of a HPPN model
"""
class HPPNGenerator:

    def __init__(self, name):
        self.hppn = HPPN(name)
        self.pI = 1
        self.trI = 1

    def _numerical_place_with(self, equation):
        def equal(ssr1, ssr2):
            c1 = ssr1.stateEquation == ssr2.stateEquation
            c2 = ssr1.outputEquation == ssr2.outputEquation
            c3 = np.all(ssr1.stateNoiseEquation.mu == ssr2.stateNoiseEquation.mu)
            c4 = np.all(ssr1.stateNoiseEquation.sigma == ssr2.stateNoiseEquation.sigma)
            c5 = np.all(ssr1.stateNoiseEquation.mufunc == ssr2.stateNoiseEquation.mufunc)
            c6 = np.all(ssr1.stateNoiseEquation.sigmafunc == ssr2.stateNoiseEquation.sigmafunc)
            c7 = np.all(ssr1.outputNoiseEquation.mu == ssr2.outputNoiseEquation.mu)
            c8 = np.all(ssr1.outputNoiseEquation.sigma == ssr2.outputNoiseEquation.sigma)
            c9 = np.all(ssr1.outputNoiseEquation.mufunc == ssr2.outputNoiseEquation.mufunc)
            c10 = np.all(ssr1.outputNoiseEquation.sigmafunc == ssr2.outputNoiseEquation.sigmafunc)
            return c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9 and c10
        return next((p for p in self.hppn.numerical_places() if equal(p.ssr, equation)), None)

    def _hybrid_place_with(self, equation):
        def equal(ssr1, ssr2):
            return ssr1.stateEquation == ssr2.stateEquation
        return next((p for p in self.hppn.hybrid_places() if equal(p.ssr, equation)), None)
    
    def add_mode(self, name, cEqu, hEqu):
        #ps = SymbolicPlace('ps{}'.format(self.pI))
        ps = SymbolicPlace('p{}'.format(self.pI))
        self.hppn.add_places([ps])
        self.pI += 1
        # try to find existing place
        pn = self._numerical_place_with(cEqu)
        if pn is None:
            #pn = NumericalPlace('pn{}'.format(self.pI), cEqu)
            pn = NumericalPlace('p{}'.format(self.pI), cEqu)
            self.hppn.add_places([pn])
            self.pI += 1
        ph = self._hybrid_place_with(hEqu)
        if ph is None:
            #ph = HybridPlace('ph{}'.format(self.pI), hEqu)
            ph = HybridPlace('p{}'.format(self.pI), hEqu)
            self.hppn.add_places([ph])
            self.pI += 1
        
        self.hppn.add_mode(name, ps.name, pn.name, ph.name)

    def set_state_space(self, low, high):
        self.hppn.stateSpace = [low, high]

    def set_symbolic_weight_function(self, func):
        """ Set a particular symbolic weight function
        This is a function returning a weight from two set of events
        Default is the discrete_gaussian function from equation.py package 
        """
        self.hppn.symbolic_weight_function = func

    def set_event_labels(self, udo, ydo, uo):
        """ Specify to the model the discrete event label sets:
          - observable command events (udo)
          - observable sensor events (ydo)
          - unobservable events (uo)
        """
        self.hppn.udo = udo.copy()
        self.hppn.ydo = ydo.copy()
        self.hppn.uo = uo.copy()
        self.hppn.o = udo | ydo

    def add_mode_transition(self, modeName1, modeName2, symbolicCondition, numericalCondition, hybridCondition):
        t = Transition('t{}'.format(self.trI), symbolicCondition, numericalCondition, hybridCondition)
        mi = self.hppn.get_mode_by_name(modeName1)
        mo = self.hppn.get_mode_by_name(modeName2)
        inputs = [mi.ps, mi.pn, mi.ph]
        outputs = [mo.ps, mo.pn, mo.ph]
        self.hppn.add_transition(inputs, t, outputs)

        self.trI += 1

    def sym_cond(self, event):
        def sym(c): 
            try: return sorted(c.events, key=lambda e: e[1])[-1][0] == event
            except: return False
        sym.__name__ = 'occ({})'.format(to_sub(event))
        sym.event= event
        return sym

    def get_hppn(self):
        return self.hppn
