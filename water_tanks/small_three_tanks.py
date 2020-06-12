import sys
sys.path.append("..")
from hppn_generator import HPPNGenerator
from equation import *
import drawhppn
from unicode_converter import to_sub
import numpy as np
from scipy.integrate import odeint

def model(name, config):
    """
    >>> import config_default as config
    >>> model = model('small three-tanks', config)
    >>> model.draw()
    >>> from multimode import Multimode
    >>> Multimode(model).draw()
    >>> import diagnoser, prognoser
    >>> diagnoser.Diagnoser(model).draw()
    >>> prognoser.Prognoser(model).draw()
    """

    dnm, dns, onm, ons = config.noise
    S1, S2, S3, K13, K32, K20, K1, K2, K3, liquid_density = config.parameters

    generator = HPPNGenerator(name)
    model = generator.get_hppn()
    
    
    # events
    observables = {'ouvr v13', 'ferm v13', 'ouvr v32', 'ferm v32'}
    faults = {'f{}'.format(i) for i in range(6)}
    generator.set_event_labels(observables, set(), faults)
    
    ###
    ### continuous dynamics
    ###
    
    def de_gen(f):
        def de(x, dt, u, t):
            ddt = dt/4. # max shannon: dt/2
            nb = int(np.round(dt / ddt))
            for _ in range(nb):
                xdot = f(x, ddt, u)
                xdot *= ddt
                x += xdot
                x = np.maximum(x,0) # prevent water level to be < 0
            return x
            #a = odeint(f, x, [0,dt], args = (u,))
            #return a[1]
        return de
    
    def oe_gen():
        def oe(x, dt, u, t):
            #res = np.mat('[0.; 0.]')
            #res[0] = x[0]
            #res[1] = sum(i*S1*liquid_density for i in x[0:3])
            #return res
            return np.array([x[0], sum(S1 * liquid_density * x)])
    
        return oe
    
    oe = oe_gen()
    dn = NoiseEquation(dnm, dns)
    on = NoiseEquation(onm, ons)

    # d1
    def f(x,t,u):
        h = np.maximum(x,0.)
        dh = np.zeros(3)
        dh[0] = 1./S1 * (u[0] - K13 * np.sign(h[0]-h[2]) * np.sqrt(np.abs(h[0]-h[2])))
        dh[1] = 1./S2 * (K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])) - K20 * np.sqrt(h[1]))
        dh[2] = 1./S3 * (K13 * np.sign(h[0]-h[2]) * np.sqrt(np.abs(h[0]-h[2])) - K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])))
        return dh
    de = de_gen(f)
    d1 = StateSpaceRepresentation(de, oe, dn, on)
    d1.name = 'd1'

    # d2
    def f(x,t,u):
        h = np.maximum(x,0.)
        dh = np.zeros(3)
        dh[0] = 1./S1 * u[0]
        dh[1] = 1./S2 * (K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])) - K20 * np.sqrt(h[1]))
        dh[2] = 1./S3 * (- K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])))
        return dh
    de = de_gen(f)
    d2 = StateSpaceRepresentation(de, oe, dn, on)
    d2.name = 'd2'
        
    # d1f1
    def f(x,t,u):
        h = np.maximum(x,0.)
        dh = np.zeros(3)
        dh[0] = 1./S1 * (u[0] - K13 * np.sign(h[0]-h[2]) * np.sqrt(np.abs(h[0]-h[2])) - K1 * np.sqrt(h[0]))
        dh[1] = 1./S2 * (K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])) - K20 * np.sqrt(h[1]))
        dh[2] = 1./S3 * (K13 * np.sign(h[0]-h[2]) * np.sqrt(np.abs(h[0]-h[2])) - K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])))
        return dh
    de = de_gen(f)
    d1f1 = StateSpaceRepresentation(de, oe, dn, on)
    d1f1.name = 'd1f1'

    # d2f1
    def f(x,t,u):
        h = np.maximum(x,0.)
        dh = np.zeros(3)
        dh[0] = 1./S1 * (u[0] - K1 * np.sqrt(h[0]))
        dh[1] = 1./S2 * (K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])) - K20 * np.sqrt(h[1]))
        dh[2] = 1./S3 * (- K32 * np.sign(h[2]-h[1]) * np.sqrt(np.abs(h[2]-h[1])))
        return dh
    de = de_gen(f)
    d2f1 = StateSpaceRepresentation(de, oe, dn, on)
    d2f1.name = 'd2f1'

    ###
    ### degradation
    ###
    
    shapes = [1, 1, 1, 1, 1]

    def de_gen(scales):
        def f(h, s, tr,t):
            diffH = np.zeros(5)
            for i in range(5):
                if s[i] > 0:
                    diffH[i] = - np.exp((-1/s[i])*(tr[i]+t)) + np.exp((-1/s[i])*tr[i])
                else: 
                    diffH[i] = 1-h[i]
            return diffH
        def de(x, events, h, modes , t):
            s = scales.copy()
            nb = 1 + sum(1 for e in events if e[0] == 'open v13' or e[0] == 'close v13')
            s[3] /= nb
            nb = 1 + sum(1 for e in events if e[0] == 'open v32' or e[0] ==  'close v32')
            s[4] /= nb
            return h + f(h, s, np.log(-h+1) * (- s), t)
        return de
    
    # deg1: nom1, d1
    ea, eb, ec, e13, e32 = 100000, 80000, 60000, 30000000, 30000000
    scales = np.array([ea, ea, ea, e13, e32])
    deg1 = StateSpaceRepresentation(de_gen(scales))
    deg1.name ='deg1'

    # deg2: nom2, d2
    scales = np.array([ec, ea, ea, e13, e32])
    deg2 = StateSpaceRepresentation(de_gen(scales))
    deg2.name = 'deg2'
    
    # deg3: 12f4, d2
    scales = np.array([ec, ea, ea, 0, e32])
    deg3 = StateSpaceRepresentation(de_gen(scales))
    deg3.name = 'deg3'
    
    # deg4: 1f1, d1f1
    scales = np.array([0, ea, ea, e13, e32])
    deg4 = StateSpaceRepresentation(de_gen(scales))
    deg4.name = 'deg4'
    
    # deg5: 2f1, d2f1 = deg4
    deg5 = deg4
    deg5.name = 'deg5'

    # deg56: 12f1f4, d2f1
    scales = np.array([0, ea, ea, 0, e32])
    deg6 = StateSpaceRepresentation(de_gen(scales))
    deg6.name = 'deg6'
    

    ###
    ### modes
    ###
    
    # Nominal 1 = nom1, d1, deg1
    generator.add_mode('Nom1', d1, deg1)
    model.get_mode_by_name('Nom1').ps.color = drawhppn.green
    
    # Nominal 2 = nom2, d2, deg2
    generator.add_mode('Nom2', d2, deg2)
    model.get_mode_by_name('Nom2').ps.color = drawhppn.green
    
    # 12f4 = 12f4, d2, deg3
    generator.add_mode('Deg1', d2, deg3)
    model.get_mode_by_name('Deg1').ps.color = drawhppn.orange
    
    # 1f1 = 1f1, d1f1, deg4
    generator.add_mode('Deg2', d1f1, deg4)
    model.get_mode_by_name('Deg2').ps.color = drawhppn.orange
    
    # 2f1 = 2f1, d2f1, deg5
    generator.add_mode('Deg3', d2f1, deg5)
    model.get_mode_by_name('Deg3').ps.color = drawhppn.orange
   
    # 12f1f4 = 12f1f4, d2f1, deg6
    generator.add_mode('Deg4', d2f1, deg6)
    model.get_mode_by_name('Deg4').ps.color = drawhppn.orange
   
    # 12f4f0 = 12f4f0, d2, deg3
    generator.add_mode('Def1', d2, deg3)
    model.get_mode_by_name('Def1').ps.color = drawhppn.red
    
    # 12f1f4f0 = 12f1f4f0, d2f1, deg6
    generator.add_mode('Def2', d2f1, deg6)
    model.get_mode_by_name('Def2').ps.color = drawhppn.red
    
    # 1f1f0 = 1f1f0, d1f1, deg4
    generator.add_mode('Def4', d1f1, deg4)
    model.get_mode_by_name('Def4').ps.color = drawhppn.red
    
    # 2f1f0 = 2f1f0, d2f1, deg4
    generator.add_mode('Def3', d2f1, deg5)
    model.get_mode_by_name('Def3').ps.color = drawhppn.red

    ###
    ### transitions
    ###
    
    # nominal
    sym = generator.sym_cond('ferm v13')
    generator.add_mode_transition('Nom1', 'Nom2', sym, None, None)
    generator.add_mode_transition('Deg2', 'Deg3', sym, None, None)
    sym = generator.sym_cond('ouvr v13')
    generator.add_mode_transition('Nom2', 'Nom1', sym, None, None)
    generator.add_mode_transition('Deg3', 'Deg2', sym, None, None)

    # simple f1
    sym = generator.sym_cond('f1')
    def deg_cond(ht):
        return ht.h[0] > config.seuilPf1 #np.random.normal(.92, .01)
    deg_cond.__name__ = to_sub('p\u2092(f1)') +' > 0.9'
    generator.add_mode_transition('Nom1', 'Deg2', sym, None, deg_cond)
    generator.add_mode_transition('Nom2', 'Deg3', sym, None, deg_cond)

    # simple f4
    sym = generator.sym_cond('f4')
    def deg_cond(ht):
        return ht.h[3] > config.seuilPf4 #np.random.normal(.92, .01)
    deg_cond.__name__ = to_sub('p\u2092(f4)') + ' > 0.9'
    generator.add_mode_transition('Nom1', 'Deg1', sym, None, deg_cond)
    generator.add_mode_transition('Nom2', 'Deg1', sym, None, deg_cond)
    generator.add_mode_transition('Deg2', 'Deg4', sym, None, deg_cond)
    generator.add_mode_transition('Deg3', 'Deg4', sym, None, deg_cond)

    # f0 = h2 < .5
    f0 = generator.sym_cond('f0')
    def h2_fail(pi):
        return pi.x[1] < .6
    h2_fail.__name__ = to_sub('l2') + ' < 0.6'
    generator.add_mode_transition('Deg2', 'Def4', f0, h2_fail, None)
    generator.add_mode_transition('Deg3', 'Def3', f0, h2_fail, None)
    generator.add_mode_transition('Deg1', 'Def1', f0, h2_fail, None)
    generator.add_mode_transition('Deg4', 'Def2', f0, h2_fail, None)

    def event_set_weight(e, o):
        # truly trust f0 because it is defined by user,
        # there is no change of behavior, just the mission is decided to be failed
        if any(ee[0]=='f0' for ee in e):
            return 2 # best weight with Gaussian(0,1) = 0.4 -> largely more important
        else:
            return discrete_gaussian(e,o, .5)
    generator.set_symbolic_weight_function(event_set_weight)
         
    return model

if __name__ == "__main__":
    import doctest
    import logging
    logging.disable(logging.INFO)
    doctest.testmod()
   
