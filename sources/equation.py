""" This module implements some class used as interface to manipulate system equation (state equations, output equations, noise equations) and other algorithms such as particle resampling and Stochastic Scaling Algorithm (SSA).
"""

import numpy as np
import itertools as it
from functools import reduce # python3 comptibility
import matplotlib.pyplot as plt
import random
import bisect
import log
import pylab

def identity(x, *args):
    """ Return the identity function

    >>> identity(5, 6)
    5
    """
    return x

class LinearEquation:
    """ This class represents time-variant differential equation
    """

    def __init__(self, a, b):
        """ Create a LinearEquation with the given matrix A and B

        >>> eq = LinearEquation([1.], [0.5, 0.5, 0.5])
        >>> eq = LinearEquation([[1.], [2.]], [0.5, 0.5, 0.5])
        >>> eq = LinearEquation([1.], [[0.5] , [0.5]])
        """
        self.a = np.array(a)
        self.b = np.array(b)

    def compute(self, x, dt, u, t):
        """ Compute the LinearEquation with the given parameters X and U

        >>> eq = LinearEquation([1.], [0.5, 0.5, 0.5])
        >>> eq.compute([0.5], 1, [[1.], [1.], [1.]], 0)
        array([2.5])
        >>> eq = LinearEquation([[1, 1, 1.], [0, 1, 0.], [2, 1, 1.]], [[0.5, 0.5], [0.5, 1.], [2, 2.]])
        >>> eq.compute([0.5, 0.5, 0.5], 1, [1., 1.], 0)
        array([3. , 2.5, 6.5])
        """
        return np.add(x, np.add(np.dot(self.a, x), np.dot(self.b, u)) * dt)

    def __call__(self, x, dt, u, t):
        """ Compute the LinearEquation with the given parameters X and U

        >>> eq = LinearEquation([1.], [0.5, 0.5, 0.5])
        >>> eq([0.5], 1, [[1.], [1.], [1.]], 0)
        array([2.5])
        >>> eq = LinearEquation([[1, 1, 1.], [0, 1, 0.], [2, 1, 1.]], [[0.5, 0.5], [0.5, 1.], [2, 2.]])
        >>> eq([0.5, 0.5, 0.5], 1, [1., 1.], 0)
        array([3. , 2.5, 6.5])
        """
        return self.compute(x, dt, u, t)

    def copy(self):
        """
        >>> eq = LinearEquation([1.], [0.5, 0.5, 0.5])
        >>> eq2 = eq.copy()
        >>> all(eq2.a == eq.a)
        True
        >>> eq2.a = [0]
        >>> all(eq2.a == eq.a)
        False
        """
        return self.__class__(self.a, self.b)
    
class NoiseEquation:
    """ This class represents a normal noise law
    """

    def __init__(self, mu, sigma, mufunc = identity, sigmafunc = identity):
        """ Create a NoiseEquation with the given mu and sigma column matrices
        By default, the noise is invariant

        Mean and scale arrays are converted to float. Noise is float.
        This rule should have save me a lot of time.

        >>> mu = [1., 2., 3.]
        >>> sigma = [0.1, 0.2, 0.3]
        >>> n = NoiseEquation(mu,sigma)
        >>> n = NoiseEquation([1], [1, 2])
        Traceback (most recent call last):
            ...
        Exception: mu and sigma have not the same shape
        """
        self.mu = np.array(mu, dtype='float')
        self.sigma = np.array(sigma, dtype='float')
        if (self.mu.shape != self.sigma.shape):
            raise Exception("mu and sigma have not the same shape")
        self.mufunc = mufunc
        self.sigmafunc = sigmafunc

    def random(self, *args) :
        """ Generate a noise column vector from the column vector parameters mu and sigma of the NoiseEquation

        >>> mu = [1., 2., 0.]
        >>> sigma = [0.1, 0.2, 3]
        >>> n = NoiseEquation(mu,sigma)
        >>> len(n.random()) == 3
        True
        >>> np.sum(n.random()) != np.sum(n.random())
        True
        >>> def sigmafunc(s, t):
        ...     s2 = s.copy()
        ...     if t != 0:
        ...         s2[2] = s2[2]/np.exp(np.sqrt(t)/3) + 0.1
        ...     return s2
        >>> n = NoiseEquation(mu, sigma, identity, sigmafunc)
        >>> val = [np.average([abs(n.random(i)[2]) for _ in range(1000)]) for i in range(0, 100, 10)]
        >>> sorted(val, reverse = True) == val
        True
        """
        #mu = self.mufunc(self.mu, t)
        #sigma = self.sigmafunc(self.sigma, t)
        #print(t, mu, sigma)
        #return np.random.normal(mu, sigma)
        try:
            return np.random.normal(self.mufunc(self.mu, *args), self.sigmafunc(self.sigma, *args))
        except:
            print(self.mu[0], 0.0000000001)
            print(self.mufunc(self.mu, *args)[-1])

    def gaussian(self, mu, *args):
        """ Compute the weight of the given column matrix by using a Gaussian function

        >>> v = np.array([1., 2., 3.])
        >>> o = np.array([4., 5., 6.])
        >>> mu = [0., 0., 0.]
        >>> sigma = [.8, .8, .8]
        >>> n = NoiseEquation(mu, sigma)
        >>> n.gaussian(v - o)
        8.561719766390716e-11
        >>> n.gaussian(o - v)
        8.561719766390716e-11
        >>> v = np.array([1., 2., 3.])
        >>> o = np.array([1., 2., 3.])
        >>> sigma = [.1, .1, .1]
        >>> n = NoiseEquation(mu, sigma)
        >>> n.gaussian(v - o)
        63.49363593424098
        >>> sigma = [.01, .01, .01]
        >>> n = NoiseEquation(mu, sigma)
        >>> n.gaussian(v - o)
        63493.63593424098
        >>> v = np.array([55., 50., 45.])
        >>> o = np.array([54., 49., 44.])
        >>> n.gaussian(v - o)
        0.0
        >>> v = np.array([53., 48., 43.])
        >>> n.gaussian(v - o)
        0.0
        >>> sigma = [1, 1, 1]
        >>> n = NoiseEquation(mu, sigma)
        >>> v = np.array([55., 50., 45.])
        >>> o = np.array([54., 49., 44.])
        >>> n.gaussian(v - o)
        0.014167345154413289
        >>> v = np.array([53., 48., 43.])
        >>> n.gaussian(v - o)
        0.014167345154413289
        """
        #gauss = [gaussian(sm, m, s) if s != 0 else np.nan for sm, m, s in zip(self.mu, mu, self.sigma)]
        #gauss = gaussian(self.mu, mu, self.sigma)
        #gauss2 = [g for g in gauss if g is not np.nan]
        #res = np.multiply.reduce([gaussian(sm, m, s) for sm, m, s in zip(self.mu, mu, self.sigma) if s != 0])
        #res = np.multiply.reduce(gaussian(self.mu, mu, self.sigma))
        #if reduce(mul, self.sigma) == 0 and res == 0:
        #    for i, (mu,sigma, g) in enumerate(zip(mu, self.sigma, gauss)):
        #       if sigma != 0:
        #           print(i,mu, sigma, g)
        #if res != 0:
        #  for i, (mu,sigma, g) in enumerate(zip(mu, self.sigma, gauss)):
        #    sig = sum(1 for i in self.sigma if i == 100000)
        #    if i in range(61,65) and sig == 2:
        #      print(i, mu, sigma, g, res, sig)
        #gauss = gaussian(self.mufunc(self.mu, t), mu, self.sigmafunc(self.sigma, t))
        #gauss2 = np.multiply.reduce(gauss)
        #if np.isnan(gauss2):
        #    ids = [i for i, g in enumerate(gauss) if np.isnan(g)]
        #    print()
        #    for i in ids:
        #        print(t, i, self.mu[i], self.mufunc(self.mu, t)[i], self.sigma[i], self.sigmafunc(self.sigma, t)[i], mu[i])
        return np.multiply.reduce(gaussian(self.mufunc(self.mu, *args), mu, self.sigmafunc(self.sigma, *args)))

    def error(self, mu):
        return np.sqrt(sum([m**2 for m,s in zip(mu,self.sigma) if s != 0]))

    def copy(self):
        """
        >>> n = NoiseEquation([0], [1])
        >>> n2 = n.copy()
        """
        if self.mu is not None: mu = self.mu.copy() 
        else: mu = None
        if self.sigma is not None: sigma = self.sigma.copy() 
        else: sigma = None
        return self.__class__(mu, sigma, self.mufunc, self.sigmafunc)

class NoiselessEquation(NoiseEquation):
    """ NoiseEquation but without noise
    """

    def __init__(self, noiseEquation = None):
        if noiseEquation is not None:
            mu = np.zeros_like(noiseEquation.mu)
            sigma = np.zeros_like(noiseEquation.sigma)
        else:
            mu = np.array([0])
            sigma = np.array([0])
        super().__init__(mu, sigma)

    def random(self, *args) :
        """ Return a zeros filled array

        >>> mu = [1., 2., 3.]
        >>> sigma = [0.1, 0.2, 0.3]
        >>> n = NoiseEquation(mu,sigma)
        >>> n = NoiselessEquation(n)
        >>> len(n.random()) == 3
        True
        >>> n.random()
        array([0., 0., 0.])
        """
        return self.mu

    def gaussian(self, *args):
        """ Return one

        >>> mu = [1., 2., 3.]
        >>> sigma = [0.1, 0.2, 0.3]
        >>> n = NoiseEquation(mu,sigma)
        >>> n = NoiselessEquation(n)
        >>> n.gaussian(0)
        1
        """
        return 1

    def copy(self):
        """
        >>> n = NoiseEquation([0], [1])
        >>> n = NoiselessEquation(n)
        >>> n2 = n.copy()
        """
        return self.__class__(self)

class StateSpaceRepresentation:
    """ This class represents a state space representation system
    """

    def __init__(self, stateEquation = None, outputEquation = None, stateNoiseEquation = None, outputNoiseEquation = None):
        """ Create a StateSpaceRepresentation with the given state and output equation
        >>> ssr = StateSpaceRepresentation()
        """
        self.stateEquation = stateEquation
        self.outputEquation = outputEquation
        self.stateNoiseEquation = stateNoiseEquation
        self.outputNoiseEquation = outputNoiseEquation

    def noisy_state(self, *args):
        """ Compute the state equation of the StateSpaceRepresentation
        The noise can be time variant and then depends on the time

        >>> n = NoiseEquation([0], [1])
        >>> def eq(a,b,c,d): return a*b
        >>> ssr = StateSpaceRepresentation(eq, eq, n, n)
        >>> ssr.state(3, 4, 0, 0) > 0
        True
        """
        #stop = args[0][31] > .6
        #print()
        #print('toto x ', args[0][30], args[0][31])
        #print('  sigma', self.stateNoiseEquation.sigma[30], self.stateNoiseEquation.sigma[31])
        #print('  sig t',self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[30], self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[31])
        #print('  mu   ', self.stateNoiseEquation.mu[30], self.stateNoiseEquation.mu[31])
        #print('  mu t ',self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[30], self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[31])
        noise = self.stateNoiseEquation.random(*args)
        #print('titi x ', args[0][30], args[0][31])
        #print('  sigma', self.stateNoiseEquation.sigma[30], self.stateNoiseEquation.sigma[31])
        #print('  sig t',self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[30], self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[31])
        #print('  mu   ', self.stateNoiseEquation.mu[30], self.stateNoiseEquation.mu[31])
        #print('  mu t ',self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[30], self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[31])
        #print('  noise', noise[30], noise[31])
        move = self.stateEquation(*args)
        #print('tutu x ', args[0][30], args[0][31])
        #print('  sigma', self.stateNoiseEquation.sigma[30], self.stateNoiseEquation.sigma[31])
        #print('  sig t',self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[30], self.stateNoiseEquation.sigmafunc(self.stateNoiseEquation.sigma, *args)[31])
        #print('  mu   ', self.stateNoiseEquation.mu[30], self.stateNoiseEquation.mu[31])
        #print('  mu t ',self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[30], self.stateNoiseEquation.mufunc(self.stateNoiseEquation.mu, *args)[31])
        #print('  noise', noise[30], noise[31])
        #print('  move ', move[30], move[31])
        move += noise
        #print('  move ', move[30], move[31])
        #move = self.stateEquation(*args) + self.stateNoiseEquation.random(*args)
        #if stop: input()
        if len([m for m in move if np.isnan(m)]) > 0:
            ids = [i for i,m in enumerate(move) if np.isnan(m)]
            raise Exception('x{} is nan'.format(ids))
        return move

    def state(self, *args):
        """ Compute the state equation of the StateSpaceRepresentation

        >>> n = NoiseEquation([0], [1])
        >>> def eq(a,b,c,d): return a*b
        >>> ssr = StateSpaceRepresentation(eq, eq, n, n)
        >>> ssr.state(3, 4, 15, 3)
        12
        """
        return self.stateEquation(*args)

    def noisy_output(self, *args):
        """ Compute the output equation of the StateSpaceRepresentation
        The noise can be time variant and then depends the time

        >>> n = NoiseEquation([0], [1])
        >>> def eq(a,b,c,d): return a*b
        >>> ssr = StateSpaceRepresentation(eq, eq, n, n)
        >>> ssr.output(3, 4, 0, 0) > 0
        True
        """
        return self.outputEquation(*args) + self.outputNoiseEquation.random(*args)

    def output(self, *args):
        """ Compute the output equation of the StateSpaceRepresentation

        >>> n = NoiseEquation([0], [1])
        >>> def eq(a,b,c,d): return a*b
        >>> ssr = StateSpaceRepresentation(eq, eq, n, n)
        >>> ssr.output(3, 4, 0, 0)
        12
        """
        return self.outputEquation(*args)
    
    def weight(self, yc, *args):
        """ Compute the weight according to the given state vector and continuous observations

        >>> n = NoiseEquation([0], [1])
        >>> from operator import mul
        >>> ssr = StateSpaceRepresentation(mul, mul, n, n)
        >>> ssr.weight(12, 3, 4) > 0
        True
        """
        eyc = self.outputEquation(*args)
        diff = eyc - yc
        return self.outputNoiseEquation.gaussian(diff)

    def copy(self):
        """
        >>> s = StateSpaceRepresentation()
        >>> s2 = s.copy()
        """
        return self.__class__(self.stateEquation, self.outputEquation, self.stateNoiseEquation, self.outputNoiseEquation)
    
class LinearStateSpaceRepresentation(StateSpaceRepresentation):
    """ This class represents a linear system
    """

    def __init__(self, a, b, c ,d, snm, sns, onm, ons):
        """ Create a LinearStateSpaceRepresentation with the given A, B, C, D matrices and noise parameters associated with the state and output equations

        >>> a = [[1., 1.], [1., 1.]]
        >>> b = [[1, 0, 1.], [0, 1, 0.]]
        >>> c = [0.]
        >>> d = [1, 1, 0.]
        >>> snm = [1., 2.]
        >>> sns = [0.2, 0.1]
        >>> onm = [0]
        >>> ons = [1.2]
        >>> lssr = LinearStateSpaceRepresentation(a, b, c, d, snm, sns, onm, ons)
        """
        stateEquation = LinearEquation(a,b)
        outputEquation = LinearEquation(c,d)
        stateNoiseEquation = NoiseEquation(snm, sns)
        outputNoiseEquation = NoiseEquation(onm, ons)
        super().__init__(stateEquation, outputEquation, stateNoiseEquation, outputNoiseEquation)

    def copy(self):
        """
        >>> a = [[1., 1.], [1., 1.]]
        >>> b = [[1, 0, 1.], [0, 1, 0.]]
        >>> c = [0.]
        >>> d = [1, 1, 0.]
        >>> snm = [1., 2.]
        >>> sns = [0.2, 0.1]
        >>> onm = [0]
        >>> ons = [1.2]
        >>> lssr = LinearStateSpaceRepresentation(a, b, c, d, snm, sns, onm, ons)
        >>> lssr2 = lssr.copy()
        """
        a = self.stateEquation.a.copy()
        b = self.stateEquation.b.copy()
        c = self.outputEquation.a.copy()
        d = self.outputEquation.b.copy()
        snm = self.stateNoiseEquation.mu.copy()
        sns = self.stateNoiseEquation.sigma.copy()
        onm = self.outputNoiseEquation.mu.copy()
        ons = self.outputNoiseEquation.sigma.copy()
        return self.__class__(a, b, c ,d, snm, sns, onm, ons)

def gaussian(x, mu, sigma):
    """ Compute the probability of x for 1-dim Gaussian with mean mu and sigma

    >>> gaussian(0.8, 1., 0.5) == 0.7365402806066468
    True
    >>> gaussian(0.8, 1., 0.5) == 0.7365402806066468
    True
    >>> gaussian(1., 0.8, 0.5) == 0.7365402806066468
    True
    >>> gaussian(0., 0.2, 0.5) == 0.7365402806066467
    True
    >>> gaussian(100., 100., 0.1) == 3.989422804014327
    True
    >>> gaussian(0, 100, 0.001)
    0.0
    >>> gaussian(0, -0.07, 0.0001)
    0.0
    >>> gaussian(100, 100, np.inf)
    0.0
    >>> gaussian(100, 100, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    >>> #gaussian([100], [100], [0])
    array([ nan])
    >>> gaussian(np.array([0, 0]), np.array([0.2, 0.2]), np.array([0.5, 0.5]))
    array([0.73654028, 0.73654028])
    >>> gaussian(np.array([0, 0]), np.array([0.2, 0.2]), np.array([0.5, 0.5]))[0]
    0.7365402806066467
    >>> #gaussian(np.array([0, 0]), np.array([0.2, 0.2]), np.array([0.5, 0]))
    array([0.73654028,         nan])
    >>> gaussian(0, 4, 5)
    0.05793831055229655
    >>> gaussian(0, 4, 10000)
    3.989422484860515e-05
    """
    #if np.all(sigma != 0):
    return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))
    #else:
    #    raise ZeroDivisionError('float division by zero') 

def weibull(shape, scale):
    """ Return a function that compute the probability of x over a weibull function with the given parameters

    >>> w = weibull(5.93030013, 1.58463497)
    >>> (5.93030013 / 1.58463497) * ( 1.5/ 1.58463497)**(5.93030013 - 1) * np.exp(-(1.5/ 1.58463497)**5.93030013) == w(1.5)
    True
    """
    def eq(x):
        return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)
    return eq

def error(a, l):
    sum = 0.0
    for i in l: # calculate mean error
        err = np.linalg.norm(i - a) /np.linalg.norm(a)
        sum += err
    if len(l) > 0 :
        return sum / float(len(l))
    else : 
        return 0

def combinaison(l):
    """
    >>> l = (i for i in range(3))
    >>> for i in combinaison(l): print(i)
    (0, 1)
    (0, 2)
    (1, 0)
    (1, 2)
    (2, 0)
    (2, 1)
    """
    l = list(l)
    for i1 in l:
        for i2 in (i for i in l if i != i1):
            yield (i1,i2)

def true(*args):
    """ Always return True

    >>> true()
    True
    >>> true(5)
    True
    >>> true(5, [])
    True
    """
    return True

def false(*args):
    """ Always return False

    >>> false()
    False
    >>> false(5)
    False
    >>> false(5, [])
    False
    """
    return False

def sigmoid(x, scale = 1, mean = 0):
    """ The sigmoid function

    >>> sigmoid(0.458)
    0.6125396134409151
    """
    return 1 / (1 + np.exp(- scale * (x - mean)))

def subfinder(mylist, pattern = []):
    """
    >>> subfinder([1,2,3,4,5,1,2,3,5,6,6], [1,2,3,4])
    1
    >>> subfinder([1,2,3,4,5,1,2,3,5,6,6], [1,2,3])
    2
    >>> subfinder([1,2,3,4,5,1,2,3,5,6,6], [1,2,3,4,5,6])
    0
    >>> subfinder([1,2,3,4,5,1,2,3,5,6,6], [])
    0
    >>> subfinder([1,2,3], [1,2,3,4])
    0
    """
    matches = 0
    if len(pattern) > 0 :
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
                matches += 1
    return matches

class WeightedDistribution:
    """ Create a distribution of weighted states
    The weights are normalized with the sum of all the weights

    """
    def __init__(self, states, weights):
        accum = 0.0
        self.state = [s for s in states]
        self.distribution = []
        s = np.sum(weights)
        for w in weights:
            accum += w/s
            self.distribution.append(accum)

    def pick(self):
        """ Randomly pick a state based on the state weights
        """
        i = bisect.bisect_left(self.distribution, random.uniform(0,1))
        try:
            return self.state[i]
        except IndexError as e:
            # happens when all states are improbable w=0
            raise(Exception('all states are improblable'))
            #return self.state[0]
            #raise(e)

def alpha_weighted_function(s, n, a):
    """ Simple decision process

    >>> alpha_weighted_function(.2, .5, .5)
    0.35
    """
    return a * s + (1-a) * n

def stochastic_resource_assignation(scores, minResourceNumber, suffResourceNumber, maxResourceNumber) :
        """ From the given scores, distribute resources to differents users.

        >>> c = {1:1}
        >>> stochastic_resource_assignation(c, 30, 200, 1000)
        ({1: 200}, False)
        >>> c = {1:1, 2:2, 3:3}
        >>> stochastic_resource_assignation(c, 100, 100, 300)[0]
        {1: 100, 2: 100, 3: 100}
        >>> stochastic_resource_assignation(c, 100, 100, 3000)
        ({1: 100, 2: 100, 3: 100}, False)
        """
        assignations = {cluster: 0 for cluster in scores}
        clusters = list(scores.keys())
        remaining = maxResourceNumber
        matchButNotEnought = False
        while remaining > 0 and len(clusters) > 0:
            weights = [scores[cl] for cl in clusters]
            distribution = WeightedDistribution(clusters, weights)
            for _ in range(remaining):
                cl = distribution.pick()
                assignations[cl] += 1

            # scale to requirements
            for cl in assignations:
                if assignations[cl] > suffResourceNumber:
                    assignations[cl] = suffResourceNumber
                if assignations[cl] < minResourceNumber:
                    assignations[cl] = 0
                    matchButNotEnought = True
            clusters = [cl for cl, s in assignations.items() if s < suffResourceNumber]
            newRemaining = maxResourceNumber - sum(assignations.values())
            if newRemaining == remaining:
                # no affectation for some reason
                remaining = 0 # stop iteration
            else: 
                remaining = newRemaining
        return assignations, matchButNotEnought

def stochastic_resource_assignation_2(scores, maxResourceNumber) :
        """ From the given scores, distribute resources to differents users.
        """
        assignations = {cluster: 0 for cluster in scores}
        clusters = list(scores.keys())
        remaining = maxResourceNumber
        weights = [scores[cl] for cl in clusters]
        distribution = WeightedDistribution(clusters, weights)
        for _ in range(remaining):
            cl = distribution.pick()
            assignations[cl] += 1

        return assignations

def draw_scaling_assignment(scores, n_min, n_suff, n_max):
    clusters = {i: s for i,s in enumerate(sorted(scores))}
    assignment, debug = stochastic_resource_assignation(clusters, n_min, n_suff, n_max)
    print(debug)
    x = []
    y = []
    for a,b in assignment.items():
        x.append(a)
        y.append(b)
    fig = pylab.figure(1, figsize=(5,4), dpi=80, facecolor='w', edgecolor='k')
    pylab.scatter(x,y, c='b')
    pylab.rc('font', size=20)
    pylab.xlim(min(x), max(x)+1)
    pylab.ylim(ymin=-5)
    pylab.ylabel('resource number')
    pylab.xlabel('hypothesis')
    pylab.grid(True)
    pylab.rc('text', usetex=True)
    pylab.rc('font', family='serif')
    fig.tight_layout()
    print(sum(y))

    #assignment = stochastic_resource_assignation_2(clusters, n_max)
    #x = []
    #y = []
    #for a,b in assignment.items():
    #    x.append(a)
    #    y.append(b)
    #fig = pylab.figure(2, figsize=(5,4), dpi=80, facecolor='w', edgecolor='k')
    #pylab.scatter(x,y, c='r')
    #pylab.rc('font', size=20)
    #pylab.xlim(min(x), max(x)+1)
    #pylab.ylabel('resource number')
    #pylab.xlabel('hypothesis')
    #pylab.grid(True)
    #pylab.rc('text', usetex=True)
    #pylab.rc('font', family='serif')
    #fig.tight_layout()
    #print(sum(y))

    #assignment = sure_resource_assignation(clusters, n_min, n_suff, n_max)
    #x = []
    #y = []
    #for a,b in assignment.items():
    #    x.append(a)
    #    y.append(b)
    ##pylab.scatter(x,y)

    #assignment = weird_resource_assignation(clusters, n_min, n_suff, n_max)
    #x = []
    #y = []
    #for a,b in assignment.items():
    #    x.append(a)
    #    y.append(b)
    ##pylab.scatter(x,y)

    #fig = pylab.figure(3, figsize=(5,4), dpi=80, facecolor='w', edgecolor='k')
    #y = [clusters[xx] for xx in x]
    #pylab.scatter(x,y)
    #pylab.xlim(min(x), max(x)+1)
    #pylab.ylim(min(y), max(y))
    #pylab.ylabel('belief degree')
    #pylab.xlabel('hypothesis')
    #pylab.grid(True)
    #pylab.rc('text', usetex=True)
    #pylab.rc('font', family='serif')
    #fig.tight_layout()

    pylab.show()

def weird_resource_assignation(scores, minResourceNumber, suffResourceNumber, maxResourceNumber) :
        """ From the given scores, distribute resources to differents users.
        The score is normalized

        >>> c = {1:99,2:98,3:50,4:2,5:20,6:3,7:6}
        >>> weird_resource_assignation(c, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 37, 5: 200, 6: 54, 7: 110}
        >>> c = {1:99/99,2:98/99,3:50/99,4:2/99,5:20/99,6:3/99,7:6/99}
        >>> weird_resource_assignation(c, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 37, 5: 200, 6: 54, 7: 110}
        >>> c = {1:99,2:98,3:98,4:98,5:98,6:98,7:98, 8:0}
        >>> weird_resource_assignation(c, 30, 200, 1000)
        {1: 144, 2: 143, 3: 143, 4: 143, 5: 143, 6: 143, 7: 143, 8: 0}
        >>> weird_resource_assignation({1:99/99,2:98/99,3:98/99,4:98/99,5:98/99,6:98/99,7:98/99}, 30, 200, 1000)
        {1: 144, 2: 143, 3: 143, 4: 143, 5: 143, 6: 143, 7: 143}
        >>> c = weird_resource_assignation({i:i for i in range(500)}, 30, 200, 1000)
        >>> weird_resource_assignation({1:99,2:98,3:50}, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200}
        >>> weird_resource_assignation({1:99,2:98,3:50,4:0}, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 0}
        >>> c = {i: 98 for i in range(2000)}
        >>> c[2000] = 99
        >>> c = weird_resource_assignation(c, 30, 200, 1000)
        """
        assignations = {cluster: 0 for cluster in scores}
        remaining = maxResourceNumber
        interresting = sorted((cl for cl in scores if scores[cl] > 0), key=lambda cl: scores[cl], reverse=True)
        while remaining > 0 and len(interresting) > 0 :
            remainingNumber = remaining
            s = sum(scores[cluster] for cluster in interresting)
            for cluster in interresting : 
                add = int(round(scores[cluster] / s * remainingNumber))
                if assignations[cluster] + add > suffResourceNumber:
                    add = int(suffResourceNumber - assignations[cluster])
                assignations[cluster] += add
                remaining -= add
            interresting = [cluster for cluster, nb in assignations.items() if scores[cluster] > 0 and nb < suffResourceNumber]
            interresting = sorted(interresting, key=lambda cl: scores[cl], reverse=True)
            if remaining == remainingNumber : # no resource affected: happend when only 1 resource to share...
                remaining = 0 # stop iteration
        return assignations

def sure_resource_assignation(scores, minResourceNumber, suffResourceNumber, maxResourceNumber) :
        """ From the given scores, distribute resources to differents users.
        The score is normalized

        >>> c = {1:99,2:98,3:50,4:2,5:20,6:3,7:6}
        >>> sure_resource_assignation(c, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 0, 5: 200, 6: 0, 7: 200}
        >>> c = {1:99/99,2:98/99,3:50/99,4:2/99,5:20/99,6:3/99,7:6/99}
        >>> sure_resource_assignation(c, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 0, 5: 200, 6: 0, 7: 200}
        >>> c = {1:99,2:98,3:98,4:98,5:98,6:98,7:98, 8:0}
        >>> sure_resource_assignation(c, 30, 200, 1000)
        {1: 200, 2: 133, 3: 133, 4: 133, 5: 133, 6: 133, 7: 133, 8: 0}
        >>> sure_resource_assignation({1:99/99,2:98/99,3:98/99,4:98/99,5:98/99,6:98/99,7:98/99}, 30, 200, 1000)
        {1: 200, 2: 133, 3: 133, 4: 133, 5: 133, 6: 133, 7: 133}
        >>> #c = sure_resource_assignation({i:i for i in range(500)}, 30, 200, 1000)
        >>> sure_resource_assignation({1:99,2:98,3:50}, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200}
        >>> sure_resource_assignation({1:99,2:98,3:50,4:0}, 30, 200, 1000)
        {1: 200, 2: 200, 3: 200, 4: 0}
        >>> c = {i: 98 for i in range(2000)}
        >>> c[2000] = 99
        >>> #c = sure_resource_assignation(c, 30, 200, 1000)
        """
        assignations = {cluster: 0 for cluster in scores}
        remaining = maxResourceNumber
        scoreKey = lambda cl: scores[cl]
        interresting = sorted((cl for cl in scores if scores[cl] > 0), key=scoreKey, reverse=True)
        while remaining > 0 and len(interresting) > 0 :
            remainingSave = remaining
            inter = it.groupby(interresting, key=scoreKey)
            for score, cls in inter:
                cls = list(cls)
                add = int(round(remaining / len(cls)))
                if add > minResourceNumber:
                    if add > suffResourceNumber:
                        add = suffResourceNumber
                    for cl in cls: 
                        assignations[cl] = add
                        remaining -= add
                else: # happend when not enought resource to fill the score rank
                    # stop iteration
                    remaining = 0
            interresting = [cl for cl, nb in assignations.items() if scores[cl] > 0 and nb == 0]
            interresting = sorted(interresting, key=scoreKey, reverse=True)
            if remaining == remainingSave : # no resource affected: happend when only 1 resource to share...
                remaining = 0 # stop iteration
        return assignations

def discrete_gaussian(s1, s2, sigma=1):
    """ Compute a weight from two sets
         - compute the Hamming distance between the two sets
         - compute the weights with a Gaussian(0,sigma)
    >>> discrete_gaussian(set(), {'e1', 5})
    0.05399096651318806
    >>> discrete_gaussian({5}, {'e1', 5, 6})
    0.05399096651318806
    """
    dist =  len(s1 ^ s2)
    return gaussian(dist, 0, sigma)

    

def round_n(x, n):
    """ Round x with n significant digits
    """
    return round(x, -int(floor(log10(x))) + (n - 1))

def draw_dist():

    x = np.arange(-5, 10, 0.1)
    x = np.array([round(xi, 1) for xi in x])
    sample = [np.random.normal(3,1) for _ in range(10000)]
    sample = sorted(sample)
    groups = it.groupby(sample, lambda y : round(y, 1)) 
    y = np.zeros_like(x)
    for i, g in groups:
        y[np.where(x==i)[0][0]] = len(list(g))
    pylab.plot(x,y)
    pylab.show(block=False)

def draw_fault_noise():
    m = [3]
    s = [.3]

    def mufunc(m, t):
        m2 = m.copy()
        #m2[0] = np.exp(-t/m2[0]-0.25)
        m2[0] = m2[0] * np.exp(-t*20)
        #if t > 0:
        #    m2[0] = 0
        return m2

    def sigmafunc(s, t):
        s2 = s.copy()
        s2[0] = s2[0] * np.exp(-t) + 5e-3 # avoid scale to be 0
        #if t > 0:
        #    s2[0] = 1e-8
        return s2
    
    #s = [1e-8]
    #sigmafunc = identity
    n_eq = NoiseEquation(m, s, mufunc, sigmafunc)

    samples = np.zeros(1000)
    res = []
    t = np.arange(50)
    for ti in t:
        for i, s in enumerate(samples):
            samples[i] = s + n_eq.random(ti)
        res.append(samples.copy())

    pylab.plot(t, res)
    pylab.show(block=False)

def softmax(x):
    """ Compute softmax values of x
    """
    return np.exp(x)/np.sum(np.exp(x), axis=0)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
