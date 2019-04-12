import numpy as np

# parameters
g = 9.81 # m·s^−2
waterDensity = 1000 # kg.m^-3

S1 = 0.0154 # m^2
S2 = 0.0154 # m^2
S3 = 0.0154 # m^2
Sn = 0.00005 # m^2

a13 = 0.4753 # v13
a32 = 0.4833 # v32
a20 = 0.1142 # v20

a1 = 0.10 # f1
a2 = 0.05 # f2
a3 = 0.05 # f3

K13 = a13 * Sn * np.sqrt(2*g) 
K32 = a32 * Sn * np.sqrt(2*g) 
K20 = a20 * Sn * np.sqrt(2*g) 
K1 = a1 * Sn * np.sqrt(2*g) 
K2 = a2 * Sn * np.sqrt(2*g) 
K3 = a3 * Sn * np.sqrt(2*g) 
parameters = [S1, S2, S3, K13, K32, K20, K1, K2, K3, waterDensity]

# noise
dnm = np.array([0, 0, 0]) # hi dynamic noise offsets
dns = np.array([.001, .001, .001]) # hi dynamic noise scales
onm = np.array([0, 0]) # sensor noise offsets [h1, mass(h1+h2+h3)]
ons = np.array([0.01, 0.1]) # sensor noise scales [h1, mass(h1+h2+h3)]
dns *= 2 
ons *= 2
noise = [dnm, dns, onm, ons]

seuilPf1 = .9
seuilPf4 = .9

#simuPredictDeltaTime = 1        # Simulator prediction period (s) (ex: every 1 sec)
#diagPredictDeltaTime = 240       # Diagnoser prediction period (s)
#observationDeltaTime = 240       # Diagnoser observation period (s)

#simuPredictNumber = int(observationDeltaTime/simuPredictDeltaTime)
#diagPredictNumber = int(observationDeltaTime/diagPredictDeltaTime)


