import numpy as np

samplingNumber = 7300 # observation number
timeSampling = 60 # observation period (s)

initialCommand = [0.000035] # initial continuous command (m^3) / 15s

T = [i*timeSampling for i in range(samplingNumber)]
#UC : Continous input
UC = [np.array(initialCommand)] * samplingNumber
#UD : Discrete input
UD = [set()] * samplingNumber

# after 310 min, close v13 every hour and open v13 20 min after
for i in range(310, 7300, 60):
    UD[i] = {'ferm v13'}
for i in range(330, 7300, 60):
    UD[i] = {'ouvr v13'}
