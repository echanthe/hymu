# monitor configuration

#DIAGNOSER_MINIMUM_RESOURCE_NUMBER = 20
DIAGNOSER_MINIMUM_RESOURCE_NUMBER = 5
#DIAGNOSER_SUFFICIENT_RESOURCE_NUMBER = 40
DIAGNOSER_MAXIMUM_RESOURCE_NUMBER = 20
#DIAGNOSER_MAXIMUM_RESOURCE_NUMBER = 1000
DIAGNOSER_TOTAL_RESOURCE_NUMBER = 200
DIAGNOSER_INITIAL_RESOURCE_NUMBER = 10
DIAGNOSER_CONFIDENCE_SYMBOLIC_NUMERICAL = 0.5
PROGNOSER_ENABLED = [True] + [False]*29 # will be repeated as a cycle
#PROGNOSER_ENABLED = False
PROGNOSER_PREDICTION_HORIZON = 10000
PROGNOSER_MINIMUM_RESOURCE_NUMBER = 1
PROGNOSER_MAXIMUM_RESOURCE_NUMBER = 3
PROGNOSER_TOTAL_RESOURCE_NUMBER = 50
M0 = 'Nom1' # set to None the start everywhere
T0 = 0
X0 = [0.60, 0.55, 0.58] # initial continuous state (m^2) - [h1, h2, h3]
H0 = [0, 0, 0.001, 0, 0]  # initial degradation ([0-1])

UC0 = [0.000035] # initial continuous command (m^3)
