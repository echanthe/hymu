""" Wrapp system data into data that can be read by hymu

TODO: give more details and methodology for that part (construction of the SCENARIO_WRAPPER, etc.)
"""
from optparse import OptionParser
import imp
import os

parser = OptionParser(usage='usage: %prog SCENARIO_WRAPPER SCENARIO_FILE')
(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error(parser.usage)

wrapperName = args[0].split('.')[0]
scenarioName = args[1].split('.')[0]

filepath = os.getcwd()
wrapperlib = imp.load_source(wrapperName, filepath + os.path.sep + args[0])

scenario = wrapperlib.Scenario()
scenario.load(args[1])
scenario.write(scenarioName + '_wrapped.csv')

print('{} wrapped'.format(scenarioName))
