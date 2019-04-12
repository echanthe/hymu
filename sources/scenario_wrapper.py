""" This module contains a library to generate system input scenario
"""

import imp
import csv
import re
import os


import sys
sys.path.append(os.path.dirname(__file__))

from scenario import Scenario
from progressbar import ProgressBar
class Scenario(Scenario):
    """ This class implement a Scenario object used as a wrapper between python variable and a CSV file
    """
    
    def load(self, filename):
        """ Fill the Scenario with the variables in the given python file
        """

        scenarioName = re.split('[/.]', filename)[-2]
        filepath = os.getcwd()
        scenariolib = imp.load_source(scenarioName, filepath + os.path.sep + filename)

        self.t += scenariolib.T 
        self.uc += scenariolib.UC
        self.ud += scenariolib.UD

    def write(self, filename):
        """ write only commands in a CSV file
        """
        with open(filename, "w", newline='') as f:
            data = csv.writer(f, delimiter='|')

            barI = 0
            barL = len(self.t)
            bar = ProgressBar(barL, 10, 'writting scenario')
            bar.update(barI)
            
            if self.uc[-1] is not None : ucNb = len(self.uc[-1]) 
            else : ucNb = 0

            ul = ['uc_{}'.format(i) for i in range(ucNb)]
            line = ['t'] + ul + ['ud'] + ['yd']
            data.writerow(line)
            for t,uc,ud in zip(self.t, self.uc, self.ud):
                udl = ','.join(e for e in ud) if ud else 'no'
                ucl = list(uc) if uc is not None else ['no'] * ucNb
                line = [t] + ucl + [udl]
                data.writerow(line)
                barI += 1
                bar.update(barI)
            print()
            print('scenario written in {}'.format(filename))
