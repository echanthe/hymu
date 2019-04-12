""" This module provides a functions to display a progress bar in a terminal.
"""

import sys
from termcolor import colored
class ProgressBar:
    '''
    Progress bar
    '''
    def __init__ (self, valmax, maxbar, title, valmin = 0):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        self.valmin = valmin
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title
    
    def update(self, val):
        # process
        perc  = round((float(val) - float(self.valmin))/ (float(self.valmax) - float(self.valmin)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)
  
        # render 
        out = '\r %s [%s%s] %3d %%' % (self.title, '=' * bar, ' ' * (self.maxbar - bar), perc)
        out = colored(out, attrs=['bold'])
        sys.stdout.write(out)
        sys.stdout.flush()

