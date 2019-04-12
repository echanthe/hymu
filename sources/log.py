""" This module provide a simple global logging interface
"""

import logging
from termcolor import colored
from logging.handlers import RotatingFileHandler
from logging import INFO, NOTSET, disable

ENABLED = False
#LOG_LEVEL = logging.DEBUG
LOG_LEVEL = INFO

if not ENABLED:
    LOG_LEVEL = logging.ERROR

def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s :: %(message)s')

    #file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
    #file_handler.setLevel(logging.DEBUG)
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)

    start = colored('%(name)s - %(levelname)s', 'white', attrs=['dark'])
    start += colored(' ::', attrs=['bold'])
    formatter = logging.Formatter('{} %(message)s'.format(start))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

