""" This module provides functions to obtain OS and machine informations
"""

import multiprocessing
import os

def cpu_count():
    """Return the number of cores in the system (physical + simulated)."""
    return multiprocessing.cpu_count()

def cpu_count_physical():
    """Return the number of physical cores in the system."""
    mapping = {}
    current_info = {}
    with open('/proc/cpuinfo', 'rb') as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                # new section
                if (b'physical id' in current_info and
                        b'cpu cores' in current_info):
                    mapping[current_info[b'physical id']] = \
                        current_info[b'cpu cores']
                current_info = {}
            else:
                # ongoing section
                if (line.startswith(b'physical id') or
                        line.startswith(b'cpu cores')):
                    key, value = line.split(b'\t:', 1)
                    current_info[key] = int(value)
    return sum(mapping.values()) or None

def cpu_info():
    """Return CPU details."""
    mapping = {}
    current_info = {}
    with open('/proc/cpuinfo', 'rb') as f:
        for line in f:
            line = line.strip()
            if not line:
                # new section
                if (b'processor' in current_info and b'model name' in current_info):
                    mapping[current_info[b'processor']] = current_info[b'model name']
                current_info = {}
            else:
                # ongoing section
                if (line.startswith(b'processor') or line.startswith(b'model name')):
                    key, value = line.split(b'\t:', 1)
                    current_info[key] = value
        if len(set(mapping.values())) == 1:
            return str(list(mapping.values())[0].decode('utf-8')).strip()
        else:
            s = []
            for p in sorted(mapping):
                s.append('{}: {}'.format(int(p), mapping[p].decode('utf-8').strip()))
            return ', '.join(s)

def ram():
    """Return RAM amount in kB."""
    with open('/proc/meminfo', 'rb') as f:
        for line in f:
            line = line.strip()
            if line.startswith(b'MemTotal'):
                value = line.split(b':')[1].strip()
                return value.decode('utf-8')
        return None

def file_exist(filename):
    """ Return True if the given file exist and is not empty
    """
    exist = False
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            exist = f != ''
    return exist
