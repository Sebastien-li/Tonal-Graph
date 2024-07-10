"""Utility functions for the project."""
import sys
from datetime import datetime
from pathlib import Path
import logging as log
import bisect
import numpy as np


def find_le(a, x, key = None):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right([key(y) for y in a], x)
    if i:
        return a[i-1]
    raise ValueError

def display_float(x, precision = 3):
    """ Returns a string representation of a float with a given precision. Removes trailing zeros"""
    return f'{float(x):.{precision}f}'.rstrip('0').rstrip('.')

def interval_collision(interval1_start, interval1_end, interval2_start, interval2_end):
    """ Returns True if two intervals overlap."""
    return interval1_start < interval2_end and interval1_end > interval2_start

def interval_in(interval1_start, interval1_end, interval2_start, interval2_end):
    """ Returns True if interval 1 in interval 2."""
    return interval1_start >= interval2_start and interval1_end <= interval2_end

def cartesian_product(*arrays):
    """ Returns the cartesian product of a list of arrays."""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def get_multilogger():
    """ Returns a logger that writes to a file and to the console."""
    logger = log.getLogger('HALight logger')
    formater = log.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger.setLevel(log.DEBUG)

    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    handler_debug = log.FileHandler(Path(f'logs/{now}.log'), mode='w')
    handler_debug.setLevel(log.DEBUG)
    handler_debug.setFormatter(formater)
    logger.addHandler(handler_debug)

    handler_info = log.StreamHandler(sys.stdout)
    handler_info.setLevel(log.INFO)
    handler_info.setFormatter(formater)
    logger.addHandler(handler_info)
    return logger
