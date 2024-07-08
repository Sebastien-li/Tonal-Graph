"""Utility functions for the project."""

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
