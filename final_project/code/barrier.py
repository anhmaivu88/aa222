# Standard libraries
# from math import log, exp
import numpy as np
from numpy.linalg import norm
from ad import adnumber
from ad.admath import log, exp

def inv_barrier(g):
    """Inverse barrier function
    Takes a constraint evaluation, and builds a scalar inverse barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of inverse barrier for given values
    """
    res = 0
    for val in g:
        res = res - 1/val
    return res


def log_barrier(g):
    """Log barrier function
    Takes a constraint evaluation, and builds a scalar log barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of log barrier for given values
    """
    res = 0
    for val in g:
        if val < -1:
            pass
        else:
            if val >= 0:
                res = float('inf')
            else:
                res = res - log(-val)
    return res

def ext_obj(g,e):
    """Exterior Point Objective
    Creates an objective function for the exterior point method
    Usage
        res = ext_obj(g,e)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
        e = scalar value, represents interior distance from boundary sought
    Outputs
        res = value of objective function
    """
    res = 0
    for val in g:
        res = res + max(0,val+e)**2
    return res

def feasible(g):
    """Checks if a constraint value is feasible
    Usage
        res = feasible(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = boolean feasibility, True/False
    """
    slack = 1e-1
    res = True
    for val in g:
        res = res and (val<0)
    return res
