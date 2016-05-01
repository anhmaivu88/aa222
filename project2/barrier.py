from math import log

def inv_barrier(g):
    """Inverse barrier function
    Takes a constraint evaluation, and builds a scalar inverse barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, representing an leq vector constraint
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
        g = iterable of scalar values, representing an leq vector constraint
    Outputs
        res = value of log barrier for given values
    """
    res = 0
    for val in g:
        if val < -1:
            pass
        else:
            res = res - log(-val)
    return res