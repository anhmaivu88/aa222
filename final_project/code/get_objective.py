import numpy as np
from numpy.random import random
from ad import gh
from util import col

def get_objective(my_case,dim,full=False):
    """
    """
    # Dummy case
    if my_case == 1:
        pass
    # Single-Ridge Function
    else:
        # Random vector
        A = random([dim,1])
        # Ridge Function
        fcn = lambda x: 0.5*(np.array(x).dot(A))**2
        grad, _ = gh(fcn)
        # Optional outputs
        opt = A

    # Minimal return
    if full == False:
        return fcn, grad
    # Full return
    else:
        return fcn, grad, opt

if __name__ == "__main__":
    # Test
    m = 3
    F, dF, opt = get_objective(0,m,full=True)
    x = [1,2,3]
    print(" F(x)={}".format(F(x)))
    print("dF(x)={}".format(dF(x)))