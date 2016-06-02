import numpy as np
from ad import gh
from util import col
from random import random

def get_objective(my_case,dim,full=False):
    """Return an m-dimensional objective function
    Usage
        fcn, grad, name = get_objective(my_case,dim)
        fcn, grad, name, opt = get_objective(my_case,dim,full=True)
    Arguments
        my_case = type of function to return:
                    0 = Single-Ridge Function
                    1 = Double-Ridge Function
        dim     = dimension of input
    Keyword Arguments
        full    = full return flag
    Outputs
        fcn  = scalar function handle
        grad = gradient function handle
        name = name of my_case
        opt  = random parameters used to generate fcn, grad
    """
    # Double-Ridge Function
    if my_case == 1:
        # Random vector
        A = [2.*(random()-0.5) for i in range(dim)]
        B = [2.*(random()-0.5) for i in range(dim)]
        # Ridge Function
        fcn = lambda x: sum([x[i]*A[i] for i in range(len(A))])**2 + \
                        sum([x[i]*B[i] for i in range(len(B))])**2
        grad, _ = gh(fcn)
        # Optional outputs
        opt = [A,B]
        # Function type name
        name = "Double-Ridge"
    # Single-Ridge Function
    else:
        # Random vector
        A = [2.*(random()-0.5) for i in range(dim)]
        # Ridge Function
        fcn = lambda x: sum([x[i]*A[i] for i in range(len(A))])**2
        grad, _ = gh(fcn)
        # Optional outputs
        opt = A
        # Function type name
        name = "Single-Ridge"

    # Minimal return
    if full == False:
        return fcn, grad, name
    # Full return
    else:
        return fcn, grad, name, opt

if __name__ == "__main__":
    # Test
    m = 3
    F, dF, name, opt = get_objective(1,m,full=True)
    x = [1,2,3]
    print(" F(x)={}".format(F(x)))
    print("dF(x)={}".format(dF(x)))