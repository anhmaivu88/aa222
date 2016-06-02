import numpy as np
from ad import gh
from pyutil.numeric import col
from random import random, randint
from math import sqrt

def rs():
    """Random scalar
    """
    return 2.*(random()-0.5)

def normalize(A):
    """Normalize a vector by the 2-Norm
    """
    n = sqrt(sum([a**2 for a in A]))
    return [a/n for a in A]

def get_objective(my_case,dim,full=False):
    """Return an m-dimensional objective function
    Usage
        fcn, grad, name = get_objective(my_case,dim)
        fcn, grad, name, opt = get_objective(my_case,dim,full=True)
    Arguments
        my_case = type of function to return:
                    0 = Single-Ridge Function
                    1 = Double-Ridge Function
                    2 = Mixed Function
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
        A = normalize([rs() for i in range(dim)])
        B = normalize([rs() for i in range(dim)])
        # Construct function
        fcn = lambda x: sum([x[i]*A[i] for i in range(len(A))])**2 + \
                        sum([x[i]*B[i] for i in range(len(B))])**2
        grad, _ = gh(fcn)
        # Optional outputs
        opt = [A,B]
        # Function type name
        name = "Double-Ridge"
    # Mixed Function
    elif my_case == 2:
        n = dim/2
        m = dim-n
        # Sparse random vector
        A = normalize([rs() for i in range(n)] + [0]*m)
        # Complementary sparse random vector
        B = normalize([0]*n + [rs() for i in range(m)])
        # Construct function
        fcn = lambda x: sum([x[i]*A[i] for i in range(len(A))]) + \
                        sum([x[i]**2*B[i] for i in range(len(B))])
        grad, _ = gh(fcn)
        # Optional outputs
        opt = [A,B]
        # Function type name
        name = "Mixed"
    # Randomly-Mixed Function
    elif my_case == 3:
        # Sparse random vector
        A = normalize([rs()*randint(0,1) for i in range(dim)])
        # Complementary sparse random vector
        B = normalize([(A[i]==0)*rs()*randint(0,1) for i in range(dim)])
        # Construct function
        fcn = lambda x: sum([x[i]*A[i] for i in range(len(A))]) + \
                        sum([x[i]**2*B[i] for i in range(len(B))])
        grad, _ = gh(fcn)
        # Optional outputs
        opt = [A,B]
        # Function type name
        name = "Random-Mixed"
    # Single-Ridge Function
    else:
        # Random vector
        A = normalize([rs() for i in range(dim)])
        # Construct function
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