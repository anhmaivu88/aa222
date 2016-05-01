### Example problem #1
from math import sqrt

def f(x):
    return -x[0]*x[1]

def g(x):
    return (x[0]+x[1]**2-1, \
           -x[0]-x[1])

x_star = (2/3, 1/sqrt(3))
f_star = -2/(3*sqrt(3))