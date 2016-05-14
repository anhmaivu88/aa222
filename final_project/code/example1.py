### Example problem #1
from numpy import sqrt, linspace

def f(x):
    return -x[0]*x[1]

def g(x):
    return (x[0]+x[1]**2-1, \
           -x[0]-x[1])

x_star = (2.0/3.0, 1.0/sqrt(3))
f_star = -2.0/(3*sqrt(3))

constraints = (lambda x: sqrt(1-x),
               lambda x:-sqrt(1-x),
               lambda x: -x)

xc = linspace(-2.0,1.0-1e-5)