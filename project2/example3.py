### Example problem #1
from math import sqrt

def f(x):
    return x[0]-2*x[1]+x[2]

def g(x):
    return (x[0]**2+x[1]**2+x[2]**2-1,)

x_star = (-1.0/sqrt(6),sqrt(2.0/3.0),-1.0/sqrt(6))
f_star = -sqrt(6.0)