### Example problem #2
from numpy import sqrt, linspace

def f(x):
    return 4.0*x[0]**2 - x[0] - x[1] - 2.5

def g(x):
    return (-x[1]**2+1.5*x[0]+1.0, \
            x[1]**2+2.0*x[0]**2-2.0*x[0]-4.25)

x_star = (0.164439, 2.12716)
f_star = -4.68344

constraints = (lambda x: sqrt(1.5*x**2-2.0*x+1.0),
               lambda x:-sqrt(1.5*x**2-2.0*x+1.0),
               lambda x: sqrt(4.25-2.0*x**2+2.0*x),
               lambda x:-sqrt(4.25-2.0*x**2+2.0*x))

xc = linspace(-1.0,2.0-1e-5)