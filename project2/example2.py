### Example problem #2

def f(x):
    return 4*x[0]**2 - x[0] - x[1] - 2.5

def g(x):
    return (-x[1]**2+1.5x[0]+1, \
            x[1]**2+2*x[0]**2-2*x[0]-4.25)

x_star = (0.164439, 2.12716)
f_star = -4.68344