def fcn(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + \
        90*(x[3]-x[2]**2)**2 + (1-x[2])**2 + \
        10.1*((x[1]-1)**2+(x[3]-1)**2) + \
        19.8*(x[1]-1)*(x[3]-1)