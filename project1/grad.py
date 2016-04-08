import numpy as np

def grad(x,f,f0=None,h=1e-8):
    # If necessary, calculate f(x)
    if f0:
        pass
    else:
        f0 = f(x)

    # Ensure query point is np.array
    if type(x).__module__ == np.__name__:
        pass
    else:
        x = np.array(x)

    # Calculate gradient
    dF = np.empty(np.size(x)) # Reserve space
    E  = np.eye(np.size(x))   # Standard basis
    for i in range(np.size(x)):
        dF[i] = (f(x+E[:,i]*h)-f0)/h

    return dF