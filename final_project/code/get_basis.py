from ad import gh
from ad.admath import log

def get_basis(my_base,dim):
    """Returns a set of basis functions for AM pursuit
    Usage
        Phi, dPhi, Labels = get_basis(my_base)
    Arguments
        my_base = integer selection
    Returns
        Phi    = scalar basis functions
        dPhi   = gradients of basis functions
        Labels = string label for each basis function
    """
    # Define basis functions
    if my_base == 1:
        # Second Order
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: log(abs(x[i])) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
    elif my_base == 2:
        # Third Order
        # Phi = [ lambda x: x[0],
        #         lambda x: x[1],
        #         lambda x: x[2],
        #         lambda x: x[0]**2,
        #         lambda x: x[1]**2,
        #         lambda x: x[2]**2,
        #         lambda x: x[0]**3,
        #         lambda x: x[1]**3,
        #         lambda x: x[2]**3,
        #         lambda x: log(abs(x[0])),
        #         lambda x: log(abs(x[1])),
        #         lambda x: log(abs(x[2])),
        #         lambda x: x[0]**(-1),
        #         lambda x: x[1]**(-1),
        #         lambda x: x[2]**(-1),]
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: x[i]**3 for i in range(dim)] + \
              [lambda x, i=i: log(abs(x[i])) for i in range(dim)] + \
              [lambda x, i=i: x[i]**(-1) for i in range(dim)]
    else:
        # Active Subspace
        Phi = []
        Phi = Phi + [lambda x: x[i] for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]

    Labels = [] # DEBUG
    return Phi, dPhi, Labels

if __name__ == "__main__":
    import numpy as np
    m = 3

    p1,d1,l1 = get_basis(1,m)
    p2,d2,l2 = get_basis(1.5,m)

    x = [1,2,3]

    print( np.array(d1[0](x)) - np.array(d2[0](x)) )
    print( np.array(d1[1](x)) - np.array(d2[1](x)) )
    print( np.array(d1[2](x)) - np.array(d2[2](x)) )
    print( np.array(d1[3](x)) - np.array(d2[3](x)) )
    print( np.array(d1[4](x)) - np.array(d2[4](x)) )
    print( np.array(d1[5](x)) - np.array(d2[5](x)) )
    print( np.array(d1[6](x)) - np.array(d2[6](x)) )
    print( np.array(d1[7](x)) - np.array(d2[7](x)) )
    print( np.array(d1[8](x)) - np.array(d2[8](x)) )
