from ad import gh
from ad.admath import log

def get_basis(my_base,dim):
    """Returns a set of basis functions for AM pursuit
    Usage
        Phi, dPhi, name, Labels = get_basis(my_base)
    Arguments
        my_base = integer selection
    Returns
        Phi    = scalar basis functions
        dPhi   = gradients of basis functions
        name   = name of basis type
        Labels = string label for each basis function
    """
    # Define basis functions
    if my_base == 1:
        # Second Order
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: log(abs(x[i])) for i in range(dim)]
        # Labels
        Labels = ["L_"+str(i) for i in range(dim)] + \
                 ["Q_"+str(i) for i in range(dim)] + \
                 ["G_"+str(i) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Name
        name = "Second-Order"
    elif my_base == 2:
        # Third Order
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: x[i]**3 for i in range(dim)] + \
              [lambda x, i=i: log(abs(x[i])) for i in range(dim)] + \
              [lambda x, i=i: x[i]**(-1) for i in range(dim)]
        # Labels
        Labels = ["L_"+str(i) for i in range(dim)] + \
                 ["Q_"+str(i) for i in range(dim)] + \
                 ["C_"+str(i) for i in range(dim)] + \
                 ["G_"+str(i) for i in range(dim)] + \
                 ["I_"+str(i) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Name
        name = "Third-Order"
    elif my_base == 3:
        # 2nd Order Legendre Basis
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: 0.5*x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: 0.5*(x[i]**3-x[i]) for i in range(dim)]
        # Labels
        Labels = ["P_0_"+str(i) for i in range(dim)] + \
                 ["P_1_"+str(i) for i in range(dim)] + \
                 ["P_2_"+str(i) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Name
        name = "Legendre Second-Order"
    elif my_base == 4:
        # 3nd Order Legendre Basis
        Phi = [lambda x, i=i: x[i] for i in range(dim)] + \
              [lambda x, i=i: 0.5*x[i]**2 for i in range(dim)] + \
              [lambda x, i=i: 0.5*(x[i]**3-x[i]) for i in range(dim)] + \
              [lambda x, i=i: 0.5*(5./4.*x[i]**4-3./2.*x[i]**2) for i in range(dim)]
        # Labels
        Labels = ["P_0_"+str(i) for i in range(dim)] + \
                 ["P_1_"+str(i) for i in range(dim)] + \
                 ["P_2_"+str(i) for i in range(dim)] + \
                 ["P_3_"+str(i) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Name
        name = "Legendre Third-Order"
    else:
        # Active Subspace
        Phi = [lambda x: x[i] for i in range(dim)]
        # Labels
        Labels = ["L_"+str(i) for i in range(dim)]
        # Gradients 
        dPhi = [gh(f)[0] for f in Phi]
        # Name
        name = "Linear"

    return Phi, dPhi, name, Labels
