import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    ### Setup
    from rosenbrock import fcn
    # from simple_quad import fcn

    ### Test gradient
    x0 = [1, 1]; f0 = fcn(x0); g0 = grad(x0,fcn,f0); e0=x0-g0*0.1
    x1 = [0, 1]; f1 = fcn(x1); g1 = grad(x1,fcn,f1); e1=x1-g1*0.1
    x2 = [0,-1]; f2 = fcn(x2); g2 = grad(x2,fcn,f2); e2=x2-g2*0.1

    ### Plotting
    # Define meshgrid
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    dim = np.shape(X)
    # Compute function values
    Xv = X.flatten(); Yv=Y.flatten()
    Input = zip(Xv,Yv)
    Zv = []
    for x in Input:
        Zv.append(fcn(x))
    # Restore arrays to proper dimensions
    Z = np.array(Zv).reshape(dim)
    
    # Open figure
    plt.figure()

    # Plot contour
    CS = plt.contour(X, Y, Z)
    manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
    plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
    plt.title('Sequence of iterates')

    # Plot the gradient
    plt.plot(x0[0],x0[1],'.'); plt.plot([x0[0],e0[0]],[x0[1],e0[1]],'-')
    plt.plot(x1[0],x1[1],'.'); plt.plot([x1[0],e1[0]],[x1[1],e1[1]],'-')
    plt.plot(x2[0],x2[1],'.'); plt.plot([x2[0],e2[0]],[x2[1],e2[1]],'-')
    # plt.arrow(x0[0],x0[1],e0[0],e0[1],head_width=0.05,head_length=0.1)
    # plt.arrow(x1[0],x1[1],e1[0],e1[1],head_width=0.05,head_length=0.1)
    # plt.arrow(x2[0],x2[1],e2[0],e2[1],head_width=0.05,head_length=0.1)


    plt.show()