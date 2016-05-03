# Standard libraries
import numpy as np
# Custom libraries
from qnewton import qnewton # quasi-Newton method using finite differences
from barrier import feasibility_problem, log_barrier, inv_barrier

### TODO:
# Interior Point Method:
# - Solve feasibility problem
# - Run interior point method

# - Create driver function which does interior point
# method using quasi-Newton -- this only works for feasible points!
# - Add exterior point driver to get a feasible point
# - Modify qNewton to take initial function value, gradient,
# and approximate hessian to save evaluations

def constrained_optimization(f,g,x0,evalMax):
    """ Constrained Optimization via interior point method
    Usage
        xs = constrained_optimization(f,g,x0,evalMax)
    Inputs
        f       = objective function, R^n->R
        g       = leq constraints, R^n->R^k
        x0      = initial guess
        evalMax = maximum function evaluations
    Outputs
        xs      = optimal point
    """
    xs, fk, ct, X, it = opt_full(f,g,x0,evalMax)
    return xs

def opt_full(f,g,x0,evalMax):
    """ Constrained Optimization via interior point method
    Usage
        xs, fs, ct, X, it = opt_full(f,g,x0,evalMax)
    Inputs
        f       = objective function, R^n->R
        g       = leq constraints, R^n->R^k
        x0      = initial guess
        evalMax = maximum function evaluations
    Outputs
        xs      = optimal point
        fs      = optimal value
        ct      = function evaluations
        X       = sequence of iterates
        it      = iterations
    """
    ### Setup
    r   = 1e2   # Initial relaxation
    r_max = 1e3
    fac = 2    # Relaxation factor
    eps = 1/r  # Initial gradient tolerance

    ct  = 0  # Evaluation count
    it  = 0  # Iteration count
    s   = 1e-1 # Slack
    x0  = np.array(x0)   # initial guess
    n   = np.size(x0)    # dim of problem
    Hk  = np.eye(n)
    fk  = None
    dF0 = None
    ### Feasibility problem
    xf, gf, ct_f, X, it_f = feasibility_problem(g,x0,evalMax,slack=s)
    ct = ct_f; it = it_f
    xs = xf

    ### Interior point problem sequence
    while (ct<evalMax):
        # Relax the barrier
        fcn = lambda x: f(x) + log_barrier(g(x))/r
        # Enforce a tighter convergence criterion
        xs,fk,dF0,Hk,ct_s,Xs,it_s = qnewton(fcn,xs,evalMax-ct,eps=1/r,
                                            fk=fk,dF0=dF0,Hk=Hk)
        ct = ct + ct_s; it = it + it_s
        X = np.append(X,Xs,axis=0)
        # Increment to next problem
        if r < r_max:
            r   = r * fac
            eps = 1 / r
        else:
            r   = r_max
            eps = eps=np.finfo(float).eps

    ### Terminate
    return xs, fk, ct, X, it
