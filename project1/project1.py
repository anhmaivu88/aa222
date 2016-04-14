# from qnewton_ad import qnewton # using automatic differentiation
from qnewton import qnewton # using finite differences

def unconstrained_optimization(f,x0,maxEval):
    # Call the solver
    xs,fs,ct,Xs,it = qnewton(f,x0,maxEval)

    # Debug printout
    print "f(xs)=%f" % fs
    print "calls=%d" % ct
    print "iter=%d" % it
    
    # Return the optimal value
    return fs

def opt_full(f,x0,maxEval):
    # Call the solver
    xs,fs,ct,Xs,it = qnewton(f,x0,maxEval)

    # Return the optimal value
    return xs,fs,ct,Xs,it
