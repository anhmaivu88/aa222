# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

# Custom libraries
from unconstrained_optimization import opt_full

### ----- CONVERGENCE ----- ###
### Setup
## Rosenbrock function
# from rosenbrock import fcn
# xstar = [1,1]; x0 = [-0.5,-0.5]; title="Rosenbrock"
## Wood function
from wood import fcn
xstar = [1,1,1,1]; x0 = [0,0,0,0]; title="Wood"
## Powell function
# from powell import fcn
# xstar = [0,0,0,0]; x0 = [1,1,1,1]; title="Powell"

# Set parameters
Calls = [1,2,5,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4]
Error = []

# Generate convergence data
for c in Calls:
    ### Solver call
    xs0,_,_,_,_ = opt_full(fcn,x0,c)
    # Error calculation
    err = norm(xs0-xstar)
    Error.append(err)

### Plotting
# Open figure
plt.figure()

# Plot convergence history
plt.title('Error vs Evaluation Limit: '+title)
plt.loglog(Calls,Error,'.-b')
plt.xlabel('Evaluation Limit')
plt.ylabel('Error')

plt.show()
