# Necessary imports
from scipy.linalg import svd
from scipy import compress, transpose
from numpy.linalg import norm
from numpy import dot
from numpy import atleast_2d
from numpy import ravel
from numpy import reshape
from numpy import zeros, shape, nonzero
from copy import copy
from math import log
from numpy import diag

# Nullspace computation
# from scipy.linalg import svd
# from scipy import compress, transpose
def null(A, eps=1e-15):
    u, s, vh = svd(A)
    null_mask = (s <= eps)
    null_space = compress(null_mask, vh, axis=0)
    return transpose(null_space)

# Subspace distance
# from numpy.linalg import norm
# from numpy import dot
def subspace_distance(W1,W2):
    """Computes the subspace distance
    Note that provided matrices must 
    have orthonormal columns
    Usage
        res = subspace_distance(W1,W2)
    Inputs
        W1 = orthogonal matrix
        W2 = orthogonal matrix
    Outputs
        res = subspace distance
    """
    return norm(dot(W1,W1.T)-dot(W2,W2.T),ord=2)

# Keep 2D
# from numpy import atleast_2d
def col(M):
    """Returns column vectors
    """
    res = atleast_2d(M)
    # We're not ok with row vectors!
    # Transpose back to a column vector
    if res.shape[0] == 1:
        res = res.T
    return res

# Vectorize a matrix
# from numpy import ravel
def vec(A):
    return col(ravel(A))

# Unvectorize a matrix
# from numpy import reshape
def unvec(v,n):
    """Unvectorizes an array
    Usage
        M = unvec(v,n)
    Inputs
        v = vector of length l=n*m
        n = number of rows for M
    Outputs
        M = 
    """
    return reshape(v,(n,-1))

# Active Subspace Dimension
# from copy import copy
def as_dim(Lam,eps=2.5):
    """Finds the dimension of the 
    most accurate Active Subspace
    Usage
        dim = as_dim(Lam)
    Arguments
        Lam = positive Eigenvalues, sorted by decreasing magnitude
    Keyword Arguments
        eps = order of magnitude gap needed for AS
    """
    # Normalize by eigenvalue energy
    s = sum(Lam)
    L = [l/s for l in Lam]
    # Check eigenvalue gaps
    G = [L[i]-L[i+1] for i in range(len(L)-1)]
    Gp= [L[i]/L[i+1] for i in range(len(L)-1)]
    # If no gap exceeds eps, full dimensional
    gap = max(G)
    ind = G.index(gap)
    p   = log(Gp[ind])
    # print("pow={}".format(p))
    if p < eps:
        return len(L)
    # Return dimension
    else:
        return G.index(gap)+1

# Normalize the columns of a matrix
# from numpy import diag
def norm_col(M):
    E = diag( [1/norm(M[:,i]) for i in range(M.shape[1])] )
    return M.dot(E)

# Rounds by smallest non-zero magnitude vector element
# from numpy import zeros, shape, nonzero
def round_out(M):
    C = zeros(shape(M))
    for i in range(shape(M)[1]):
        c = min(min(M[nonzero(M[:,i]),i]))
        C[:,i] = M[:,i] / c
    return C

# Test code
if __name__ == "__main__":
    import numpy as np

    ### Test vec and unvec
    # A = np.arange(9); A.shape=(3,3)
    # v = vec(A)
    # M = unvec(v,3)
    # print(M)

    ### Test as_dim
    # Lam1 = [1,0.9] # Should be 2D
    # Lam2 = [1e3,1e1,0.5e1,1e0] # should be 1D
    # Lam3 = [1e3,0.9e3,1e1,1e0] # should be 2D
    # Lam4 = [1e3,0.9e3,0.8e3,1e0] # should be 3D
    # Lam5 = [1e3,0.9e3,0.8e3,0.7e3] # should be 4D
    
    # print("AS 1 dim={}".format(as_dim(Lam1)))
    # print("AS 2 dim={}".format(as_dim(Lam2)))
    # print("AS 3 dim={}".format(as_dim(Lam3)))
    # print("AS 4 dim={}".format(as_dim(Lam4)))
    # print("AS 5 dim={}".format(as_dim(Lam5)))

    ### Test subspace_distance
    # M = np.

    ### Test column normalization
    M = np.reshape(np.arange(9),(3,3))
    Mn= norm_col(M)
    print( Mn )
