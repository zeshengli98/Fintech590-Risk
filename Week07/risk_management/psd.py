import numpy as np
import pandas as pd

"""
For this method, we can apply near_psd(A) and Higham_near_psd(A) to find the nearest PSD matrix
"""
def chol_psd(A):
    """Performs a Cholesky decomposition of a matrix, the matrix
    should be a symmetric and PD matrix.
    return: the lower triangle matrix."""
    n = len(A)

    # Create zero matrix for L
    L = np.array([[0.0] * n for _ in range(n)])

    # Perform the Cholesky decomposition
    for j in range(n):
        s = L[j,:j]@L[j,:j].T
        #Diagonal Element
        temp = A[j,j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0

        elif temp < -1e-8:
            raise ValueError("The matrix is non-PSD!")
        L[j,j] = np.sqrt(temp);
        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if L[j,j] == 0:
            continue


        #non-Diagonal Element
        ir = 1.0/L[j,j]
        for i in range(j+1,n):
            s = L[i,:j]@L[j,:j]
            L[i,j] =(A[i,j] - s)*ir
    return L


def near_psd(A):
    """Find the nearest PSD matrix with Rebonato and Jackel method"""
    eig_val, eig_vec = np.linalg.eigh(A)
    # construct a new Λ that all eigen values >= 0
    eig_val[eig_val < 0] = 0.0
    t = 1 / (eig_vec ** 2 @ eig_val)

    # construct the diagonal scaling matrix
    T_sqrt = np.diagflat(np.sqrt(t))
    la_sqrt = np.diagflat(np.sqrt(eig_val))

    # Let B = sqrt(T) * S * sqrt(Λ)
    B = T_sqrt @ eig_vec @ la_sqrt
    # the nearest matrix is C = B*B'
    return B @ B.T


def Frobenius_norm(A):
    n = len(A)
    s = 0
    for i in range(n):
        for j in range(n):
            s+=A[i,j]**2
    return s

def proj_u(A):
    corr_ = A.copy()
    np.fill_diagonal(corr_,1)
    return corr_

def proj_s(A):
    eig_val, eig_vec = np.linalg.eigh(A)
    eig_val[eig_val<0] = 0
    p = eig_vec@ np.diagflat(eig_val)@eig_vec.T
    return p


def Higham_near_psd(A):
    #∆𝑆0 = 0, 𝑌0 = 𝐴, γ0 = 𝑚𝑎𝑥 𝐹𝑙𝑜𝑎𝑡
    dS = 0
    Y = A
    last_gamma = float("inf")
    iteration = 100000
    tol = 1e-10
    #𝐿𝑜𝑜𝑝 𝑘 ∈ 1... 𝑚𝑎𝑥 𝐼𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠
    for i in range(iteration):
        R = Y - dS                       #𝑅𝑘 = 𝑌𝑘−1 − ∆𝑆𝑘−1
        X = proj_s(R)                    #𝑋𝑘 = 𝑃𝑆(𝑅𝑘)
        dS = X - R                       #∆𝑆𝑘 = 𝑋𝑘 − 𝑅𝑘
        Y = proj_u(X)                    #𝑌𝑘 = 𝑃𝑈(𝑋𝑘)
        gamma = Frobenius_norm(Y - A)    #γ𝑘 = γ(𝑌𝑘)
        if abs(gamma-last_gamma)< tol:   #𝑖𝑓 γ𝑘−1 − γ𝑘 < 𝑡𝑜𝑙 𝑡ℎ𝑒𝑛 𝑏𝑟𝑒𝑎𝑘
            break
        last_gamma = gamma
    return Y

