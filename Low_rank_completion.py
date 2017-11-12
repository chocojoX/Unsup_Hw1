import numpy as np
import copy
from tools import *


def P_Omega(X, W):
    assume np.sum(np.abs(W * (1-W))) == 0, "Omeaga should be composed only of zeros and ones"
    return W * X

def S_tau(Sigma, tau):
    for i in range(Sigma.shape[0]):
        val = Sigma[i, i]
        if val > tau:
            Sigma[i, i] = val-tau
        elif val<-tau:
            Sigma[i, i] = val+tau
        else:
            Sigma[i, i] = 0
    return Sigma


def D_tau(X, tau):
    U, Sigma, V = SVD(X)
    Sigma = S_tau(Sigma, tau)
    return inverse_SVD(U, Sigma, V)


def lrmc(X,W,tau,beta):
    Z = P_Omega(X, W)
    A = np.zeros_like(X)
    EPS = 0.01
    dist = EPS + 1

    while dist>EPS:
        A_old= copy.copy(A)
        A = D_tau(Z, tau)
        Z = Z + beta * (P_Omega(X-A))
        dist = np.sum(np.abs(A-A_old))

    return A
