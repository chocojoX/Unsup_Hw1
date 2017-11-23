import numpy as np
from tools import *
import copy


def P_Omega(X, W):
    assert(np.sum(np.abs(W * (1-W))) == 0) #"Omega should be composed only of zeros and ones"
    return W * X


def D_tau(X, tau):

    U, Sigma, V = SVD(X)

    Sigma = threshold_shrinkage(Sigma, tau)

    return inverse_SVD(U, Sigma, V)


def lrmc(X, W, tau, beta):
    Z = P_Omega(X, W)
    A = copy.copy(X)
    EPS = 0.03 * X.shape[0] * X.shape[1]
    dist = EPS + 1
    nb_iter = 0
    while dist>EPS and dist < 10**13:
        A_old= np.copy(A)
        A = D_tau(P_Omega(Z, W), tau)
        Z = Z + beta * (P_Omega(X-A,W))

        dist = np.sum(np.abs(A-A_old))
        nb_iter += 1
    print("nb_iter : %i" % (nb_iter))
    return A


def run_test(individual, p, tau):
    #Loading images, deleting part of each
    all_images, width, height = get_all_flat_pictures(individual)
    all_images = all_images[:10,:]
    noisy_images = remove_values(all_images, p=p)

    #W = (noisy_images != 0).astype(int)
    W = (noisy_images == all_images).astype(int)
    M = np.sum(W)
    D,N = all_images.shape
    beta = min(2,D*N/M)

    completed_images = lrmc(noisy_images, W, tau, beta)
    return all_images, noisy_images, completed_images, width, height


if __name__=="__main__":
    condition = 12
    individual = 1
    p = 0.2
    tau = 40000

    all_images, noisy_images, completed_images, width, height = run_test(individual, p, tau)
    plot_reconstruction(all_images, noisy_images, completed_images, condition, width, height)
