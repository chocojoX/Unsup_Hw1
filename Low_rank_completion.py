import numpy as np
from tools import *


def P_Omega(X, W):
    assert(np.sum(np.abs(W * (1-W))) == 0) #"Omega should be composed only of zeros and ones"
    return W * X


def D_tau(X, tau):

    U, Sigma, V = SVD(X)

    Sigma = threshold_shrinkage(Sigma, tau)

    return inverse_SVD(U, Sigma, V)


def lrmc(X, W, tau, beta):
    Z = P_Omega(X, W)
    A = X
    EPS = 0.001 * X.shape[0] * X.shape[1]
    dist = EPS + 1

    while dist>EPS:
        A_old= np.copy(A)
        A = D_tau(Z, tau)
        Z = Z + beta * (P_Omega(X-A,W))
        dist = np.sum(np.abs(A-A_old))
    return A


def run_test(individual, p, tau):
    #Loading images, deleting part of each
    all_images, width, height = get_all_flat_pictures(individual)
    all_images = all_images[:30,:]
    noisy_images = remove_values(all_images, p=p)

    W = (noisy_images != 0).astype(int)
    M = np.sum(W)
    D,N = all_images.shape
    beta = min(2,D*N/M)

    completed_images = lrmc(noisy_images, W, tau, beta)
    return all_images, noisy_images, completed_images, width, height



if __name__=="__main__":
    condition = 2
    individual = 1
    p=0.2
    tau = 40000

    all_images, noisy_images, completed_images, width, height = run_test(individual, p)

    image = all_images[condition,:]
    image = unflatten_picture(image, width, height)
    noisy_image = noisy_images[condition, :]
    noisy_image = unflatten_picture(noisy_image, width, height)
    completed_image = completed_images[condition, :]
    completed_image = unflatten_picture(completed_image, width, height)

    print(completed_image - noisy_image)
    plt.subplot(1,3,1)
    plt.imshow(image, plt.cm.gray)
    plt.title("Original Image")

    plt.subplot(1,3,2)
    plt.imshow(noisy_image, plt.cm.gray)
    plt.title("Partially Destroyed Image")

    plt.subplot(1,3,3)
    plt.imshow(completed_image +254/2, plt.cm.gray)
    plt.title("Reconstructed Image")
    plt.show()
