import numpy as np
from tools import *


def P_Omega(X, W):
    assert(np.sum(np.abs(W * (1-W))) == 0) #"Omega should be composed only of zeros and ones"
    return W * X


def D_tau(X, tau):

    U, Sigma, V = SVD(X)

    Sigma = threshold_shrinkage(Sigma, tau)

    return inverse_SVD(U, Sigma, V)


def lrmc(X,W,tau,beta):
    Z = P_Omega(X, W)
    A = np.zeros_like(X)
    EPS = 0.01
    dist = EPS + 1

    while dist>EPS:
        A_old= np.copy(A)
        A = D_tau(Z, tau)
        Z = Z + beta * (P_Omega(X-A,W))
        dist = np.sum(np.abs(A-A_old))

    return A



if __name__=="__main__":
    tau = 25
    beta = 0.3


    all_images, width, height = get_all_flat_pictures(1)
    all_images = all_images[:30,:]
    image = all_images[0,:]
    image = unflatten_picture(image, width, height)
    noisy_images = remove_values(all_images, p=0.2)
    noisy_image = noisy_images[0, :]
    noisy_image = unflatten_picture(noisy_image, width, height)


    W = (noisy_images != 0).astype(int)
    completed_images = lrmc(noisy_images, W, tau, beta)

    completed_image = completed_images[0, :]
    completed_image = unflatten_picture(completed_image, width, height)


    plt.subplot(1,3,1)
    plt.imshow(image, plt.cm.gray)
    plt.title("Original Image")

    plt.subplot(1,3,2)
    plt.imshow(noisy_image, plt.cm.gray)
    plt.title("Partially Destroyed Image")
    
    plt.subplot(1,3,3)
    plt.imshow(completed_image, plt.cm.gray)
    plt.title("Reconstructed Image")
    plt.show()





