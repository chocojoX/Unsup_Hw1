import numpy as np


def SVD(X):
    # Return, U, Sigma, V such that X = U.Sigma.V^T
    U, Sigma, V = np.linalg.svd(X)
    # Careful, np.linalg.svd return U, sigma, transpose(V) --> V need to be transposed.
    return U, np.diag(Sigma), np.transpose(V)


def inverse_SVD(U, Sigma, V):
    # Returns U.Sigma.V^T
    # inverse_SVD(SVD(X)) should be equal to X
    return np.dot(U, np.dot(Sigma, np.transpose(V)))


if __name__=="__main__":
    """ Use this main function only to debug """

    ### Testing SVD
    X = np.array([[15,1,1], [1,20,1], [1,1,25]])
    print("X = ")
    print(X)
    U, Sigma, V = SVD(X)
    print("Sigma_X = ")
    print(Sigma)
    print("inverse_SVD(SVD(X)) = ")
    print(inverse_SVD(U, Sigma, V))
