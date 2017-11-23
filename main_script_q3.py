import numpy as np
import matplotlib.pyplot as plt
from Low_rank_completion import *
from tools import *

def question3():
    print('#'*50)
    print('Starting movie recommendation algorithm')
    print('#'*50)
    matrices = load_movie_ratings()
    # horror, romance, matrix_all_movies = matrices
    names = ['horror movies', 'romance movies', 'all movies']

    for i in range(3):
        print("Processing category : %s" %(names[i]))
        train, test, W, where_test = split_train_test_netflix(matrices[i], p_train=0.8)
        n_train = np.sum(W)
        average_rating = np.sum(train)/n_train
        D,N = train.shape
        beta = min(3,D*N/n_train)
        taus = []
        errors = []
        for tau in range(2, 30, 2):
            print("Tau = %i" % (tau))
            error = []
            for j in range(10):
                reconstructed = lrmc(train-average_rating*W, W, tau, beta)+average_rating
                # Convert the resulting matrix to integer type between 1 and 5
                reconstructed = (np.maximum(1, np.minimum(5, reconstructed+0.5))).astype(int)
                error.append(np.sqrt(np.sum((reconstructed*where_test - test*where_test)**2) / np.sum(where_test)))
            taus.append(tau);
            errors.append(np.mean(error))
        plt.plot(taus, errors, label = names[i])
    plt.legend(loc='best')
    plt.xlabel('tau')
    plt.ylabel('Root mean square error')
    plt.title('Reconstruction error as a function of Tau')
    plt.show()




if __name__=="__main__":
    question3()
