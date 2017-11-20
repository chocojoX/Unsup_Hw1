import numpy as np
import matplotlib.pyplot as plt
from Low_rank_completion import *
from tools import *


def question2():
    print('#'*50)
    print('Starting image completion algorithm')
    print('#'*50)
    individual = 1

    for p in [0, 0.2, 0.4]:
        print("Lauching tests for %i %% missing pixels" %(100*p))
        all_errors = []
        taus = [10, 100, 1000, 3000, 7500, 10000, 17500, 25000, 35000, 50000, 62500, 75000, 100000, 150000, 175000, 200000]
        for tau in taus:
            all_images, noisy_images, completed_images, w, h = run_test(individual, p, tau)
            error = compute_L2_error(all_images, completed_images)
            all_errors.append(np.sqrt(error/(all_images.shape[0]*all_images.shape[1])))

        plt.plot(taus, all_errors)
        plt.title("Evolution of the L2 reconstruction error with tau \n for p = %.1f (RMSE per pixel)" %p)
        plt.xlabel("Tau")
        plt.ylabel("Error")
        plt.show()

        best_tau = taus[np.argmin(all_errors)]
        print("Best result obtained for tau = %i" %best_tau)

        # Recomputing the lrmc for the best tau and displaying best and worst reconstructed picture
        all_images, noisy_images, completed_images, width, height = run_test(individual, p, best_tau)
        picture_errors = compute_columnwise_L2(all_images, completed_images)

        best_condition = np.argmin(picture_errors)
        worst_condition = np.argmax(picture_errors)

        plot_reconstruction(all_images, noisy_images, completed_images, best_condition, width, height, message="Best reconstructed face")
        plot_reconstruction(all_images, noisy_images, completed_images, worst_condition, width, height, message = "worst reconstructed face")


def question3():
    print('#'*50)
    print('Starting movie recommendation algorithm')
    print('#'*50)
    matrices = load_movie_ratings()
    # horror, romance, matrix_all_movies = matrices
    names = ['horror movies', 'romance movies', 'all movies']

    for i in range(3):
        train, test, W, where_test = split_train_test_netflix(matrices[i], p_train=0.8)
        n_train = np.sum(W)
        average_rating = np.sum(train)/n_train
        D,N = train.shape
        beta = min(3,D*N/n_train)
        taus = []
        errors = []
        for tau in range(10, 30, 2):
            reconstructed = lrmc(train-average_rating*W, W, tau, beta)+average_rating
            # Convert the resulting matrix to integer type between 1 and 5
            reconstructed = (np.maximum(1, np.minimum(5, reconstructed+0.5))).astype(int)
            error = np.sqrt(np.sum((reconstructed*where_test - test*where_test)**2) / np.sum(where_test))
            taus.append(tau); errors.append(error)
        plt.plot(taus, errors, label = names[i])
    plt.legend(loc='best')
    plt.xlabel('tau')
    plt.ylabel('Root mean square error')
    plt.title('Reconstruction error as a function of Tau')
    plt.show()




if __name__=="__main__":
    # question2()
    question3()
