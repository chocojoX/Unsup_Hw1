import numpy as np
import matplotlib.pyplot as plt
from Low_rank_completion import *
from tools import *


def question2(individual = 1):
    print('#'*50)
    print('Starting image completion algorithm')
    print('#'*50)


    for p in [0, 0.2, 0.4]:
        print("Lauching tests for %i %% missing pixels" %(100*p))
        all_errors = []
        taus = [10, 100, 1000, 3000, 7500, 10000, 17500, 25000, 35000, 50000, 62500, 75000, 100000, 150000, 175000, 200000]
        for tau in taus:
            print("Tau = %i" %(tau))
            all_images, noisy_images, completed_images, w, h = run_test(individual, p, tau, False)
            error = compute_L2_error(all_images, completed_images)
            all_errors.append(np.sqrt(error/(all_images.shape[0]*all_images.shape[1])))

        plt.plot(taus, all_errors)
        plt.title("Evolution of the L2 reconstruction error with tau \n for p = %.1f (RMSE per pixel)" %p)
        plt.xlabel("Tau")
        plt.ylabel("Error")
        plt.show()

        best_tau = taus[np.argmin(all_errors)]
        print("Best result obtained for tau = %i" %best_tau)

    # Recomputing the lrmc for the best tau and displaying picture
    condition = 9
    p = 0.2
    best_tau = 75000
    print("Reconstruction with p=%.2f, tau = %i and only 10 images for training" %(p,best_tau))
    all_images, noisy_images, completed_images, width, height = run_test(individual, p, best_tau, False)
    plot_reconstruction(all_images, noisy_images, completed_images, condition, width, height)


    condition = 9
    p = 0.4
    best_tau = 35000
    print("Reconstruction with p=%.2f, tau = %i and only 10 images for training" %(p,best_tau))
    all_images, noisy_images, completed_images, width, height = run_test(individual, p, best_tau, False)
    plot_reconstruction(all_images, noisy_images, completed_images, condition, width, height)


    condition = 9
    p = 0.7
    best_tau = 75000
    print("Reconstruction with p=%.2f, tau = %i and all images for training" %(p,best_tau))
    all_images, noisy_images, completed_images, width, height = run_test(individual, p, best_tau, True)
    plot_reconstruction(all_images, noisy_images, completed_images, condition, width, height)


if __name__=="__main__":
    question2(individual = 1)
