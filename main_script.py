import numpy as np
import matplotlib.pyplot as plt
from Low_rank_completion import *
from tools import *


def question2():
    individual = 1
    p = 0.4
    all_errors = []
    taus = [10, 100, 1000, 3000, 7500, 10000, 17500, 25000, 35000, 50000, 62500, 75000, 100000, 150000, 175000, 200000]
    for tau in taus:
        all_images, noisy_images, completed_images, w, h = run_test(individual, p, tau)
        error = compute_L2_error(all_images, completed_images)
        all_errors.append(error)

    plt.plot(taus, all_errors)
    plt.title("Evolution of the L2 reconstruction error with tau")
    plt.xlabel("Tau")
    plt.ylabel("Error")
    plt.show()

    best_tau = taus[np.argmin(all_errors)]
    print("Best result obtained for tau = %i" %best_tau)
    condition = 12

    all_images, noisy_images, completed_images, width, height = run_test(individual, p, best_tau)

    image = all_images[condition,:]
    image = unflatten_picture(image, width, height)
    noisy_image = noisy_images[condition, :]
    noisy_image = unflatten_picture(noisy_image, width, height)
    completed_image = completed_images[condition, :]
    completed_image = unflatten_picture(completed_image, width, height)

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



if __name__=="__main__":
    question2()
