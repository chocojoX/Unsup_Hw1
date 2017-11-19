import numpy as np
import os
import matplotlib.pyplot as plt
import re


def SVD(X):
    # Return, U, Sigma, V such that X = U.Sigma.V^T
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    # Careful, np.linalg.svd return U, sigma, transpose(V) --> V need to be transposed.
    return U, np.diag(Sigma), np.transpose(V)


def inverse_SVD(U, Sigma, V):
    # Returns U.Sigma.V^T
    # inverse_SVD(SVD(X)) should be equal to X
    temp = np.dot(Sigma, np.transpose(V))
    return np.dot(U, temp)


def get_all_conditions(individual):
    # Returns the list of all pgm files representing the individual if it exists (else, it should crash...)
    folder_path = "data/YaleB-Dataset/images"
    if individual < 4:
    	individual_path = folder_path + "/yaleB0" + str(individual)
    else:
    	individual_path = folder_path + "/outliers"

    files = os.listdir( individual_path )
    files = [f for f in files if '.pgm' in f]
    return files


def read_pgm(filename, byteorder='>'):
    # Return image data from a raw PGM file as numpy array.
    # Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    # Credits : 'cgohlke' on https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P.\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def load_image(individual, condition):
    # Load the image number 'condition' for the individual and returns it.
    folder_path = "data/YaleB-Dataset/images"
    if individual <4:
    	individual_path = folder_path + "/yaleB0" + str(individual)
    else:
    	individual_path = folder_path + "/outliers"
    files = get_all_conditions(individual)

    # assume condition<len(files), "Condition number too high, please choose another one (smaller than %i)" %(len(files))
    file_name = individual_path + "/" + files[condition]
    image = read_pgm(file_name)
    return image


def threshold_shrinkage(X, tau):
    X[np.abs(X) <= tau] = 0
    X[X >= tau] -= tau
    X[X <= -tau] += tau
    return X


def flatten_picture(pict):
    return pict.ravel(), pict.shape[0], pict.shape[1]


def unflatten_picture(flat_pict, width, height):
    # unflatten_picture(flatten_picture(pict)) = pict
    return flat_pict.reshape(width, height)


def get_all_flat_pictures(individual):
    # Returns a matrix which lines are the flatten pictures of all individual
    files = get_all_conditions(individual)
    n = len(files)
    all_images = []
    for i in range(n):
        image = load_image(individual, i)
        image, width, height = flatten_picture(image)
        all_images.append(image)

    all_images = np.array(all_images)
    return all_images, width, height


def remove_values(X, p=0.2):
    # removes entries of X with probability p
    Omega = np.random.rand(X.shape[0], X.shape[1])
    Omega[Omega<p] = 0
    Omega[Omega>=p] = 1
    return X * Omega


def compute_L2_error(X, X_star):
    return np.sum((X - X_star)**2)


def compute_columnwise_L2(X, X_star):
    return np.sum((X - X_star), axis = 1)


def plot_reconstruction(all_images, noisy_images, completed_images, condition, width, height, message=None):
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
    completed_image = np.maximum(0, np.minimum(completed_image, 255))
    plt.imshow(completed_image, plt.cm.gray)
    if message is None:
        plt.title("Reconstructed Image")
    else:
        plt.title(message)
    plt.show()


def load_movie_ratings():
    with open('data/romance_horror.txt', 'r') as f:
        lines = f.read()

    # Let's build the rating matrix !
    users_cpt = 0; genre_1_cpt = 0; genre_2_cpt = 0; movies_cpt = 0
    # Create dictionaries to create new ids
    users = {}
    genre_1 = {}
    genre_2 = {}
    movies = {}

    lines = lines.split('\n')
    for l in lines[1:]:
        l = l.split(',')
        user = l[0]
        genre = l[1]
        movie = l[2]
        if user not in users.keys():
            users[user] = users_cpt
            users_cpt += 1
        if genre == '1':
            if movie not in genre_1.keys():
                genre_1[movie] = genre_1_cpt
                movies[movie] = movies_cpt
                genre_1_cpt += 1
                movies_cpt += 1
        if genre == '2':
            if movie not in genre_2.keys():
                genre_2[movie] = genre_2_cpt
                genre_2_cpt += 1
                movies[movie] = movies_cpt
                movies_cpt += 1

    matrix_genre1 = np.zeros((users_cpt, genre_1_cpt))
    matrix_genre2 = np.zeros((users_cpt, genre_2_cpt))
    matrix_all_movies = np.zeros((users_cpt, movies_cpt))

    # Now, let's get the data to the matrices !

    for l in lines[1:]:
        l = l.split(',')
        user_id = users[l[0]]
        genre = l[1]
        rating = l[3]
        if genre == '1':
            genre1_id = genre_1[l[2]]
            movie_id = movies[l[2]]
            matrix_genre1[user_id, genre1_id] = float(rating)
            matrix_all_movies[user_id, movie_id] = float(rating)
        if genre == '2':
            genre2_id = genre_2[l[2]]
            movie_id = movies[l[2]]
            matrix_genre2[user_id, genre2_id] = float(rating)
            matrix_all_movies[user_id, movie_id] = float(rating)

    return matrix_genre1, matrix_genre2, matrix_all_movies


def split_train_test_netflix(data, p_train=0.8):
    W = np.sign(data)
    p_test = 1-p_train
    Omega = np.random.rand(data.shape[0], data.shape[1])
    Omega[Omega<p_test] = 0
    Omega[Omega>=p_test] = 1

    train = data * W * Omega   # No need for W multiplication but it was added for clarity
    test = data * W * (1 - Omega)

    where_train = W * Omega
    where_test = W * (1-Omega)
    return train, test, where_train, where_test
    # where_train is equal to 1 where train values are non zero, where_test = 1 where test is equal to 1.





if __name__=="__main__":
    """ Use this main function only to debug """

    horror, romance, matrix_all_movies = load_movie_ratings()
    train, test, where_train, where_test = split_train_test_netflix(matrix_all_movies)


    ### Testing SVD
    X = np.array([[15,1,1], [1,20,1], [1,1,25]])
    print("X = ")
    print(X)
    U, Sigma, V = SVD(X)
    print("Sigma_X = ")
    print(Sigma)
    print("inverse_SVD(SVD(X)) = ")
    print(inverse_SVD(U, Sigma, V))

    # Testing loading
    image = load_image(2, 1)
    plt.imshow(image, plt.cm.gray)
    plt.show()

    # Testing Flatten and unflatten
    flat, width, height = flatten_picture(image)
    pict = unflatten_picture(flat, width, height)
    plt.imshow(pict, plt.cm.gray)
    plt.show()

    # Testing get_all_flat_pictures
    all_images, width, height = get_all_flat_pictures(1)
    noisy_images = remove_values(all_images, p=0.4)
    pict = noisy_images[0, :]
    pict = unflatten_picture(pict, width, height)
    plt.imshow(pict, plt.cm.gray)
    plt.show()
