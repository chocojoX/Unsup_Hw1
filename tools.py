import numpy as np
import os
import matplotlib.pyplot as plt
import re


def SVD(X):
    # Return, U, Sigma, V such that X = U.Sigma.V^T
    U, Sigma, V = np.linalg.svd(X)
    # Careful, np.linalg.svd return U, sigma, transpose(V) --> V need to be transposed.
    return U, np.diag(Sigma), np.transpose(V)


def inverse_SVD(U, Sigma, V):
    # Returns U.Sigma.V^T
    # inverse_SVD(SVD(X)) should be equal to X
    return np.dot(U, np.dot(Sigma, np.transpose(V)))


def get_all_conditions(individual):
    # Returns the list of all pgm files representing the individual if it exists (else, it should crash...)
    folder_path = "data/YaleB-Dataset/images"
    individual_path = folder_path + "/yaleB0" + str(individual)

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
            b"(^P5\s(?:\s*#.*[\r\n])*"
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
    individual_path = folder_path + "/yaleB0" + str(individual)
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
    print(all_images.shape)
    return all_images, width, height





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

    image = load_image(1, 2)
    plt.imshow(image, plt.cm.gray)
    plt.show()

    flat, width, height = flatten_picture(image)
    pict = unflatten_picture(flat, width, height)
    plt.imshow(pict, plt.cm.gray)
    plt.show()

    all_images, width, height = get_all_flat_pictures(1)
