"""
PCA - Principal Components Analysis
http://www.wikizmsi.zut.edu.pl/wiki/ADIUM/L/z1

Stores PCA matrix in oliv_pca.p
If file does not exists then calculates PCA
"""

from sklearn import datasets
import numpy as np
import time
import pickle
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def pca(data, components=None):
    """
    Perform Principal Components Analysis
    :param data: source data for PCA
    :return: covariance, correlation, l matrix, v real matrix
    """
    start_time = time.time()
    cov = np.cov(data, rowvar=False)  # przyklady ulozone wierszami
    elapsed_time = time.time() - start_time
    print("time cov:", elapsed_time)

    start_time = time.time()
    cor = np.corrcoef(data, rowvar=False)  # przyklady ulozone wierszami
    elapsed_time = time.time() - start_time
    print("time cor:", elapsed_time)

    # L - wartosci wlasne lambdy - wariancje wzgledem kierunkow
    # V - wektory wlasne - kolumnami
    start_time = time.time()
    l, v = np.linalg.eig(cov)
    L = np.real(l)
    V = np.real(v)
    ordering = np.argsort(-L)
    L = L[ordering]
    V = V[:, ordering]

    if components is not None:
        L = L[:components]
        V = V[:, :components]

    elapsed_time = time.time() - start_time
    print("time eig:", elapsed_time)
    return L, V


def pickle_all(some_list, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    f = open(fname, 'wb')
    pickle.dump(some_list, f)
    f.close()


def unpickle_all(fname):
    f = open(fname, 'rb')
    some_list = pickle.load(f)
    f.close()
    return some_list


def show_some_images(images, indexes=None, as_grid=True, title=None):
    print('SHOW SOME IMAGES')
    t1 = time.time()
    if indexes is None:
        indexes = range(len(images))
    shape = images[0].shape
    if len(shape) == 1: # flattened images
        img_side = int(np.sqrt(shape))
        images = images.reshape(images.shape[0], img_side, img_side)
    fig = plt.figure()
    plt.gray()
    if title is not None:
        fig.canvas.set_window_title(title)
    grid = int(np.ceil(np.sqrt(len(indexes))))
    for i, index in enumerate(indexes):
        if as_grid:
            plt.subplot(grid, grid, i+1)
        else:
            plt.subplot(1, len(indexes), i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[index])
    t2 = time.time()
    print('SHOW SOME IMAGES DONE. [TIME: ' + str(t2 - t1) + ' s.]')
    plt.show()


def print_image(img):
    plt.gray()
    plt.grid(False)
    plt.imshow(img)
    plt.show()


def load_pca_or_generate(data):
    file_exists = os.path.isfile('./data/oliv_pca.p')

    if file_exists:
        with open('./data/oliv_pca.p', 'rb') as f:
            cov, cor, l, v = pickle.load(f)
            return cov, cor, l, v
    else:
        print("data/oliv_pca.p doesn't exist, need to calculate PCA")
        cov, cor, l, v = pca(data.data)
        return cov, cor, l, v


def main():
    olivetti = datasets.fetch_olivetti_faces()
    y = olivetti.target # ktora to osoba (pozniej ktora ma okulary)

    show_some_images(olivetti.data, range(0, 10, 1))

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    x_mean = np.mean(X_train, axis=0)
    print("x_mean:", x_mean)
    print(X_test.shape)
    print(X_train.shape)

    print_image(x_mean.reshape(64, 64))

    L, V = pca(X_train, components=None)
    pickle_all([L, V], 'data/olivetti_pca.pik')
    #L, V = unpickle_all('data/olivetti_pca.pik')

    n_eigen_faces = 16
    some_eigen_faces = V[:, :n_eigen_faces].T
    show_some_images(some_eigen_faces, range(n_eigen_faces))
    #show_some_images()


    # rekonstrukcja obrazow testowych na podstawie twarzy wlasnych
    # funkcja reconstructions(img, b, dims, x_mean=x_mean)

    # img = X_test[0, :].T
    # dims = [0, 1, 10, 20, 30, 50, 100..., 4096]
    # reconstr = reconstructions(img, b, dims, x_mean=x_mean)
    # originals = np.tile(img, (len(doms,), 1))

    # rekonstrukcje iles razy bedzie mnozyc macierz V * pozycje
    # po lewej oryginal po prawej rekonstrukcja
    # 0 skladowych glownych, 10 pierwszych - wyswietlac blad srednia roznice bezwzgledna -
    # orignal - rekonstrukjca
    # robi rekonstrukcje i wizualizuje wraz z pomiarem bledow
    #

if __name__ == '__main__':
    main()
