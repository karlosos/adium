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
import matplotlib.gridspec as gridspec
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
    plt.show()


def print_image(img):
    plt.gray()
    plt.grid(False)
    plt.imshow(img)
    plt.show()


def load_pca_or_generate(X_train):
    file_exists = os.path.isfile('./data/olivetti_pca.pik')

    if file_exists:
        L, V = unpickle_all('data/olivetti_pca.pik')
        return L, V
    else:
        L, V = pca(X_train, components=None)
        pickle_all([L, V], 'data/olivetti_pca.pik')
        return L, V


def reconstructions(x, V, dims=[10, 20, 30], x_mean=None):
    n = V.shape[0]
    if x_mean is not None:
        x = x - x_mean
    x_new = np.dot(V[:, 0:np.max(dims)].T, x)
    reconstr = np.zeros((len(dims), n))
    for i, dim in enumerate(dims):
        reconstr[i] = V[:, :dim].dot(x_new[:dim])
        if x_mean is not None:
            reconstr[i] += x_mean

    return reconstr


def show_image_pairs(originals, reconstrs, title, titles_2, maes):
    number_of_pairs = originals.shape[0]

    shape = originals[0].shape
    if len(shape) == 1:  # flattened images
        img_side = int(np.sqrt(shape))
        originals = originals.reshape(originals.shape[0], img_side, img_side)

    shape = reconstrs[0].shape
    if len(shape) == 1:  # flattened images
        img_side = int(np.sqrt(shape))
        reconstrs = reconstrs.reshape(reconstrs.shape[0], img_side, img_side)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)

    grid_size = int(np.ceil(np.sqrt(number_of_pairs)))
    outer = gridspec.GridSpec(grid_size, grid_size, wspace=0.2, hspace=0.2)

    for i in range(number_of_pairs):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, inner[0])
        ax.set_title(titles_2[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(originals[i])
        fig.add_subplot(ax)

        bx = plt.Subplot(fig, inner[1])
        bx.set_title(np.around(maes[i], 5))
        bx.set_xticks([])
        bx.set_yticks([])
        bx.imshow(reconstrs[i])
        fig.add_subplot(bx)

    fig.show()
    plt.show()


def main():
    olivetti = datasets.fetch_olivetti_faces()
    y = olivetti.target # ktora to osoba (pozniej ktora ma okulary)

    #show_some_images(olivetti.data, range(0, 10, 1))

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    x_mean = np.mean(X_train, axis=0)
    print("x_mean:", x_mean)
    print(X_test.shape)
    print(X_train.shape)

    #print_image(x_mean.reshape(64, 64))

    L, V = load_pca_or_generate(X_train)

    n_eigen_faces = 16
    some_eigen_faces = V[:, :n_eigen_faces].T
    #show_some_images(some_eigen_faces, range(n_eigen_faces), title='First eigen faces')

    img = X_test[5, :].T
    dims = [0, 1, 100, 200, 300, 400, 450, 500, 750, 1000, 2000, 2500, 3000, 3500, 4096]
    reconstrs = reconstructions(img, V, dims, x_mean=x_mean)
    originals = np.tile(img, (len(dims), 1))
    maes = [np.sum(np.abs(originals[i]-reconstrs[i])) / reconstrs[i].size for i in range(len(dims))]
    plt.plot(dims, maes)
    plt.show()
    show_image_pairs(originals, reconstrs, 'Rekonstrukcja PCA', titles_2=['dim:' + str(dim) for dim in dims], maes=maes)

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
