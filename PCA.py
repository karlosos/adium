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


def pca(data, components=None, variance_sum_ratio=None):
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

    if variance_sum_ratio is not None:
        for i in range(len(L)):
            if L[:i]/L >= variance_sum_ratio:
                L, V, _ = variance_sum_ratio(L, V, variance_sum_ratio)

    elif components is not None:
        L = L[:components]
        V = V[:, :components]

    elapsed_time = time.time() - start_time
    print("time eig:", elapsed_time)
    return L, V


def slice_variance_sum_ratio(L, V, variance_sum_ratio):
    for i in range(len(L)):
        if np.sum(L[:i])/np.sum(L) >= variance_sum_ratio:
            L = L[:i]
            V = V[:, :i]
            return L, V, i


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


def reconstructions_components(x, V, dims, x_mean=None):
    # size of data (size of single image)
    l = V.shape[0]
    if x_mean is not None:
        x = x - x_mean

    # l_i = V.T \cdot xi
    # otrzymujemy wagi
    x = np.dot(V.T, x)

    # rekonstrukcja
    # tworzymy listi obrazow rekonstrukcji
    reconstructions = np.zeros((len(dims), l))
    for i, dim in enumerate(dims):
        reconstructions[i] = V[:, :dim].dot(x[:dim])
        if x_mean is not None:
            reconstructions[i] += x_mean

    return reconstructions


def reconstructions_variance_sum_ratios(x, L, V, variance_sum_ratios, x_mean=None):
    # size of data (size of single image)
    l = V.shape[0]
    if x_mean is not None:
        x = x - x_mean
    x = np.dot(V.T, x)
    reconstructions = np.zeros((len(variance_sum_ratios), l))
    for i, variance_sum_ratio in enumerate(variance_sum_ratios):
        _, _, slice_index = slice_variance_sum_ratio(L, V, variance_sum_ratio)
        reconstructions[i] = V[:, :slice_index].dot(x[:slice_index])
        if x_mean is not None:
            reconstructions[i] += x_mean

    return reconstructions


def show_image_pairs(originals, reconstructions, main_title, subtitles, maes):
    number_of_pairs = originals.shape[0]

    shape = originals[0].shape
    if len(shape) == 1:  # flattened images
        img_side = int(np.sqrt(shape))
        originals = originals.reshape(originals.shape[0], img_side, img_side)

    shape = reconstructions[0].shape
    if len(shape) == 1:  # flattened images
        img_side = int(np.sqrt(shape))
        reconstructions = reconstructions.reshape(reconstructions.shape[0], img_side, img_side)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(main_title)

    grid_size = int(np.ceil(np.sqrt(number_of_pairs)))
    outer = gridspec.GridSpec(grid_size, grid_size, wspace=0.2, hspace=0.2)

    for i in range(number_of_pairs):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, inner[0])
        ax.set_title(subtitles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(originals[i])
        fig.add_subplot(ax)

        bx = plt.Subplot(fig, inner[1])
        bx.set_title(np.around(maes[i], 5))
        bx.set_xticks([])
        bx.set_yticks([])
        bx.imshow(reconstructions[i])
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
    plt.imshow(img.reshape(64, 64))
    plt.show()

    components = [0, 1, 100, 200, 300, 400, 450, 500, 750, 1000, 2000, 2500, 3000, 3500, 4096]
    reconstructions = reconstructions_components(img, V, components, x_mean=x_mean)
    originals = np.tile(img, (len(components), 1))
    maes = [np.sum(np.abs(originals[i]-reconstructions[i])) / reconstructions[i].size for i in range(len(components))]
    plt.plot(components, maes)
    plt.show()
    show_image_pairs(originals, reconstructions, 'Rekonstrukcja PCA', subtitles=['dim of V:' + str(dim) for dim in components], maes=maes)


    img = X_test[5, :].T
    variance_sum_ratios = np.asarray([0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.95, 0.98, 0.99, 0.995, 0.999])
    reconstructions = reconstructions_variance_sum_ratios(img, L, V, variance_sum_ratios, x_mean=x_mean)
    originals = np.tile(img, (len(variance_sum_ratios), 1))
    maes = [np.sum(np.abs(originals[i]-reconstructions[i])) / reconstructions[i].size for i in range(len(variance_sum_ratios))]
    plt.plot(variance_sum_ratios, maes)
    plt.show()
    show_image_pairs(originals, reconstructions, 'Rekonstrukcja PCA', subtitles=['vsr:' + str(vsr) for vsr in variance_sum_ratios], maes=maes)


if __name__ == '__main__':
    main()
