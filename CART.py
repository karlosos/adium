"""
CART
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
from classifiers import DecisionTree
import numpy as np
np.set_printoptions(threshold=np.inf)

def slice_variance_sum_ratio(L, V, variance_sum_ratio):
    for i in range(len(L)):
        if np.sum(L[:i])/np.sum(L) >= variance_sum_ratio:
            L = L[:i]
            V = V[:, :i]
            return L, V, i

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
        L, V, _ = slice_variance_sum_ratio(L, V, variance_sum_ratio)

    elif components is not None:
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


def load_pca_or_generate(X_train):
    file_exists = os.path.isfile('./data/olivetti_pca.pik')

    if file_exists:
        L, V = unpickle_all('data/olivetti_pca.pik')
        return L, V
    else:
        L, V = pca(X_train, components=None, variance_sum_ratio=0.95)
        pickle_all([L, V], 'data/olivetti_pca.pik')
        return L, V


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

def main():
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    #print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))
    # tutaj do rozpoznawania czy okulary czy ludzieklasyfikujemy
    y = y_glasses
    #y = y.target
    #show_some_images(olivetti.images, glasses, title="Okularnicy")

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    x_mean = np.mean(X_train, axis=0)
    L, V = load_pca_or_generate(X_train)

    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    #print(data_all[0, :5])
    #print(X_test[0, :5])

    dt = DecisionTree(impurity="impurity_entropy")
    dt.fit(X_train_pca, y_train)
    np.set_printoptions(precision=1)
    print(dt.tree_)
    print(dt.tree_.shape)

    #show_some_images(V.T, indexes=[6, 3, 7])
    print(dt.predict(X_test_pca[:10, :]))
    print(dt.score(X_train_pca, y_train))
    print(dt.score(X_test_pca, y_test))

    print(np.sum(y_test == dt.predict(X_test_pca)) / y_test.size)
    show_some_images(X_test[:10, :])

if __name__ == '__main__':
    main()
