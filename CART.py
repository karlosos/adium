"""
CART
"""

from sklearn import datasets
import time
from sklearn.model_selection import train_test_split
from classifiers import DecisionTree
import classifiers
import numpy as np
from misc import *  # funkcje do picklowania
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def pca(data, components=None, variance_sum_ratio=None):
    """
    Perform Principal Components Analysis
    :param components: Zmniejszanie wymiarowości. Liczba pierwszych komponentów jakie
    mają byc brane pod uwagę. Obcina macierz V i wektor L do tej liczby.
    :param variance_sum_ratio: Zmniejszanie wymiarowości. Liczba w zakresie od 0 do 1, która określa
    ile składowych ma być grane pod uwagę. Dobiera taką najmniejszą liczbę początkowych składowych głównych,
    które objaśniają zadaną część całkowitej wariancji.
    """

    cov = np.cov(data, rowvar=False)  # przyklady ulozone wierszami

    # L - wartości wlasne - lambdy - wariancje względem kierunków
    # V - wektory własne - ułożone kolumnami
    start_time = time.time()
    l, v = np.linalg.eig(cov)
    L = np.real(l)
    V = np.real(v)
    ordering = np.argsort(-L)
    L = L[ordering]
    V = V[:, ordering]

    # Redukcja wymiarowości
    if variance_sum_ratio is not None:
        L, V, _ = slice_variance_sum_ratio(L, V, variance_sum_ratio)
    elif components is not None:
        L = L[:components]
        V = V[:, :components]

    elapsed_time = time.time() - start_time
    print("time eig:", elapsed_time)
    return L, V


def slice_variance_sum_ratio(L, V, variance_sum_ratio):
    """
    Zmniejszanie wymiarowości. Dobiera taką najmniejszą liczbę początkowych składowych głównych,
    które objaśniają zadaną część całkowitej wariancji. Zwraca zredukowane L, V oraz indeks i do którego
    zostało wykonane przycięcie.

    :return: zredukowane L, V oraz indeks do którego zostało wykonane przycięcie
    """
    for i in range(len(L)):
        if np.sum(L[:i])/np.sum(L) >= variance_sum_ratio:
            L = L[:i]
            V = V[:, :i]
            return L, V, i


def load_pca_or_generate(X_train):
    """
    Jeżeli istnieją zapiklowane dane PCA to je wczytaj. W innym przypadku wykonaj PCA na podanych danych.
    :return: wartości własne i macierze własne
    """
    file_exists = os.path.isfile('./data/olivetti_pca.pik')

    if file_exists:
        L, V = unpickle_all('data/olivetti_pca.pik')
        return L, V
    else:
        L, V = pca(X_train, components=None, variance_sum_ratio=0.95)
        pickle_all([L, V], 'data/olivetti_pca.pik')
        return L, V


def show_some_images(images, indexes=None, as_grid=True, title=None, subtitles=None):
    """
    Wyświetla wiele obrazków jednocześnie.

    :param images: dane z obrazkami. Może to być macierz o trzech wymiarach. images[i, j, k], gdzie i to numer zdjęcia,
    j, k - współrzędne pikselu, lub o dwóch wymiarach images[i, j], gdzie i to numer zdjęcia a j
    :param indexes: indeksy które chcemy wyświetlać
    :param as_grid: czy chcemy wyświetlać w siatce - True, lub w linii poziomej.
    """
    if indexes is None:
        indexes = range(len(images))
    shape = images[0].shape
    if len(shape) == 1:  # obrazki spłaszczone - reprezentowane w postaci wektora, przyjmujemy że obrazki są kwadratowe
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
        if subtitles is not None:
            plt.title(subtitles[index])

    plt.show()


def dimension_comprasion():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)

    ##
    # Classificatione experiments
    ##
    dimensions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    errors_train = []
    errors_test = []
    #n = 50
    for n in dimensions:
        X_train_pca = X_train.dot(V[:, :n])
        X_test_pca = X_test.dot(V[:, :n])
        data_all = olivetti.data.dot(V[:, :n])

        dt = DecisionTree(impurity="impurity_entropy")
        t1 = time.time()
        dt.fit(X_train_pca, y_train)
        t2 = time.time()
        print("Time: ", t2 - t1)

        print(dt.tree_)
        print(dt.tree_.shape)
        print(np.sum(dt.tree_[:, DecisionTree.COL_CHILD_LEFT] == 0.0))

        predictions = dt.predict(X_test_pca[:10, :])

        print(predictions)
        print("Dimension:", n)
        print("Wynik klasyfikacji dla zbioru uczącego:", dt.score(X_train_pca, y_train))
        errors_test.append(dt.score(X_test_pca, y_test))
        print("Wynik klasyfikacji dla zbioru testowego:", dt.score(X_test_pca, y_test))
        print("Wynik klasyfikacji dla zbioru testowego (custom):", np.sum(y_test == dt.predict(X_test_pca)) / y_test.size)

    plt.figure()
    plt.plot(dimensions, errors_test)
    plt.title("Dokładność testowa dla liczby użytych cech")
    plt.xlabel("Lizba użytych cech")
    plt.ylabel("Dokładność testowy")
    plt.savefig("docs/dimensions_test.eps")

def dimension_impurity_comprasion():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)

    ##
    # Classificatione experiments
    ##
    dimensions = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    impurities = ['impurity_error', 'impurity_entropy', 'impurity_gini']
    errors_test = np.zeros((len(impurities), len(dimensions)))
    errors_times = np.zeros((len(impurities), len(dimensions)))
    for dimension_index, n in enumerate(dimensions):
        X_train_pca = X_train.dot(V[:, :n])
        X_test_pca = X_test.dot(V[:, :n])

        for impurity_index, impurity in enumerate(impurities):
            dt = DecisionTree(impurity=impurity)
            t1 = time.time()
            dt.fit(X_train_pca, y_train)
            t2 = time.time()
            errors_times[impurity_index, dimension_index] = t2-t1
            errors_test[impurity_index, dimension_index] = 1-dt.score(X_test_pca, y_test)

    plt.figure()
    for i in range(len(impurities)):
        plt.plot(dimensions, errors_test[i], label=impurities[i])


    plt.title("Błąd testowy dla liczby użytych cech")
    plt.xlabel("Lizba użytych cech")
    plt.ylabel("Błąd testowy")
    plt.legend()
    plt.savefig("docs/dimensions_entropy_test.eps")


def depth_comparison():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)
    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    dt = DecisionTree(impurity="impurity_entropy")
    dt.fit(X_train_pca, y_train)
    predictions = dt.predict(X_test_pca[:10, :])

    max_depth = int(np.max(dt.tree_[:, DecisionTree.COL_DEPTH]))
    errors_train = np.zeros(max_depth + 1)
    errors_test = np.zeros(max_depth + 1)
    for d in range(max_depth + 1):
        dt = DecisionTree(impurity="impurity_entropy", max_depth=d)
        dt.fit(X_train_pca, y_train)
        print('depth: ', d, 'shape:', dt.tree_.shape)
        errors_train[d] = 1 - dt.score(X_train_pca, y_train)
        errors_test[d] = 1 - dt.score(X_test_pca, y_test)

    np.set_printoptions(threshold=np.inf, precision=5)
    best_depth = np.argmin(errors_test)

    plt.figure()
    plt.plot(errors_train, marker='o', label="train errors")
    plt.plot(errors_test, marker='o', label="test errors")
    plt.legend()
    plt.title("Głębokość drzewa")
    plt.savefig("docs/depth_test.eps")

def sample_size_comparison():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)
    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    dt = DecisionTree(impurity="impurity_entropy")
    dt.fit(X_train_pca, y_train)
    predictions = dt.predict(X_test_pca[:10, :])

    min_node_vals = np.arange(0.10, 0, -0.01)
    errors_train = np.zeros(min_node_vals.size)
    errors_test = np.zeros(min_node_vals.size)
    for i, min_node_examples in enumerate(min_node_vals):
        dt = DecisionTree(impurity="impurity_entropy", min_node_examples=min_node_examples)
        t1 = time.time()
        dt.fit(X_train_pca, y_train)
        t2 = time.time()
        print("time:", t2-t1)
        print('min node examples: ', min_node_examples)
        errors_train[i] = 1 - dt.score(X_train_pca, y_train)
        errors_test[i] = 1 - dt.score(X_test_pca, y_test)

    np.set_printoptions(threshold=np.inf, precision=5)
    best_depth = np.argmin(errors_test)
    print('BEST DEPTH:', str(best_depth), " WITH TEST ACCURACY:", 1 - errors_test[best_depth])
    print('ERRORS TEST: ', errors_test)
    print('ERRORS TRAIN: ', errors_train)

    plt.figure()
    plt.plot(min_node_vals, errors_train, marker='o', label="train errors")
    plt.plot(min_node_vals, errors_test, marker='o', label="test errors")
    plt.xlim(np.max(min_node_vals), np.min(min_node_vals))
    plt.legend()
    plt.title("Procentowa zawartość przykładów w węźle")
    plt.savefig("docs/min_node_vals_test.eps")


def pruning_comparison_greedy():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)
    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    dt = DecisionTree(impurity="impurity_entropy")
    dt.fit(X_train_pca, y_train)

    pentalties = np.arange(0.015, 0.0, -0.0025)
    errors_train = np.zeros(pentalties.size)
    errors_test = np.zeros(pentalties.size)
    for i, penalty in enumerate(pentalties):
        print('penalty', penalty)
        dt = DecisionTree(impurity="impurity_entropy", pruning='greedy_subtrees', penalty=penalty)
        t1 = time.time()
        dt.fit(X_train_pca, y_train)
        t2 = time.time()
        print('time:', t2-t1)
        errors_train[i] = 1 - dt.score(X_train_pca, y_train)
        errors_test[i] = 1 - dt.score(X_test_pca, y_test)

    np.set_printoptions(threshold=np.inf, precision=5)
    best_penalty_index = np.argmin(errors_test)
    print('BEST PENALTY:', str(pentalties[best_penalty_index]), " WITH TEST ACCURACY:", 1 -
          errors_test[best_penalty_index])
    print('ERRORS TEST: ', errors_test)
    print('ERRORS TRAIN: ', errors_train)

    plt.figure()
    plt.plot(pentalties, errors_train, color='black', marker='o', label="train")
    plt.plot(pentalties, errors_test, color='red', marker='o', label="test")
    plt.legend()
    plt.xlabel("penalty")
    plt.xlim(np.max(pentalties), np.min(pentalties))
    plt.title("Pruning - greedy subtrees")
    plt.savefig("docs/pruning_greedy.eps")


def pruning_comparison_exhaustive():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)
    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    dt = DecisionTree(impurity="impurity_entropy")
    dt.fit(X_train_pca, y_train)

    pentalties = np.arange(0.015, 0.0, -0.0025)
    errors_train = np.zeros(pentalties.size)
    errors_test = np.zeros(pentalties.size)
    for i, penalty in enumerate(pentalties):
        print('penalty', penalty)
        dt = DecisionTree(impurity="impurity_entropy", pruning='exhaustive_subtrees', penalty=penalty)
        t1 = time.time()
        dt.fit(X_train_pca, y_train)
        t2 = time.time()
        print('time:', t2-t1)
        errors_train[i] = 1 - dt.score(X_train_pca, y_train)
        errors_test[i] = 1 - dt.score(X_test_pca, y_test)

    np.set_printoptions(threshold=np.inf, precision=5)
    best_penalty_index = np.argmin(errors_test)
    print('BEST PENALTY:', str(pentalties[best_penalty_index]), " WITH TEST ACCURACY:", 1 -
          errors_test[best_penalty_index])
    print('ERRORS TEST: ', errors_test)
    print('ERRORS TRAIN: ', errors_train)

    plt.figure()
    plt.plot(pentalties, errors_train, color='black', marker='o', label="train")
    plt.plot(pentalties, errors_test, color='red', marker='o', label="test")
    plt.legend()
    plt.xlabel("penalty")
    plt.xlim(np.max(pentalties), np.min(pentalties))
    plt.title("Pruning - exhaustive subtrees")
    plt.savefig("docs/pruning_exhaustive.eps")

def main():
    np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses
    # y = y.target

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)

    ##
    # Classificatione experiments
    ##
    n = 50
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    dt = DecisionTree(impurity="impurity_entropy")
    t1 = time.time()
    dt.fit(X_train_pca, y_train)
    t2 = time.time()
    print("Time: ", t2-t1)

    print(dt.tree_)
    print(dt.tree_.shape)
    print(np.sum(dt.tree_[:, DecisionTree.COL_CHILD_LEFT] == 0.0))

    predictions = dt.predict(X_test_pca[:10, :])

    print(predictions)
    print("Wynik klasyfikacji dla zbioru uczącego:", dt.score(X_train_pca, y_train))
    print("Wynik klasyfikacji dla zbioru testowego:", dt.score(X_test_pca, y_test))
    print("Wynik klasyfikacji dla zbioru testowego (custom):", np.sum(y_test == dt.predict(X_test_pca)) / y_test.size)
    #
    # # show_some_images(V.T, indexes=[6, 3, 7])
    # show_some_images(X_test[:10, :], subtitles=predictions)
    #
    # ##
    # # Testy dla głębokości
    # ##
    #
    # max_depth = int(np.max(dt.tree_[:, DecisionTree.COL_DEPTH]))
    # errors_train = np.zeros(max_depth + 1)
    # errors_test = np.zeros(max_depth + 1)
    # for d in range(max_depth + 1):
    #     dt = DecisionTree(impurity="impurity_entropy", max_depth=d)
    #     dt.fit(X_train_pca, y_train)
    #     print('depth: ', d, 'shape:', dt.tree_.shape)
    #     errors_train[d] = 1 - dt.score(X_train_pca, y_train)
    #     errors_test[d] = 1 - dt.score(X_test_pca, y_test)
    #
    # np.set_printoptions(threshold=np.inf, precision=5)
    # best_depth = np.argmin(errors_test)
    # print('BEST DEPTH:', str(best_depth), " WITH TEST ACCURACY:", 1 - errors_test[best_depth])
    # print('ERRORS TEST: ', errors_test)
    # print('ERRORS TRAIN: ', errors_train)
    #
    # plt.figure()
    # plt.plot(errors_train, color='black', marker='o')
    # plt.plot(errors_test, color='red', marker='o')
    # plt.show()
    #
    # ##
    # # Testy dla sample
    # ##
    #
    # min_node_vals = np.arange(0.10, 0, -0.01)
    # errors_train = np.zeros(min_node_vals.size)
    # errors_test = np.zeros(min_node_vals.size)
    # for i, min_node_examples in enumerate(min_node_vals):
    #     dt = DecisionTree(impurity="impurity_entropy", min_node_examples=min_node_examples)
    #     dt.fit(X_train_pca, y_train)
    #     print('min node examples: ', min_node_examples)
    #     errors_train[i] = 1 - dt.score(X_train_pca, y_train)
    #     errors_test[i] = 1 - dt.score(X_test_pca, y_test)
    #
    # np.set_printoptions(threshold=np.inf, precision=5)
    # best_depth = np.argmin(errors_test)
    # print('BEST DEPTH:', str(best_depth), " WITH TEST ACCURACY:", 1 - errors_test[best_depth])
    # print('ERRORS TEST: ', errors_test)
    # print('ERRORS TRAIN: ', errors_train)
    #
    # plt.figure()
    # plt.plot(errors_train, color='black', marker='o')
    # plt.plot(errors_test, color='red', marker='o')
    # plt.show()
    #
    # ##
    # # Jak kara lambda wpływa
    # ##
    # dt = DecisionTree(impurity="impurity_entropy")
    # dt.fit(X_train_pca, y_train)
    #
    # pentalties = np.arange(0.015, 0.0, -0.0025)
    # errors_train = np.zeros(pentalties.size)
    # errors_test = np.zeros(pentalties.size)
    # for i, penalty in enumerate(pentalties):
    #     print('penalty', penalty)
    #     dt = DecisionTree(impurity="impurity_entropy", pruning='greedy_subtrees', penalty=penalty)
    #     t1 = time.time()
    #     dt.fit(X_train_pca, y_train)
    #     t2 = time.time()
    #     print('time:', t2-t1)
    #     errors_train[i] = 1 - dt.score(X_train_pca, y_train)
    #     errors_test[i] = 1 - dt.score(X_test_pca, y_test)
    #
    # np.set_printoptions(threshold=np.inf, precision=5)
    # best_penalty_index = np.argmin(errors_test)
    # print('BEST PENALTY:', str(pentalties[best_penalty_index]), " WITH TEST ACCURACY:", 1 -
    #       errors_test[best_penalty_index])
    # print('ERRORS TEST: ', errors_test)
    # print('ERRORS TRAIN: ', errors_train)
    #
    # plt.figure()
    # plt.plot(errors_train, color='black', marker='o')
    # plt.plot(errors_test, color='red', marker='o')
    # plt.title("greedy")
    # plt.show()
    #
    # #
    # # Exhaustive
    # #
    # dt = DecisionTree(impurity="impurity_entropy")
    # dt.fit(X_train_pca, y_train)
    #
    # pentalties = np.arange(0.015, 0.0, -0.0025)
    # errors_train = np.zeros(pentalties.size)
    # errors_test = np.zeros(pentalties.size)
    # for i, penalty in enumerate(pentalties):
    #     print('penalty', penalty)
    #     dt = DecisionTree(impurity="impurity_entropy", pruning='exhaustive_subtrees', penalty=penalty)
    #     t1 = time.time()
    #     dt.fit(X_train_pca, y_train)
    #     t2 = time.time()
    #     print('time:', t2-t1)
    #     errors_train[i] = 1 - dt.score(X_train_pca, y_train)
    #     errors_test[i] = 1 - dt.score(X_test_pca, y_test)
    #
    # np.set_printoptions(threshold=np.inf, precision=5)
    # best_penalty_index = np.argmin(errors_test)
    # print('BEST PENALTY:', str(pentalties[best_penalty_index]), " WITH TEST ACCURACY:", 1 -
    #       errors_test[best_penalty_index])
    # print('ERRORS TEST: ', errors_test)
    # print('ERRORS TRAIN: ', errors_train)

    # svc = SVC()
    # svc.fit(X_train, y_train)
    # print("SVC Default scores [train, test]:" + str(svc.score(X_train, y_train)) + ', ' + str(svc.score(X_test, y_test)))
    #
    # svc = SVC(C=10.0**1, kernel='rbf')
    # svc.fit(X_train, y_train)
    # print("SVC Default scores [train, test]:" + str(svc.score(X_train, y_train)) + ', ' + str(svc.score(X_test, y_test)))

    # Cs = 2.0**np.arange(-8, 2)
    # svm_errs_train = np.zeros(Cs.size)
    # svm_errs_test = np.zeros(Cs.size)
    #
    # for i, C in enumerate(Cs):
    #     svc = SVC(C=C, kernel='linear')
    #     svc.fit(X_train_pca, y_train)
    #     print(
    #         "SVC Default scores [train, test]:" + str(svc.score(X_train_pca, y_train)) + ', ' + str(svc.score(X_test_pca, y_test)))
    #     svm_errs_test[i] = svc.score(X_test_pca, y_test)
    #     svm_errs_train[i] = svc.score(X_train_pca, y_train)
    #
    # plt.figure()
    # plt.plot(np.log(Cs), svm_errs_test, color='black', marker='o')
    # plt.plot(np.log(Cs), svm_errs_train, color='red', marker='o')
    # plt.title("")
    # plt.grid(True)
    # plt.show()

    #
    # Drzewko z sklearn
    #
    sklearn_tree = DecisionTreeClassifier(min_samples_split=2, criterion='entropy', random_state=0)
    t1 = time.time()
    sklearn_tree.fit(X_train_pca, y_train)
    t2 = time.time()
    print("Sklearn time", t2-t1)
    print("Node count", sklearn_tree.tree_.node_count)
    print("Train, test", sklearn_tree.score(X_train_pca, y_train), sklearn_tree.score(X_test_pca, y_test))

def tune_penalty():
    #np.set_printoptions(threshold=np.inf, precision=1)
    olivetti = datasets.fetch_olivetti_faces()

    glasses = np.genfromtxt('olivetti_glasses.txt', delimiter=',').astype(int)

    # tworzymy wektor z labelkami, czy dane zdjęcie przedstawia okularnika
    y_glasses = np.zeros(olivetti.data.shape[0])
    y_glasses = y_glasses.astype(int)
    y_glasses[glasses] = 1

    # ile osób ma okulary w zbiorze danych
    # print(np.where(y_glasses == 1)[0].size / float(olivetti.data.shape[0]))

    # Wybraliśmy, że będziemy uczyć klasyfikator po okularach.
    y = y_glasses

    # show_some_images(olivetti.images, glasses, title="Okularnicy")

    X_train, X_test, y_train, y_test = train_test_split(olivetti.data, y, test_size=0.2,
                                                        stratify=y, random_state=0)
    L, V = load_pca_or_generate(X_train)

    ##
    # Classificatione experiments
    ##
    n = 10
    X_train_pca = X_train.dot(V[:, :n])
    X_test_pca = X_test.dot(V[:, :n])
    data_all = olivetti.data.dot(V[:, :n])

    #penalties = np.arange(0.015, 0.0, -0.0025)
    penalties = np.arange(0.02, 0.0, -0.005)
    dt = classifiers.choose_best_penalty(X_train_pca, y_train, k=4, penalties=penalties)

    print("Wynik klasyfikacji dla zbioru uczącego:", dt.score(X_train_pca, y_train))
    print("Wynik klasyfikacji dla zbioru testowego:", dt.score(X_test_pca, y_test))


if __name__ == '__main__':
    plt.style.use('default')
    #main()
    tune_penalty()
    #dimension_comprasion()
    #dimension_impurity_comprasion()
    #depth_comparison()
    #sample_size_comparison()
    #pruning_comparison_exhaustive()
    #pruning_comparison_greedy()
