"""
TODO:
- [x] rysowanie marginesu
- [x] rysowanie płaszczyzn w 3d
- [x] cvxopt dla 3d
- [x] svm soft margin (wartości C=0.1, 1.0, 10
- [x] svm kernel='rbf' - wizualizacja wykres warstwicowy
- [ ] svm kernel='rvf' obliczenie granicy
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MyClassifier():
    def __init__(self):
        self.coef_ = []
        self.intercept_ = []
        self.support_vectors_ = None


def svm(X, y, C=1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=0)

    clf = SVC(C=C, kernel='linear')
    clf.fit(X_train, y_train)
    print(
        "SVC Default scores [train, test]:" + str(clf.score(X_train, y_train)) + ', ' + str(clf.score(X_test, y_test)))
    return clf, [X_train, X_test, y_train, y_test]


def cvxopt(X, y):
    m, n = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # w parameter in vectorized form
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)

    clf = MyClassifier()
    clf.coef_.append(w.flatten())
    clf.coef_ = np.asarray(clf.coef_)
    clf.intercept_ = b[0].tolist()
    clf.support_vectors_ = X[S]
    return clf, []


def x1_visualisation(X, fn):
    y = X[:, 2]
    X = X[:, :2]
    t1 = time.time()
    clf, _ = fn(X, y)
    t2 = time.time()
    print(fn.__name__, " x1 - time:", t2-t1)
    print('w = ', clf.coef_)
    print('b = ', clf.intercept_)
    print('Support vectors = ', clf.support_vectors_)
    w = clf.coef_[0]
    print("SVC margines separacji tao =", (1 / np.linalg.norm(w)))
    print("SVC margines separacji tao =", 1 / np.sqrt(np.sum(clf.coef_ ** 2)))
    print("========================")
    visualisation_2d(clf, X, y)


def x2_visualisation(X, fn):
    y = X[:, 3]
    X = X[:, :3]
    t1 = time.time()
    clf, _ = fn(X, y)
    t2 = time.time()
    print(fn.__name__, " x1 - time:", t2-t1)
    print('w = ', clf.coef_)
    print('b = ', clf.intercept_)
    print('Support vectors = ', clf.support_vectors_)
    print("SVC margines separacji tao =", 1 / np.sqrt(np.sum(clf.coef_ ** 2)))
    print("========================")
    visualisation_3d(clf, X, y)


def x3_experiment(X):
    y = X[:, 2]
    X = X[:, :2]

    Cs = [0.1, 1.0, 10.0]
    svm_errs_train = np.zeros(len(Cs))
    svm_errs_test = np.zeros(len(Cs))

    fig = plt.figure()
    number_of_subplots = len(Cs)

    svm_errs_train = np.zeros(len(Cs))
    svm_errs_test = np.zeros(len(Cs))

    for i, C in enumerate(Cs):
        clf, [X_train, X_test, y_train, y_test] = svm(X, y, C)
        print("SVM C =", C)
        print('w = ', clf.coef_)
        print('b = ', clf.intercept_)
        print('Support vectors = ', clf.support_vectors_)
        w = clf.coef_[0]
        print("SVC margines separacji tao =", (1 / np.linalg.norm(w)))
        print("SVC margines separacji tao =", 1 / np.sqrt(np.sum(clf.coef_ ** 2)))
        print("========================")

        svm_errs_test[i] = clf.score(X_test, y_test)
        svm_errs_train[i] = clf.score(X_train, y_train)

        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx_min = np.min(X[:, 0])
        xx_max = np.max(X[:, 0])
        xx = np.linspace(xx_min, xx_max)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        ax = plt.subplot(number_of_subplots, 1, i+1)
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        # TODO zaznaczyć margines (można go poprowadzić np. od prostej separacji do punktów podparcia) - czerwona linia
        # plt.plot([clf.support_vectors_[0, 0], 1], [clf.support_vectors_[0, 1], 1], 'ro-')

        ax.title.set_text("C="+str(C) + " margines=" + str((1 / np.linalg.norm(w))))

    plt.show()

    plt.figure()
    plt.plot(Cs, svm_errs_test, color='black', marker='o')
    plt.plot(Cs, svm_errs_train, color='red', marker='o')
    plt.title("SVM Soft Margin")
    plt.grid(True)
    plt.show()


def x3_rbf(X):
    y = X[:, 2]
    X = X[:, :2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=0)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    print(
        "SVC Default scores [train, test]:" + str(clf.score(X_train, y_train)) + ', ' + str(clf.score(X_test, y_test)))

    plot_decision_regions(X, y, classifier=clf)
    plt.tight_layout()
    plt.show()


def visualisation_2d(clf, X, y):
    print(clf.coef_)
    print(clf.intercept_)
    w = clf.coef_[0]
    intercept = clf.intercept_
    a = -w[0] / w[1]
    b = -intercept[0] / w[1]
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    xx = np.linspace(x_min, x_max)
    separation = a * xx + b

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    b_down = b - margin
    b_up = b + margin
    margin_down = a * xx + b_down
    margin_up = a * xx + b_up

    if clf.__class__ == MyClassifier:
        x1 = clf.support_vectors_[-1][0]
        y1 = clf.support_vectors_[-1][1]
    else:
        x1 = clf.support_vectors_[0][0]
        y1 = clf.support_vectors_[0][1]

    # prostopadla
    a_perp = -1 / a
    b_perp = x1 * (1 / a + a) + b_down

    # przeciecie prostopadlej i srodkowej
    x_mid = (b - b_perp) / (a_perp - a)
    y_mid = a_perp * x_mid + b_perp

    plt.figure()
    plt.clf()
    plt.plot(xx, separation)
    plt.plot(xx, margin_down, 'k--')
    plt.plot(xx, margin_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.plot([x1, x_mid], [y1, y_mid], 'o-')
    plt.axes().set_aspect('equal')
    plt.show()


def visualisation_3d(clf, X, Y):
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    tmp = np.linspace(-6, 6, 51)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z(x, y))

    ax.plot_surface(x, y, z(x, y) + np.sqrt(1 + a ** 2) * margin/2, color='gray')
    ax.plot_surface(x, y, z(x, y) - np.sqrt(1 + a ** 2) * margin/2, color='gray')

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], clf.support_vectors_[:, 2], s=80,
                facecolors='green', zorder=10, edgecolors='k')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.show()


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, resolution=0.02):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.decision_function(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx2.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=plt.cm.Paired, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')


def main():
    D = loadmat("data/data_for_svm.mat")
    X1 = D['X1']
    X2 = D['X2']
    X3 = D['X3']

    x1_visualisation(X1, svm)
    x1_visualisation(X1, cvxopt)
    x2_visualisation(X2, svm)
    x2_visualisation(X2, cvxopt)
    x3_experiment(X3)
    x3_rbf(X3)


if __name__ == '__main__':
    main()