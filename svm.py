"""
TODO:
- [ ] rysowanie marginesu
- [ ] rysowanie płaszczyzn w 3d
- [ ] oblicze
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MyClassifier():
    def __init__(self):
        self.coef_ = []
        self.intercept_ = []
        self.support_vectors_ = None


def svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=0)

    clf = SVC(C=1000, kernel='linear')
    clf.fit(X_train, y_train)
    print(
        "SVC Default scores [train, test]:" + str(clf.score(X_train, y_train)) + ', ' + str(clf.score(X_test, y_test)))
    return clf


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

    # Setting solver parameters (change default to decrease tolerance)
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    # Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # w parameter in vectorized form
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)

    # # Display results
    # print('Alphas = ', alphas[alphas > 1e-4])
    # print('w = ', w.flatten())
    # print('b = ', b[0])

    clf = MyClassifier()
    clf.coef_.append(w.flatten())
    clf.coef_ = np.asarray(clf.coef_)
    clf.intercept_ = b[0].tolist()
    clf.support_vectors_ = X[S]
    return clf


def x1_visualisation(X, fn):
    y = X[:, 2]
    X = X[:, :2]
    clf = fn(X, y)
    print('w = ', clf.coef_)
    print('b = ', clf.intercept_)
    print('Support vectors = ', clf.support_vectors_)
    visualisation_2d(clf, X, y)
    w = clf.coef_[0]
    print("SVC margines separacji tao =", (1 / np.linalg.norm(w)))
    print("SVC margines separacji tao =", 1 / np.sqrt(np.sum(clf.coef_ ** 2)))


def x2_visualisation_svc(X):
    y = X[:, 3]
    X = X[:, :3]
    clf = svm(X, y)
    visualisation_3d(clf, X, y)


def visualisation_2d(clf, X, y):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx_min = np.min(X[:, 0])
    xx_max = np.max(X[:, 0])
    xx = np.linspace(xx_min, xx_max)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    plt.figure()
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    # TODO zaznaczyć margines (można go poprowadzić np. od prostej separacji do punktów podparcia)
    plt.plot([clf.support_vectors_[0, 0], 1], [clf.support_vectors_[0, 1], 1], 'ro-')
    plt.plot()

    plt.show()


def visualisation_3d(clf, X, Y):
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    tmp = np.linspace(-6, 6, 51)
    x, y = np.meshgrid(tmp, tmp)

    # Plot stuff.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z(x, y))

    # TODO fix surface
    ax.plot_surface(x, y, z(x, y) + np.sqrt(1 + a ** 2) * margin)
    ax.plot_surface(x, y, z(x, y) - np.sqrt(1 + a ** 2) * margin)

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], clf.support_vectors_[:, 2], s=80,
                facecolors='green', zorder=10, edgecolors='k')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
    # TODO zaznaczyć margines (można go poprowadzić np. od prostej separacji do punktów podparcia)
    plt.show()


def main():
    D = loadmat("data/data_for_svm.mat")
    X1 = D['X1']
    X2 = D['X2']
    X3 = D['X3']

    x1_visualisation(X1, svm)
    x1_visualisation(X1, cvxopt)
    #x2_visualisation_svc(X2)


if __name__ == '__main__':
    main()