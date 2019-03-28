from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class DecisionTree(BaseEstimator, ClassifierMixin):
    """
    TODO
    """
    COL_PARENT = 0
    COL_SPLIT_FEATURE = 1
    COL_SPLIT_VALUE = 2
    COL_CHILD_LEFT = 3
    COL_CHILD_RIGHT = 4
    COL_Y = 5
    COL_DEPTH = 6

    def __init__(self, impurity='impurity_entropy'):
        self.tree_ = None
        self.class_labels_ = None
        self.impurity_ = getattr(self, impurity)

    def best_split(self, X, y, indexes):
        n = X.shape[1]
        best_k = None
        best_v = None
        best_expect = np.inf
        for k in range(n):
            X_indexes_k = X[indexes, k]
            u = np.unique(X_indexes_k)
            vals = 0.5 * (u[:-1] + u[1:])
            for v in vals:
                indexes_left = indexes[np.where(X_indexes_k < v)[0]]
                indexes_right = indexes[np.where(X_indexes_k >= v)[0]]
                distr_left = self.y_distribution(y, indexes_left)
                distr_right = self.y_distribution(y, indexes_right)
                expect = 1.0 / float(indexes.size) * (indexes_left.size * self.impurity_(distr_left) + \
                         indexes_right.size * self.impurity_(distr_left))
                if expect < best_expect:
                    best_expect = expect
                    best_k = k
                    best_v = v

        return best_k, best_v, best_expect

    def grow_tree(self, X, y, indexes, node_index, depth):
        if self.tree_ is None:
            self.tree_ = np.zeros((1,7))
            self.tree_[0, 0] = -1.0
        self.tree_[node_index, DecisionTree.COL_DEPTH] = depth
        y_distr = self.y_distribution(y, indexes)
        self.tree_[node_index, DecisionTree.COL_Y] = self.class_labels_[np.argmax(y_distr)]
        imp = self.impurity_(y_distr)
        #print('impurity:' + str(imp))
        if imp == 0.0:
            return self.tree_

        k, v, expect = self.best_split(X, y, indexes)

        if expect >= imp:
            return self.tree_

        self.tree_[node_index, DecisionTree.COL_SPLIT_FEATURE] = k
        self.tree_[node_index, DecisionTree.COL_SPLIT_VALUE] = v
        nodes_so_far = self.tree_.shape[0]
        self.tree_[node_index, DecisionTree.COL_CHILD_LEFT] = nodes_so_far
        self.tree_[node_index, DecisionTree.COL_CHILD_RIGHT] = nodes_so_far + 1
        self.tree_ = np.r_[self.tree_, np.zeros((2, 7))]
        self.tree_[nodes_so_far, DecisionTree.COL_PARENT] = node_index
        self.tree_[nodes_so_far + 1, DecisionTree.COL_PARENT] = node_index

        X_indexes_k = X[indexes, k]
        indexes_left = indexes[np.where(X_indexes_k < v)[0]]
        indexes_right = indexes[np.where(X_indexes_k >= v)[0]]
        self.grow_tree(X, y, indexes_left, nodes_so_far, depth + 1)
        self.grow_tree(X, y, indexes_right, nodes_so_far + 1, depth + 1)

        return self.tree_

    def fit(self, X, y):
        """
        Uczenie
        :param X:
        :param y: klasy uczace
        :return:
        """
        self.class_labels_ = np.unique(y)
        self.grow_tree(X, y, np.arange(X.shape[0]), 0, 0)

        return self

    def predict(self, X):
        """
        Przewidywanie
        :param X:
        :return:
        """
        return self

    def y_distribution(self, y, indexes):
        distr = np.zeros(self.class_labels_.size)
        y_indexes = y[indexes]
        for i, label in enumerate(self.class_labels_):
            distr[i] = np.where(y_indexes == label)[0].size / float(indexes.size)

        return distr

    def impurity_error(self, y_distr):
        """
        Blad nieczystosci

        :param y_distr: rozklad prawdopodobienstwa nad klasami
        :return:
        """
        return 1.0 - np.max(y_distr)

    def impurity_entropy(self, y_distr):
        """

        :param y_distr: rozklad prawdopodobienstwa
        :return:
        """
        y_distr = y_distr[y_distr > 0.0]

        return -np.sum(y_distr * np.log2(y_distr))

    def impurity_gini(self, y_distr):
        return 1.0 - np.sum(y_distr**2)


if __name__ == '__main__':
    y_distr = np.array([0.5, 0.0, 0.25, 0.25])
    dt = DecisionTree()
    print(dt.impurity_error(y_distr))
    print(dt.impurity_entropy(y_distr))
    print(dt.impurity_gini(y_distr))
