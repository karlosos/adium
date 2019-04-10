from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class DecisionTree(BaseEstimator, ClassifierMixin):
    """
    Drzewo decyzyjne CART.
    """
    COL_PARENT = 0
    COL_SPLIT_FEATURE = 1
    COL_SPLIT_VALUE = 2
    COL_CHILD_LEFT = 3
    COL_CHILD_RIGHT = 4
    COL_Y = 5
    COL_DEPTH = 6

    def __init__(self, impurity='impurity_entropy', max_depth = None, min_node_examples=0.01, pruning=False):
        self.tree_ = None
        self.class_labels_ = None
        self.impurity_ = getattr(self, impurity)
        self.max_depth_ = max_depth
        self.min_node_examples_ = min_node_examples

    def best_split(self, X, y, indexes):
        """
        Znajdź najlepsze rozcięcie dla węzła.

        Generuje wszystkie możliwe cięcia i liczy ich wartości oczekiwane nieczystości. Wybiera to rozwiązanie
        z najmniejszą nieczystością.

        :param X: dane
        :param y: etykiety
        :param indexes:
        :return:
        """
        n = X.shape[1]
        best_k = None
        best_v = None
        best_expect = np.inf

        for k in range(n):
            X_indexes_k = X[indexes, k]
            u = np.unique(X[indexes, k])
            vals = 0.5 * (u[:-1] + u[1:])
            for v in vals:
                indexes_left = indexes[np.where(X_indexes_k < v)[0]]
                indexes_right = indexes[np.where(X_indexes_k >= v)[0]]
                distr_left = self.y_distribution(y, indexes_left)
                distr_right = self.y_distribution(y, indexes_right)
                expect = 1.0 / float(indexes.size) * (indexes_left.size * self.impurity_(distr_left) +
                                                      indexes_right.size * self.impurity_(distr_right))
                if expect < best_expect:
                    best_expect = expect
                    best_k = k
                    best_v = v

        return best_k, best_v, best_expect

    def grow_tree(self, X, y, indexes, node_index, depth):
        """
        Budowanie pełnego drzewa CART

        1. Dla węzła wylicz rozkład p(y|t) i wyznacz najbardziej prawdopodobną klasę
        2. Oblicz nieczystość dla węzła t, jeżeli wynosi 0 to przerwij rekurencję.
        3. Wybierz najlepsze cięcie w węźle t.
        4. Jeżeli wartość oczekiwana nieczystości dla najlepszego cięcia równa się nieczystości
        aktualnego węzła to przerwij gałąź rekurencji.
        5. Wykonaj najlepsze cięcie - dołącz do drzewa węzły t_l(k_best, v_best), t_r(k_best, v_best)
        6. Wywołaj rekurencję na rzecz lewego potomka.
        7. Wywołaj rekurencję na rzecz prawego potomka.

        :param X:
        :param y:
        :param indexes: dostępne indeksy do sprawdzenia, na początku wszystkie możliwe. Z każdą iteracją zmniejszane
        o połowę.
        :param node_index: aktualny węzeł
        :param depth:
        :return: drzewo w formie macierzy
        """
        if self.tree_ is None:
            self.tree_ = np.zeros((1, 7))
            self.tree_[0, 0] = -1.0

        y_distr = self.y_distribution(y, indexes)
        imp = self.impurity_(y_distr)

        self.tree_[node_index, DecisionTree.COL_DEPTH] = depth  # todo sprawdzić
        # skojarz z węzłem najbardziej prawdopodobną w nim klasę
        self.tree_[node_index, DecisionTree.COL_Y] = self.class_labels_[np.argmax(y_distr)]

        if imp == 0.0 or ((self.max_depth_ is not None) and (depth == self.max_depth_)) or \
                (indexes.size < self.min_node_examples_ * X.shape[0]):
            return self.tree_

        # znajdź najlepsze rozcięcie
        # k - numer zmiennej
        # v - możiwe wartości
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
        self.grow_tree(X, y, indexes_left, nodes_so_far, depth)
        self.grow_tree(X, y, indexes_right, nodes_so_far + 1, depth + 1)

        return self.tree_

    def fit(self, X, y):
        """
        Uczymy model na danych X i labelkach y
        """
        self.class_labels_ = np.unique(y)
        self.grow_tree(X, y, np.arange(X.shape[0]), 0, 0)

        return self

    def predict(self, X):
        """
        Przewidujemy labelki dla danych X
        """
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self.predict_x(X[i])

        return predictions

    def predict_x(self, x):
        node_index = 0
        while True:
            if self.tree_[node_index, DecisionTree.COL_CHILD_LEFT] == 0.0:
                return self.tree_[node_index, DecisionTree.COL_Y]
            k, v = int(self.tree_[node_index, DecisionTree.COL_SPLIT_FEATURE]), self.tree_[node_index,
                                                                                           DecisionTree.COL_SPLIT_VALUE]

            if x[k] < v:
                node_index = int(self.tree_[node_index, DecisionTree.COL_CHILD_LEFT])
            else:
                node_index = int(self.tree_[node_index, DecisionTree.COL_CHILD_RIGHT])

    def y_distribution(self, y, indexes):
        """
        Prawdopodobienstwa warunkowe klasy y pod warunkiem że jesteśmy w węzłach o indeksach. P(y|t)
        """
        distr = np.zeros(self.class_labels_.size)
        y_indexes = y[indexes]
        for i, label in enumerate(self.class_labels_):
            distr[i] = np.where(y_indexes == label)[0].size / float(indexes.size)

        return distr

    def impurity_error(self, y_distr):
        """
        Błąd klasyfikacji

        :param y_distr: rozklad prawdopodobienstwa nad klasami P(y|t)
        """
        return 1.0 - np.max(y_distr)

    def impurity_entropy(self, y_distr):
        """
        Entropia - miara nieuporządkowania

        :param y_distr: rozklad prawdopodobienstwa nad klasami P(y|t)
        """
        y_distr = y_distr[y_distr > 0.0]

        return -np.sum(y_distr * np.log2(y_distr))

    def impurity_gini(self, y_distr):
        """
        Indeks giniego

        :param y_distr: rozklad prawdopodobienstwa nad klasami P(y|t)
        """
        return 1.0 - np.sum(y_distr**2)


if __name__ == '__main__':
    y_distr = np.array([0.5, 0.0, 0.25, 0.25])
    dt = DecisionTree()
    print(dt.impurity_error(y_distr))
    print(dt.impurity_entropy(y_distr))
    print(dt.impurity_gini(y_distr))
