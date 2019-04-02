import pickle
import os


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
