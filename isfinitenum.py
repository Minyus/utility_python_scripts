import numpy as np


def isfinitenum(x):
    # wrapper of numpy.isfinite() that supports any types such as None, str, list, etc.
    return isinstance(x, (float, int)) and np.isfinite(x)


if __name__ == "__main__":
    for x in [None, np.nan, np.Inf, -np.Inf, 'text', [1,2], -2, 0.5]:
        print(x, isfinitenum(x))