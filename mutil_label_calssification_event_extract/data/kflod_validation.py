import pandas as pd
from sklearn.model_selection import KFold

if __name__ == '__main__':
    n_splits = 5
    skf = KFold(n_splits=n_splits)
