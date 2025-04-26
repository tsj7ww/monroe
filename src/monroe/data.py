import os
import dask.dataframe as dd

class Data:
    def __init__(self, data_fpath, target_column):
        self.data_fpath = data_fpath
        self.target_column = target_column

    def load_data(self):
        self.data = dd.read_csv(self.data_fpath)

    def test_train_split(self):
        