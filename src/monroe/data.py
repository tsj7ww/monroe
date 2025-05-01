"""
Data API and metadata
"""

import os
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from .config import Config

class Data:
    def __init__(self,
                 data_fpath: str,
                 target_column: str,
                 config: Config):
        self.data_fpath = data_fpath
        self.target_column = target_column
        self.config = config

    def load_data(self):
        self.data = dd.read_csv(self.data_fpath)

    def parse_data(self):
        self.x = self.data.drop([target_column], axis=1)
        self.x_original = self.x
        self.y = self.data[target_column]
        self.y_original = self.y
        self.x_info = {
            i: {'dtype': str(j)}
            for i, j in self.x.dtypes.to_dict().items()
        }
        self.y_info = {'dtype': str(self.y.dtypes)}

    def get_column_info(
        self, max_values_as_discrete: int
    ):
        for col, info in self.x_info.items():
            if info['dtype'] == 'object':
                info['category'] = 'discrete'
                info['needs_numification'] = True
            else:
                info['needs_numification'] = False
                unique_values = len(self.x[col].unique())
                if unique_values <= max_values_as_discrete:
                    info['category'] = 'discrete'
                else:
                    info['category'] = 'continuous'
        if self.y_info['dtype'] == 'object':
            self.y_info['needs_numification'] = True

    def regression_or_classification(
        self, max_values_as_classification: int
    ):
        unique_values = len(self.y.unique())
        if unique_values <= max_values_as_classification:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'

    def split_data(self, test_size=0.2, random_state=42):
        (self.x_train,
         self.x_test,
         self.y_train,
         self.y_test) = train_test_split(
             self.x, self.y, test_size=test_size,
             random_state=random_state
         )