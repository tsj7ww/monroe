"""
Data preprocessing
- Cleaning: dtype changes, outliers
- Transformation: normalization, log,
    binning, categorical encoding
- Handle missing data
- Reduction (e.g. PCA)
- Balancing (e.g. weighting / imbalanced)
"""

from .config import Config
from .data import Data

class Preprocess:
    def __init__(self,
                 config: Config,
                 data: Data):
        self.config = config
        self.data = data

    def categorical_to_numeric(self):
        for col, info in self.data.x_info.items():
            if info['needs_numification']:
                self.x[col] = self.x[col].cat.codes
        if self.data.y_info['needs_numification']:
            self.data.y = self.data.y.cat.codes

    def handle_outliers(self):
        outlier_method = self.config.get('data.outliers.method')
        for col, info in self.data.x_info.items():
            if outlier_method == 'auto':
                None
            elif outlier_method == 'drop':
                None
            elif outlier_method == 'iqr':
                self.data.x[col] = self._iqr_cap_floor(
                    self.data.x[col],
                    lower_iqr=self.config.get('data.outliers.lower_iqr'),
                    upper_idr=self.config.get('data.outliers.upper_iqr'),
                    factor=self.config.get('data.outliers.iqr_factor'),
                )
    def _iqr_cap_floor(
        series,
        lower_iqr: float,
        upper_iqr: float,
        factor: float,
    ):
        lower_q = series.quantile(lower_iqr)
        upper_q = series.quantile(upper_iqr)
        iqr = upper_q - lower_q
        lower = lower_q - (factor * iqr)
        upper = upper_q + (factor * iqr)
        return series.clip(lower=lower, upper=upper)

    def normalize(self):
        None

    def log_transform(self):
        None

    def binning(self):
        None

    def categorical_encoding(self):
        None

    def handle_missing_values(self, method: str):
        if method == 'auto':
            # understand missing data pattern
            # -> random or not random
            # then determine if imputation or dropping
            # results in better prediction results
            None
        elif method == 'drop':
            self.data.dropna(how='any', axis=0)
        elif method == 'impute':
            self._impute()

    def _impute(self):
        None

    def pca_reduction(self):
        None

    def balance_target(self):
        None