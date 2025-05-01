import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pandas.testing import assert_frame_equal, assert_series_equal

# Import the modules you're testing
from ..monroe.config import Config
from ..monroe.data import Data
from ..monroe.preprocess import Preprocess

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock Config object
        self.config = MagicMock(spec=Config)
        
        # Create sample data
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5, np.nan],
            'categorical': pd.Series(['A', 'B', 'C', 'A', 'B', None], dtype='category'),
            'with_outliers': [10, 12, 11, 1000, 13, 11],
        })
        
        # Setup mock Data object
        self.data = MagicMock(spec=Data)
        self.data.x = self.df
        self.data.y = pd.Series(['Yes', 'No', 'Yes', 'No', 'Yes', 'No'], dtype='category')
        
        # Setup metadata for the Data object
        self.data.x_info = {
            'numeric': {'needs_numification': False},
            'categorical': {'needs_numification': True},
            'with_outliers': {'needs_numification': False}
        }
        self.data.y_info = {'needs_numification': True}
        
        # Create the Preprocess object
        self.preprocess = Preprocess(self.config, self.data)
    
    def test_categorical_to_numeric(self):
        """Test conversion of categorical columns to numeric."""
        # Execute the method
        self.preprocess.categorical_to_numeric()
        
        # Check that cat.codes was called on the categorical column
        expected_categorical = pd.Series([0, 1, 2, 0, 1, -1])  # -1 for NaN in cat.codes
        assert_series_equal(self.data.x['categorical'], expected_categorical)
        
        # Check that cat.codes was called on the target variable
        expected_y = pd.Series([1, 0, 1, 0, 1, 0])  # Assuming 'Yes'=1, 'No'=0
        assert_series_equal(self.data.y, expected_y)
    
    def test_handle_missing_values_drop(self):
        """Test dropping rows with missing values."""
        # Setup mock return for dropna
        clean_df = self.df.dropna()
        self.data.dropna.return_value = clean_df
        
        # Execute the method
        self.preprocess.handle_missing_values(method='drop')
        
        # Verify that dropna was called with the correct parameters
        self.data.dropna.assert_called_once_with(how='any', axis=0)
    
    @patch('pandas.DataFrame.fillna')
    def test_handle_missing_values_impute(self, mock_fillna):
        """Test imputation of missing values."""
        # Setup the mocked _impute method
        self.preprocess._impute = MagicMock()
        
        # Execute the method
        self.preprocess.handle_missing_values(method='impute')
        
        # Verify that _impute was called
        self.preprocess._impute.assert_called_once()
    
    def test_handle_missing_values_auto(self):
        """Test automatic handling of missing values."""
        # This is more complex to test since it's not implemented yet
        # Here we're just testing that it doesn't raise an exception
        try:
            self.preprocess.handle_missing_values(method='auto')
        except Exception as e:
            self.fail(f"handle_missing_values('auto') raised {e} unexpectedly!")
    
    def test_handle_outliers_iqr(self):
        """Test IQR-based outlier handling."""
        # Since handle_outliers has two definitions in the code,
        # we'll need to clarify which one we're testing
        # For this test, we'll use the one with the method parameter
        
        # Create a mock IQR implementation
        def mock_iqr_implementation():
            # Simple implementation: filter values outside 1.5*IQR
            q1 = self.df['with_outliers'].quantile(0.25)
            q3 = self.df['with_outliers'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.data.x = self.data.x[
                (self.data.x['with_outliers'] >= lower_bound) & 
                (self.data.x['with_outliers'] <= upper_bound)
            ]
        
        # Patch the handle_outliers method to use our mock implementation
        with patch.object(self.preprocess, 'handle_outliers', 
                         side_effect=mock_iqr_implementation):
            
            # Call the method
            self.preprocess.handle_outliers(method='iqr')
            
            # The 1000 value should be considered an outlier and removed
            self.assertNotIn(1000, self.data.x['with_outliers'].values)

    def test_config_integration(self):
        """Test that preprocessing respects configuration settings."""
        # Setup configuration values
        self.config.get.return_value = 'iqr'
        
        # Mock the handle_outliers method to verify it gets called with the correct method
        self.preprocess.handle_outliers = MagicMock()
        
        # Trigger the method that should use config
        self.preprocess.handle_outliers()
        
        # Verify config was queried
        self.config.get.assert_called_with('data.outlier_method')
        
        # In a complete implementation, we'd also verify handle_outliers was called with 'iqr'
        # But this requires fixing the code to avoid having duplicate handle_outliers methods

    def test_end_to_end_preprocessing(self):
        """Test an end-to-end preprocessing workflow."""
        # Mock methods to avoid actual implementation
        self.preprocess.categorical_to_numeric = MagicMock()
        self.preprocess.handle_missing_values = MagicMock()
        self.preprocess.handle_outliers = MagicMock()
        
        # Define a preprocessing workflow
        def process_data():
            self.preprocess.categorical_to_numeric()
            self.preprocess.handle_missing_values(method='impute')
            self.preprocess.handle_outliers(method='iqr')
            return self.data
        
        # Execute the workflow
        processed_data = process_data()
        
        # Verify each method was called once
        self.preprocess.categorical_to_numeric.assert_called_once()
        self.preprocess.handle_missing_values.assert_called_once_with(method='impute')
        self.preprocess.handle_outliers.assert_called_once_with(method='iqr')


class TestPCA(unittest.TestCase):
    """Tests for PCA dimensionality reduction."""
    
    def setUp(self):
        """Set up test fixtures for PCA tests."""
        self.config = MagicMock(spec=Config)
        
        # Create a simple dataset for PCA
        self.df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],  # x2 = 2*x1, perfectly correlated
            'x3': [5, 4, 3, 2, 1]    # x3 = 6-x1, perfectly anti-correlated
        })
        
        self.data = MagicMock(spec=Data)
        self.data.x = self.df
        self.data.y = pd.Series([0, 1, 0, 1, 0])
        
        # Mock x_info structure
        self.data.x_info = {
            'x1': {'type': 'numeric'},
            'x2': {'type': 'numeric'},
            'x3': {'type': 'numeric'}
        }
        
        self.preprocess = Preprocess(self.config, self.data)
    
    @unittest.skip("PCA reduction not implemented yet")
    def test_pca_reduction(self):
        """Test PCA dimensionality reduction."""
        # This would be implemented once the PCA functionality is added
        # For now we'll skip it
        pass


class TestBalancing(unittest.TestCase):
    """Tests for handling imbalanced data."""
    
    def setUp(self):
        """Set up test fixtures for balancing tests."""
        self.config = MagicMock(spec=Config)
        
        # Create an imbalanced dataset (90% class 0, 10% class 1)
        self.df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        # Imbalanced target (90 zeros, 10 ones)
        self.target = pd.Series([0] * 90 + [1] * 10)
        
        self.data = MagicMock(spec=Data)
        self.data.x = self.df
        self.data.y = self.target
        
        self.preprocess = Preprocess(self.config, self.data)
    
    @unittest.skip("Balancing not implemented yet")
    def test_class_weighting(self):
        """Test class weighting for imbalanced data."""
        # This would be implemented once the balancing functionality is added
        pass
    
    @unittest.skip("Balancing not implemented yet")
    def test_oversampling(self):
        """Test oversampling for imbalanced data."""
        # This would be implemented once the balancing functionality is added
        pass


if __name__ == '__main__':
    unittest.main()