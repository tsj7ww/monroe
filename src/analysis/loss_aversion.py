"""
Loss aversion analysis module.

This module implements the core analysis methods for investigating loss aversion
in stock market behavior, including statistical tests, regression analysis,
and result aggregation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LossAversionAnalyzer:
    """Class to analyze loss aversion patterns in stock market data."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        significance_level: float = 0.05
    ):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            data_dir: Directory containing processed stock data
            output_dir: Directory for analysis outputs
            significance_level: Statistical significance threshold
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.significance_level = significance_level
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_volume_response(
        self,
        event_data: pd.DataFrame,
        control_variables: List[str] = None
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze asymmetric volume response to gains and losses.
        
        Args:
            event_data: Processed event window data
            control_variables: Additional variables to control for
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Prepare data
        gain_events = event_data[event_data['Event_Type'] == 'Gain_Event']
        loss_events = event_data[event_data['Event_Type'] == 'Loss_Event']
        
        # Basic volume response ratio
        results['volume_ratio'] = (
            loss_events['Abnormal_Volume'].mean() /
            gain_events['Abnormal_Volume'].mean()
        )
        
        # Statistical test of difference
        t_stat, p_value = stats.ttest_ind(
            loss_events['Abnormal_Volume'],
            gain_events['Abnormal_Volume']
        )
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        
        # Regression analysis with controls
        if control_variables:
            X = pd.get_dummies(event_data['Event_Type'])
            for var in control_variables:
                X[var] = event_data[var]
            
            y = event_data['Abnormal_Volume']
            
            model = sm.OLS(y, sm.add_constant(X)).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': 5}
            )
            results['regression_results'] = model
            
        # Time decay analysis
        time_decay = (
            event_data.groupby(['Event_Type', 'Days_From_Event'])
            ['Abnormal_Volume']
            .mean()
            .unstack(0)
        )
        results['time_decay'] = time_decay
        
        return results
        
    def analyze_price_momentum(
        self,
        event_data: pd.DataFrame,
        window_size: int = 5
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze price momentum patterns following gains and losses.
        
        Args:
            event_data: Processed event window data
            window_size: Rolling window size for momentum calculation
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        for event_type in ['Gain_Event', 'Loss_Event']:
            event_subset = event_data[event_data['Event_Type'] == event_type]
            
            # Calculate post-event returns
            cumulative_returns = (
                event_subset[event_subset['Days_From_Event'] > 0]
                .groupby('Event_Date')['Abnormal_Return']
                .cumsum()
            )
            
            # Calculate momentum metrics
            results[f'{event_type.lower()}_momentum'] = {
                'mean_reversion': cumulative_returns.mean(),
                'autocorrelation': cumulative_returns.autocorr(),
                'volatility': cumulative_returns.std()
            }
            
            # Probability of trend continuation
            returns_sign = np.sign(event_subset['Event_Size'])
            subsequent_signs = np.sign(cumulative_returns)
            results[f'{event_type.lower()}_continuation'] = (
                (returns_sign == subsequent_signs).mean()
            )
            
        return results
        
    def analyze_magnitude_effects(
        self,
        event_data: pd.DataFrame,
        quantiles: int = 5
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze how loss aversion varies with event magnitude.
        
        Args:
            event_data: Processed event window data
            quantiles: Number of quantiles for magnitude analysis
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Create magnitude quantiles
        event_data['Magnitude_Quantile'] = pd.qcut(
            abs(event_data['Event_Size']),
            quantiles,
            labels=False
        )
        
        # Analyze volume response by magnitude
        volume_by_magnitude = (
            event_data.groupby(['Event_Type', 'Magnitude_Quantile'])
            ['Abnormal_Volume']
            .agg(['mean', 'std', 'count'])
        )
        results['volume_by_magnitude'] = volume_by_magnitude
        
        # Test for non-linearity
        for event_type in ['Gain_Event', 'Loss_Event']:
            subset = event_data[event_data['Event_Type'] == event_type]
            
            # Polynomial regression
            X = abs(subset['Event_Size'])
            X_squared = X ** 2
            y = subset['Abnormal_Volume']
            
            model = sm.OLS(
                y,
                sm.add_constant(pd.concat([X, X_squared], axis=1))
            ).fit()
            
            results[f'{event_type.lower()}_nonlinearity'] = {
                'coefficients': model.params,
                'p_values': model.pvalues
            }
            
        return results
        
    def analyze_sector_patterns(
        self,
        event_data: pd.DataFrame,
        sector_info: pd.DataFrame
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze sector-specific loss aversion patterns.
        
        Args:
            event_data: Processed event window data
            sector_info: DataFrame with sector classifications
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Merge sector information
        event_data = event_data.merge(
            sector_info,
            left_on='ticker',
            right_index=True
        )
        
        # Calculate sector-specific metrics
        sector_metrics = []
        
        for sector in event_data['sector'].unique():
            sector_data = event_data[event_data['sector'] == sector]
            
            # Calculate volume response ratio
            gain_vol = sector_data[sector_data['Event_Type'] == 'Gain_Event']['Abnormal_Volume'].mean()
            loss_vol = sector_data[sector_data['Event_Type'] == 'Loss_Event']['Abnormal_Volume'].mean()
            
            sector_metrics.append({
                'sector': sector,
                'volume_ratio': loss_vol / gain_vol,
                'gain_volume': gain_vol,
                'loss_volume': loss_vol
            })
            
        results['sector_metrics'] = pd.DataFrame(sector_metrics)
        
        # Test for sector differences
        model = sm.OLS(
            event_data['Abnormal_Volume'],
            pd.get_dummies(event_data[['sector', 'Event_Type']], drop_first=True)
        ).fit()
        
        results['sector_tests'] = {
            'coefficients': model.params,
            'p_values': model.pvalues
        }
        
        return results
        
    def analyze_temporal_stability(
        self,
        event_data: pd.DataFrame,
        window_size: int = 60
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze temporal stability of loss aversion patterns.
        
        Args:
            event_data: Processed event window data
            window_size: Rolling window size in days
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Sort data by date
        event_data = event_data.sort_index()
        
        # Calculate rolling volume response ratio
        rolling_metrics = []
        
        for end_idx in range(window_size, len(event_data)):
            window = event_data.iloc[end_idx-window_size:end_idx]
            
            gain_vol = window[window['Event_Type'] == 'Gain_Event']['Abnormal_Volume'].mean()
            loss_vol = window[window['Event_Type'] == 'Loss_Event']['Abnormal_Volume'].mean()
            
            rolling_metrics.append({
                'date': event_data.index[end_idx],
                'volume_ratio': loss_vol / gain_vol
            })
            
        results['rolling_metrics'] = pd.DataFrame(rolling_metrics)
        
        # Test for structural breaks
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        
        cusum_test = breaks_cusumolsresid(
            results['rolling_metrics']['volume_ratio'].values,
            ddof=0
        )
        
        results['structural_breaks'] = {
            'test_statistic': cusum_test[0],
            'p_value': cusum_test[1]
        }
        
        return results
        
    def run_full_analysis(
        self,
        save_results: bool = True
    ) -> Dict[str, Dict]:
        """
        Run complete loss aversion analysis.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Load processed data
            event_data = pd.read_parquet(
                self.data_dir / 'processed' / 'event_windows.parquet'
            )
            sector_info = pd.read_parquet(
                self.data_dir / 'processed' / 'sector_info.parquet'
            )
            
            # Run analyses
            results = {}
            
            results['volume_response'] = self.analyze_volume_response(
                event_data,
                control_variables=['Market_Volatility']
            )
            
            results['price_momentum'] = self.analyze_price_momentum(
                event_data
            )
            
            results['magnitude_effects'] = self.analyze_magnitude_effects(
                event_data
            )
            
            results['sector_patterns'] = self.analyze_sector_patterns(
                event_data,
                sector_info
            )
            
            results['temporal_stability'] = self.analyze_temporal_stability(
                event_data
            )
            
            # Save results if requested
            if save_results:
                self._save_results(results)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise
            
    def _save_results(self, results: Dict):
        """Save analysis results to files."""
        try:
            # Create results directory structure
            for analysis_type in results:
                output_path = self.output_dir / analysis_type
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save DataFrames to parquet
                for key, value in results[analysis_type].items():
                    if isinstance(value, pd.DataFrame):
                        value.to_parquet(output_path / f"{key}.parquet")
                        
            # Save summary statistics
            summary = {
                analysis_type: {
                    k: v for k, v in results[analysis_type].items()
                    if isinstance(v, (int, float, str))
                }
                for analysis_type in results
            }
            
            pd.to_pickle(
                summary,
                self.output_dir / 'summary_statistics.pkl'
            )
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    """Main function to run the analysis."""
    try:
        # Initialize analyzer
        analyzer = LossAversionAnalyzer(
            data_dir='data',
            output_dir='results'
        )
        
        # Run analysis
        results = analyzer.run_full_analysis(save_results=True)
        
        # Log completion
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()