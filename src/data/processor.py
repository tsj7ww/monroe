"""
Data processor module for loss aversion analysis.

This module handles the processing of stock market data for loss aversion analysis,
including event detection, feature engineering, and data preparation for statistical testing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LossAversionProcessor:
    """Class to process stock data for loss aversion analysis."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        market_factor: str = 'SPY',
        volatility_window: int = 20,
        event_threshold: float = 2.0,
        min_lookback: int = 120,
        min_lookahead: int = 20
    ):
        """
        Initialize the processor with configuration parameters.

        Args:
            data_dir: Directory containing processed stock data
            market_factor: Ticker for market benchmark
            volatility_window: Window for volatility calculation (trading days)
            event_threshold: Number of standard deviations for event detection
            min_lookback: Minimum trading days required before event
            min_lookahead: Minimum trading days required after event
        """
        self.data_dir = Path(data_dir)
        self.market_factor = market_factor
        self.volatility_window = volatility_window
        self.event_threshold = event_threshold
        self.min_lookback = min_lookback
        self.min_lookahead = min_lookahead
        
        # Load market factor data
        self.market_data = self._load_market_data()
        
    def _load_market_data(self) -> pd.DataFrame:
        """Load and process market benchmark data."""
        try:
            market_path = self.data_dir / 'processed' / f"{self.market_factor}_processed.parquet"
            df = pd.read_parquet(market_path)
            return self._calculate_market_factors(df)
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            raise
            
    def _calculate_market_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-wide factors from benchmark data."""
        df['Market_Return'] = df['Returns']
        df['Market_Volume'] = df['Relative_Volume']
        df['Market_Volatility'] = df['Volatility_20d']
        return df
            
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect significant price events for loss aversion analysis.
        
        Args:
            df: Stock price and volume data
            
        Returns:
            DataFrame with event indicators
        """
        # Calculate normalized returns
        rolling_std = df['Returns'].rolling(window=self.volatility_window).std()
        normalized_returns = df['Returns'] / rolling_std
        
        # Detect events
        gain_events = normalized_returns > self.event_threshold
        loss_events = normalized_returns < -self.event_threshold
        
        # Create event indicators
        df['Gain_Event'] = gain_events
        df['Loss_Event'] = loss_events
        
        # Add event size
        df['Event_Size'] = np.where(
            gain_events | loss_events,
            normalized_returns,
            np.nan
        )
        
        return df
        
    def calculate_abnormal_metrics(
        self,
        df: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate abnormal returns and volume relative to market.
        
        Args:
            df: Stock data with events
            market_data: Market benchmark data
            
        Returns:
            DataFrame with abnormal metrics
        """
        # Merge with market data
        merged = df.join(market_data[['Market_Return', 'Market_Volume']], how='left')
        
        # Calculate market-adjusted metrics
        merged['Abnormal_Return'] = merged['Returns'] - merged['Market_Return']
        merged['Abnormal_Volume'] = merged['Relative_Volume'] / merged['Market_Volume']
        
        # Calculate market model parameters (beta)
        returns = merged['Returns'].values
        market_returns = merged['Market_Return'].values
        
        try:
            model = sm.OLS(returns, sm.add_constant(market_returns)).fit()
            beta = model.params[1]
            
            # Calculate market model abnormal returns
            merged['Market_Model_AR'] = (
                merged['Returns'] - 
                (model.params[0] + beta * merged['Market_Return'])
            )
        except Exception as e:
            logger.warning(f"Error calculating market model: {str(e)}")
            merged['Market_Model_AR'] = merged['Abnormal_Return']
            
        return merged
        
    def calculate_event_windows(
        self,
        df: pd.DataFrame,
        pre_event: int = 10,
        post_event: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate metrics for event windows around gains and losses.
        
        Args:
            df: Processed stock data with events
            pre_event: Trading days before event
            post_event: Trading days after event
            
        Returns:
            Tuple of DataFrames for gain and loss events
        """
        event_windows = []
        
        for event_type in ['Gain_Event', 'Loss_Event']:
            # Get event dates
            event_dates = df[df[event_type]].index
            
            for event_date in event_dates:
                try:
                    # Get window around event
                    start_idx = df.index.get_loc(event_date) - pre_event
                    end_idx = df.index.get_loc(event_date) + post_event
                    
                    if start_idx < 0 or end_idx >= len(df):
                        continue
                        
                    window_data = df.iloc[start_idx:end_idx + 1].copy()
                    
                    # Add event information
                    window_data['Event_Type'] = event_type
                    window_data['Event_Date'] = event_date
                    window_data['Event_Size'] = df.loc[event_date, 'Event_Size']
                    window_data['Days_From_Event'] = range(-pre_event, post_event + 1)
                    
                    event_windows.append(window_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing event window: {str(e)}")
                    continue
                    
        # Combine all event windows
        if event_windows:
            return pd.concat(event_windows, ignore_index=True)
        else:
            return pd.DataFrame()
            
    def calculate_loss_aversion_metrics(
        self,
        event_windows: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate key loss aversion metrics from event windows.
        
        Args:
            event_windows: Processed event window data
            
        Returns:
            Dictionary of loss aversion metrics
        """
        metrics = {}
        
        # Separate gain and loss events
        gain_events = event_windows[event_windows['Event_Type'] == 'Gain_Event']
        loss_events = event_windows[event_windows['Event_Type'] == 'Loss_Event']
        
        # Calculate volume response ratios
        metrics['volume_response_ratio'] = (
            loss_events['Abnormal_Volume'].mean() /
            gain_events['Abnormal_Volume'].mean()
        )
        
        # Calculate price momentum metrics
        metrics['gain_momentum'] = gain_events['Abnormal_Return'].autocorr()
        metrics['loss_momentum'] = loss_events['Abnormal_Return'].autocorr()
        
        # Calculate statistical significance
        vol_tstat, vol_pval = stats.ttest_ind(
            loss_events['Abnormal_Volume'],
            gain_events['Abnormal_Volume']
        )
        
        metrics['volume_tstat'] = vol_tstat
        metrics['volume_pval'] = vol_pval
        
        return metrics
        
    def process_stock(self, ticker: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Process single stock data for loss aversion analysis.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of processed DataFrame and loss aversion metrics
        """
        try:
            # Load processed stock data
            df = pd.read_parquet(self.data_dir / 'processed' / f"{ticker}_processed.parquet")
            
            # Detect events
            df = self.detect_events(df)
            
            # Calculate abnormal metrics
            df = self.calculate_abnormal_metrics(df, self.market_data)
            
            # Get event windows
            event_windows = self.calculate_event_windows(df)
            
            # Calculate loss aversion metrics
            metrics = self.calculate_loss_aversion_metrics(event_windows)
            
            logger.info(f"Successfully processed {ticker}")
            return df, metrics
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            raise
            
def main():
    """Main function to run the processor."""
    try:
        # Initialize processor
        processor = LossAversionProcessor(
            data_dir='data',
            market_factor='SPY',
            event_threshold=2.0
        )
        
        # Process stocks
        results = {}
        metrics_list = []
        
        # Get list of processed files
        processed_dir = Path('data') / 'processed'
        stock_files = processed_dir.glob('*_processed.parquet')
        
        for stock_file in stock_files:
            ticker = stock_file.stem.replace('_processed', '')
            if ticker != 'SPY':  # Skip market factor
                df, metrics = processor.process_stock(ticker)
                results[ticker] = df
                metrics['ticker'] = ticker
                metrics_list.append(metrics)
                
        # Combine metrics
        all_metrics = pd.DataFrame(metrics_list)
        all_metrics.to_parquet(processed_dir / 'loss_aversion_metrics.parquet')
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()