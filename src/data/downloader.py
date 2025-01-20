"""
Stock market data downloader module for loss aversion analysis.

This module handles the downloading and initial processing of stock market data
using the yfinance API. It includes functionality for batch downloading,
error handling, and data validation.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Union, Optional
import json

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataDownloader:
    """Class to handle stock market data downloading and initial processing."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize the downloader with configuration parameters.

        Args:
            data_dir: Directory to store downloaded data
            start_date: Start date for data download (YYYY-MM-DD)
            end_date: End date for data download (YYYY-MM-DD)
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set date range
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Default to 5 years of data if not specified
            start_datetime = datetime.strptime(self.end_date, '%Y-%m-%d') - timedelta(days=5*365)
            self.start_date = start_datetime.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        # Initialize yfinance
        yf.pdr_override()
        
    def download_sp500_constituents(self) -> List[str]:
        """
        Download current S&P 500 constituents list.
        
        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            tickers = df['Symbol'].str.replace('.', '-').tolist()
            
            # Save tickers list
            with open(self.raw_dir / 'sp500_tickers.json', 'w') as f:
                json.dump(tickers, f)
            
            logger.info(f"Successfully downloaded {len(tickers)} S&P 500 constituents")
            return tickers
            
        except Exception as e:
            logger.error(f"Error downloading S&P 500 constituents: {str(e)}")
            raise
            
    def download_stock_data(
        self,
        tickers: List[str],
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Download stock data for given tickers in batches.
        
        Args:
            tickers: List of stock tickers to download
            batch_size: Number of stocks to download in each batch
            
        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        all_data = {}
        
        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            try:
                # Download batch data
                data = pdr.get_data_yahoo(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    group_by='ticker'
                )
                
                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        if isinstance(data, pd.DataFrame):
                            ticker_data = data.xs(ticker) if len(batch) > 1 else data
                        else:
                            ticker_data = data[ticker]
                            
                        # Validate data
                        if self._validate_data(ticker_data):
                            all_data[ticker] = ticker_data
                            
                            # Save individual ticker data
                            save_path = self.raw_dir / f"{ticker}_data.parquet"
                            ticker_data.to_parquet(save_path)
                            logger.info(f"Successfully downloaded and saved data for {ticker}")
                        else:
                            logger.warning(f"Downloaded data for {ticker} failed validation")
                            
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error downloading batch: {str(e)}")
                continue
                
        return all_data
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate downloaded data meets quality requirements.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Boolean indicating if data passes validation
        """
        if df.empty:
            return False
            
        # Check for minimum number of trading days
        min_days = 100
        if len(df) < min_days:
            return False
            
        # Check for required columns
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
        if not required_columns.issubset(df.columns):
            return False
            
        # Check for excessive missing values
        max_missing_pct = 0.1
        missing_pct = df[list(required_columns)].isnull().mean().max()
        if missing_pct > max_missing_pct:
            return False
            
        return True
        
    def process_raw_data(self, ticker: str) -> pd.DataFrame:
        """
        Process raw data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Processed DataFrame
        """
        try:
            # Load raw data
            raw_path = self.raw_dir / f"{ticker}_data.parquet"
            df = pd.read_parquet(raw_path)
            
            # Calculate daily returns
            df['Returns'] = df['Adj Close'].pct_change()
            
            # Calculate log returns
            df['Log_Returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
            
            # Calculate rolling volatility
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
            
            # Calculate volume metrics
            df['Volume_MA_5d'] = df['Volume'].rolling(window=5).mean()
            df['Relative_Volume'] = df['Volume'] / df['Volume_MA_5d']
            
            # Save processed data
            processed_path = self.processed_dir / f"{ticker}_processed.parquet"
            df.to_parquet(processed_path)
            
            logger.info(f"Successfully processed data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            raise
            
def main():
    """Main function to run the downloader."""
    try:
        # Initialize downloader
        downloader = MarketDataDownloader(
            data_dir='data',
            start_date='2019-01-01',
            end_date=None  # Use current date
        )
        
        # Download S&P 500 constituents
        tickers = downloader.download_sp500_constituents()
        
        # Download stock data
        stock_data = downloader.download_stock_data(tickers)
        
        # Process each stock's data
        for ticker in stock_data:
            downloader.process_raw_data(ticker)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()