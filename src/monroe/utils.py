# utils.py

import os
import re
import json
import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import joblib
from contextlib import contextmanager
import time
import hashlib
import psutil
import gc
import random
import string

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up global logger
def setup_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger
logger = setup_logger("automl_utils")

# ----- DATA LOADING AND PROCESSING UTILITIES -----

def detect_file_type(file_path: str) -> str:
    """
    Detect file type from file extension
    
    Args:
        file_path: Path to the data file
        
    Returns:
        File type string: csv, parquet, json, excel, or unknown
    """
    file_path = file_path.lower()
    if file_path.endswith('.csv'):
        return 'csv'
    elif file_path.endswith('.parquet'):
        return 'parquet'
    elif file_path.endswith('.json'):
        return 'json'
    elif file_path.endswith(('.xls', '.xlsx')):
        return 'excel'
    else:
        return 'unknown'

def load_data(file_path: str, config: Dict[str, Any], 
              use_dask: bool = True) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        config: Configuration dictionary
        use_dask: Whether to use Dask for large datasets
        
    Returns:
        Pandas or Dask DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    
    file_type = detect_file_type(file_path)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    
    # Use dask for large files if enabled
    use_dask_for_this = use_dask and file_size > config.get("data.dask_threshold_mb", 100)
    
    try:
        if file_type == 'csv':
            if use_dask_for_this:
                return dd.read_csv(
                    file_path, 
                    blocksize=config.get("data.chunk_size", "100MB"),
                    assume_missing=True
                )
            else:
                return pd.read_csv(file_path)
                
        elif file_type == 'parquet':
            if use_dask_for_this:
                return dd.read_parquet(file_path)
            else:
                return pd.read_parquet(file_path)
                
        elif file_type == 'json':
            if use_dask_for_this:
                return dd.read_json(file_path, blocksize=config.get("data.chunk_size", "100MB"))
            else:
                return pd.read_json(file_path)
                
        elif file_type == 'excel':
            # Dask doesn't support Excel files well, always use pandas
            return pd.read_excel(file_path)
            
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_dataset_info(df: Union[pd.DataFrame, dd.DataFrame], sample_size: int = 10000) -> Dict[str, Any]:
    """
    Analyze dataset and return metadata
    
    Args:
        df: Input dataframe (Pandas or Dask)
        sample_size: Number of rows to sample for analysis
        
    Returns:
        Dictionary with dataset properties
    """
    is_dask = isinstance(df, dd.DataFrame)
    
    # Sample if using Dask
    if is_dask:
        df_sample = df.sample(frac=min(1.0, sample_size / df.shape[0].compute()))
        df_sample = df_sample.compute()
        total_rows = df.shape[0].compute()
        memory_usage = "N/A (Dask DataFrame)"
    else:
        # Sample if large pandas DataFrame
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size)
        else:
            df_sample = df
        total_rows = len(df)
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    # Get column information
    columns_info = {}
    for col in df_sample.columns:
        col_data = df_sample[col]
        missing_count = col_data.isna().sum()
        unique_values = len(col_data.dropna().unique())
        
        # Determine data type category
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                dtype_category = "integer"
            elif pd.api.types.is_float_dtype(col_data):
                dtype_category = "float"
            else:
                dtype_category = "numeric"
        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            dtype_category = "categorical"
        elif pd.api.types.is_datetime64_dtype(col_data):
            dtype_category = "datetime"
        elif pd.api.types.is_bool_dtype(col_data):
            dtype_category = "boolean"
        else:
            dtype_category = "other"
        
        # Store column info
        columns_info[col] = {
            "dtype": str(col_data.dtype),
            "dtype_category": dtype_category,
            "missing_count": int(missing_count),
            "missing_percentage": float((missing_count / len(df_sample)) * 100),
            "unique_values": int(unique_values),
            "is_unique": unique_values == len(df_sample) - missing_count,
            "sample_values": list(col_data.dropna().sample(min(5, len(col_data.dropna()))).values)
        }
        
        # Add statistics for numeric columns
        if dtype_category in ["integer", "float", "numeric"]:
            col_data_clean = col_data.dropna()
            if len(col_data_clean) > 0:
                columns_info[col].update({
                    "min": float(col_data_clean.min()),
                    "max": float(col_data_clean.max()),
                    "mean": float(col_data_clean.mean()),
                    "median": float(col_data_clean.median()),
                    "std": float(col_data_clean.std()),
                    "skew": float(col_data_clean.skew() if hasattr(col_data_clean, 'skew') else 0)
                })
    
    # Dataset overall info
    dataset_info = {
        "total_rows": int(total_rows),
        "total_columns": len(df.columns),
        "memory_usage_mb": memory_usage if not isinstance(memory_usage, str) else memory_usage,
        "numeric_columns": sum(1 for col_info in columns_info.values() 
                              if col_info["dtype_category"] in ["integer", "float", "numeric"]),
        "categorical_columns": sum(1 for col_info in columns_info.values() 
                                  if col_info["dtype_category"] == "categorical"),
        "datetime_columns": sum(1 for col_info in columns_info.values() 
                               if col_info["dtype_category"] == "datetime"),
        "boolean_columns": sum(1 for col_info in columns_info.values() 
                              if col_info["dtype_category"] == "boolean"),
        "other_columns": sum(1 for col_info in columns_info.values() 
                            if col_info["dtype_category"] == "other"),
        "missing_cells": int(df_sample.isna().sum().sum()),
        "missing_cells_percentage": float((df_sample.isna().sum().sum() / (df_sample.shape[0] * df_sample.shape[1])) * 100),
        "columns_info": columns_info
    }
    
    return dataset_info

def infer_target_column(dataset_info: Dict[str, Any]) -> str:
    """
    Attempt to identify target column based on column names and properties
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        Suggested target column name
    """
    # Common target column names
    target_keywords = [
        'target', 'label', 'class', 'y', 'outcome', 
        'dependent', 'response', 'output', 'result'
    ]
    
    columns_info = dataset_info["columns_info"]
    
    # First, look for exact matches
    for keyword in target_keywords:
        if keyword in columns_info:
            return keyword
    
    # Then look for partial matches at the end of column names
    for col in columns_info:
        for keyword in target_keywords:
            if col.lower().endswith('_' + keyword) or col.lower() == keyword:
                return col
    
    # If we have a boolean column with few unique values, it's likely a target
    for col, info in columns_info.items():
        if (info["dtype_category"] in ["boolean", "categorical"] and 
            1 < info["unique_values"] <= 10):
            return col
    
    # If we have a special column name like 'fraud', 'churn', etc.
    special_targets = ['fraud', 'churn', 'default', 'converted', 'clicked', 'purchased']
    for col in columns_info:
        if col.lower() in special_targets:
            return col
    
    # Last resort: the last column is often the target
    columns = list(columns_info.keys())
    return columns[-1]

def infer_problem_type(target_col_info: Dict[str, Any]) -> str:
    """
    Infer the problem type (classification or regression)
    
    Args:
        target_col_info: Target column information
        
    Returns:
        Problem type: 'classification' or 'regression'
    """
    dtype_category = target_col_info["dtype_category"]
    unique_values = target_col_info["unique_values"]
    
    # Categorical, boolean, or few unique numeric values suggests classification
    if dtype_category in ["categorical", "boolean"] or (
        dtype_category in ["integer", "numeric"] and unique_values <= 10
    ):
        if unique_values == 2:
            return "binary_classification"
        elif unique_values > 2:
            return "multiclass_classification"
    
    # Otherwise, assume regression
    return "regression"

# ----- METRIC UTILITIES -----

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None, 
                     problem_type: str = "binary_classification") -> Dict[str, float]:
    """
    Calculate performance metrics based on problem type
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for classification)
        problem_type: Type of problem (binary_classification, multiclass_classification, regression)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    try:
        if problem_type in ["binary_classification", "multiclass_classification"]:
            # Classification metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            if problem_type == "binary_classification":
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
                
                if y_prob is not None:
                    # If we have probabilities, also calculate AUC
                    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                        # If we have probabilities for each class, take the positive class
                        y_prob_positive = y_prob[:, 1]
                    else:
                        y_prob_positive = y_prob
                    
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_positive))
            else:
                # Multiclass
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
                metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                
                if y_prob is not None and len(np.unique(y_true)) > 1:
                    try:
                        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
                    except:
                        logger.warning("Could not calculate ROC AUC for multiclass")
        
        elif problem_type == "regression":
            # Regression metrics
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            
            # Mean Absolute Percentage Error, avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
                metrics["mape"] = float(mape if not np.isinf(mape) and not np.isnan(mape) else 0)
        
        else:
            logger.warning(f"Unknown problem type: {problem_type}")
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
    
    return metrics

def get_best_metric(metrics_dict: Dict[str, float], problem_type: str) -> Tuple[str, float]:
    """
    Get the most important metric based on problem type
    
    Args:
        metrics_dict: Dictionary of metrics
        problem_type: Type of problem
        
    Returns:
        Tuple of (metric_name, metric_value)
    """
    if problem_type == "binary_classification":
        if "roc_auc" in metrics_dict:
            return "roc_auc", metrics_dict["roc_auc"]
        else:
            return "f1", metrics_dict.get("f1", 0.0)
    
    elif problem_type == "multiclass_classification":
        if "roc_auc_ovr" in metrics_dict:
            return "roc_auc_ovr", metrics_dict["roc_auc_ovr"]
        else:
            return "f1_macro", metrics_dict.get("f1_macro", 0.0)
    
    elif problem_type == "regression":
        return "rmse", metrics_dict.get("rmse", float('inf'))
    
    else:
        logger.warning(f"Unknown problem type for metrics: {problem_type}")
        # Return first metric
        first_key = next(iter(metrics_dict))
        return first_key, metrics_dict[first_key]

# ----- SYSTEM UTILITIES -----

def get_system_info() -> Dict[str, Any]:
    """
    Get system information including CPU, memory, etc.
    
    Returns:
        Dictionary with system information
    """
    import platform
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_used_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_used_percent": disk.percent
        }
        
        # Check for GPU if possible
        try:
            import torch
            system_info["gpu_available"] = torch.cuda.is_available()
            if system_info["gpu_available"]:
                system_info["gpu_count"] = torch.cuda.device_count()
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
        except:
            system_info["gpu_available"] = False
        
        return system_info
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}

def memory_usage(as_gb: bool = True) -> float:
    """
    Get current memory usage of the process
    
    Args:
        as_gb: Whether to return as GB (True) or MB (False)
        
    Returns:
        Memory usage in GB or MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    if as_gb:
        return memory_info.rss / (1024 ** 3)  # GB
    else:
        return memory_info.rss / (1024 ** 2)  # MB

@contextmanager
def timer(name: str = None, logger: logging.Logger = None) -> None:
    """
    Context manager for timing code blocks
    
    Args:
        name: Name for the timer
        logger: Logger to use
    """
    start = time.time()
    yield
    end = time.time()
    
    duration = end - start
    
    # Format duration based on magnitude
    if duration < 0.001:
        duration_str = f"{duration*1000000:.2f} μs"
    elif duration < 1:
        duration_str = f"{duration*1000:.2f} ms"
    elif duration < 60:
        duration_str = f"{duration:.2f} sec"
    elif duration < 3600:
        duration_str = f"{duration/60:.2f} min"
    else:
        duration_str = f"{duration/3600:.2f} hrs"
    
    message = f"Time [{name}]: {duration_str}"
    
    if logger:
        logger.info(message)
    else:
        print(message)

def clean_memory() -> None:
    """Force garbage collection to free memory"""
    gc.collect()

# ----- FILE AND PATH UTILITIES -----

def ensure_dir(directory: str) -> str:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_unique_filename(base_path: str, extension: str = "") -> str:
    """
    Generate unique filename by appending timestamp
    
    Args:
        base_path: Base path and filename
        extension: File extension
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    if extension:
        base, ext = os.path.splitext(base_path)
        if ext:
            # If base_path already has an extension, replace it
            return f"{base}_{timestamp}{extension}"
        else:
            # If base_path has no extension, add the new one
            return f"{base_path}_{timestamp}{extension}"
    else:
        return f"{base_path}_{timestamp}"

def save_dict_to_file(data: Dict[str, Any], filepath: str, format: str = "auto") -> str:
    """
    Save dictionary to file in JSON or YAML format
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
        format: 'json', 'yaml', or 'auto' (determined from extension)
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Determine format from extension if auto
    if format == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = 'json'
    
    try:
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"Saved data to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving dict to file: {str(e)}")
        raise

def load_dict_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON or YAML file
    
    Args:
        filepath: Path to file
        
    Returns:
        Loaded dictionary
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        elif ext == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")
    
    except Exception as e:
        logger.error(f"Error loading dict from file: {str(e)}")
        raise

def save_model(model: Any, filepath: str) -> str:
    """
    Save model to file using joblib
    
    Args:
        model: Model object
        filepath: Path to save model
        
    Returns:
        Path to saved model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        logger.debug(f"Model saved to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str) -> Any:
    """
    Load model from file using joblib
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(filepath):
        logger.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        model = joblib.load(filepath)
        logger.debug(f"Model loaded from {filepath}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_file_hash(filepath: str) -> str:
    """
    Calculate MD5 hash of file
    
    Args:
        filepath: Path to file
        
    Returns:
        MD5 hash as hex string
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        raise

# ----- VISUALIZATION UTILITIES -----

def set_plotting_style() -> None:
    """Set consistent style for plots"""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

def plot_feature_importance(feature_names: List[str], 
                           importance_values: List[float], 
                           title: str = "Feature Importance",
                           figsize: Tuple[int, int] = (10, 8),
                           top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    # Create DataFrame for easier sorting and plotting
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values(by='Importance', ascending=False)
    
    # Limit to top N features
    if len(feature_imp_df) > top_n:
        feature_imp_df = feature_imp_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_imp_df,
        palette='viridis',
        ax=ax
    )
    
    # Add value labels
    for i, bar in enumerate(bars.patches):
        ax.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va='center'
        )
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         normalize: bool = True,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_roc_curve(fpr: Dict[str, np.ndarray], 
                  tpr: Dict[str, np.ndarray], 
                  roc_auc: Dict[str, float],
                  title: str = "Receiver Operating Characteristic",
                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot ROC curve for multiple classifiers
    
    Args:
        fpr: Dictionary of false positive rates for each classifier
        tpr: Dictionary of true positive rates for each classifier
        roc_auc: Dictionary of ROC AUC values for each classifier
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    # Color map for multiple curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(fpr)))
    
    # Plot each ROC curve
    for i, (key, color) in enumerate(zip(fpr.keys(), colors)):
        ax.plot(
            fpr[key], 
            tpr[key], 
            color=color, 
            lw=2, 
            label=f'{key} (AUC = {roc_auc[key]:.3f})'
        )
    
    # Set axes properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc="lower right", fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_learning_curve(train_sizes: np.ndarray, 
                      train_scores: np.ndarray, 
                      test_scores: np.ndarray,
                      title: str = "Learning Curve",
                      figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot learning curve
    
    Args:
        train_sizes: Array of training sizes
        train_scores: Array of training scores for each size
        test_scores: Array of validation/test scores for each size
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for train and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    ax.grid(True, alpha=0.3)
    ax.fill_between(train_sizes, 
                   train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, 
                   alpha=0.1, color="blue")
    ax.fill_between(train_sizes, 
                   test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, 
                   alpha=0.1, color="green")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Cross-validation score")
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Training examples", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.legend(loc="best", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_residuals(y_true: np.ndarray, 
                  y_pred: np.ndarray,
                  title: str = "Residuals Plot",
                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot residuals for regression model
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of residuals
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel("Predicted Values", fontsize=14)
    ax.set_ylabel("Residuals", fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_heatmap(data: Union[pd.DataFrame, np.ndarray],
                 title: str = "Correlation Matrix",
                 mask_upper: bool = True,
                 figsize: Tuple[int, int] = (12, 10),
                 cmap: str = "viridis") -> plt.Figure:
    """
    Create correlation heatmap
    
    Args:
        data: DataFrame or correlation matrix
        title: Plot title
        mask_upper: Whether to mask upper triangle
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    # Get correlation matrix if input is DataFrame
    if isinstance(data, pd.DataFrame):
        corr = data.corr()
    else:
        corr = data
    
    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": .8}
    )
    
    # Set title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_distribution(data: pd.Series,
                    title: str = None,
                    figsize: Tuple[int, int] = (10, 6),
                    bins: int = 30,
                    kde: bool = True) -> plt.Figure:
    """
    Plot distribution of a feature
    
    Args:
        data: Series with feature values
        title: Plot title
        figsize: Figure size
        bins: Number of bins for histogram
        kde: Whether to show KDE curve
        
    Returns:
        Matplotlib figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distribution
    sns.histplot(data, bins=bins, kde=kde, ax=ax)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    else:
        ax.set_title(f"Distribution of {data.name}", fontsize=16, pad=20)
    
    # Set labels
    ax.set_xlabel(data.name, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# ----- STRING AND TEXT UTILITIES -----

def slugify(text: str) -> str:
    """
    Convert text to slug (lowercase, no special chars, spaces to hyphens)
    
    Args:
        text: Input text
        
    Returns:
        Slugified text
    """
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace spaces with hyphens
    text = re.sub(r'[-\s]+', '-', text)
    # Remove leading/trailing hyphens
    return text.strip('-')

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a random ID
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random part
        
    Returns:
        Generated ID
    """
    chars = string.ascii_lowercase + string.digits
    random_part = ''.join(random.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{random_part}"
    else:
        return random_part

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: String to append if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_number(number: Union[int, float], 
                precision: int = 2, 
                add_comma_separators: bool = True) -> str:
    """
    Format number with specific precision and comma separators
    
    Args:
        number: Number to format
        precision: Decimal precision
        add_comma_separators: Whether to add commas for thousands
        
    Returns:
        Formatted number string
    """
    if isinstance(number, int):
        if add_comma_separators:
            return f"{number:,}"
        return str(number)
    
    if add_comma_separators:
        return f"{number:,.{precision}f}"
    
    return f"{number:.{precision}f}"

def format_time_delta(seconds: float) -> str:
    """
    Format time delta into human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hrs"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

# ----- MISCELLANEOUS UTILITIES -----

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Nested dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with keys like "a.b.c"
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        
        # Build nested dict
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set value at leaf
        current[parts[-1]] = value
    
    return result

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specific size
    
    Args:
        items: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def is_notebook() -> bool:
    """
    Check if code is running in Jupyter notebook
    
    Returns:
        True if running in notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter

def configure_pandas_display() -> None:
    """Configure pandas display settings for better readability"""
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_colwidth', 100)

def get_column_types(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, List[str]]:
    """
    Categorize columns by data type
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with column types
    """
    if isinstance(df, dd.DataFrame):
        # For Dask DataFrame, compute a small sample
        sample = df.head(100)
    else:
        sample = df
    
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []
    bool_cols = []
    other_cols = []
    
    for col in sample.columns:
        # Check dtype
        dtype = sample[col].dtype
        
        # Categorize based on dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # For object dtype, check if it's text or categorical
            if pd.api.types.is_object_dtype(dtype):
                sample_values = sample[col].dropna()
                
                if len(sample_values) > 0:
                    # Check if values are strings and have many characters
                    if (sample_values.map(type) == str).all() and sample_values.str.len().mean() > 50:
                        text_cols.append(col)
                    else:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        elif pd.api.types.is_datetime64_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            bool_cols.append(col)
        else:
            other_cols.append(col)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'text': text_cols,
        'boolean': bool_cols,
        'other': other_cols
    }

def get_optimal_workers() -> int:
    """
    Get optimal number of worker processes based on CPU cores
    
    Returns:
        Suggested number of workers
    """
    # Use N-1 cores, keeping one for system processes
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = psutil.cpu_count(logical=True)
    
    if cores is None:
        return 2  # Default if detection fails
    
    return max(1, cores - 1)

def suggest_chunk_size(file_size_bytes: int, available_memory_gb: Optional[float] = None) -> int:
    """
    Suggest optimal chunk size for Dask based on file size and memory
    
    Args:
        file_size_bytes: Size of file in bytes
        available_memory_gb: Available memory in GB, auto-detected if None
        
    Returns:
        Suggested chunk size in bytes
    """
    if available_memory_gb is None:
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024 ** 3)
    
    # Use approximately 1/4 of available memory per worker process
    worker_count = get_optimal_workers()
    memory_per_worker_gb = available_memory_gb / worker_count / 4
    
    # Convert to bytes
    memory_per_worker_bytes = memory_per_worker_gb * (1024 ** 3)
    
    # Try to make at least 10 partitions for parallelism
    ideal_partitions = max(10, worker_count * 2)
    chunk_size = file_size_bytes / ideal_partitions
    
    # Cap at memory per worker
    chunk_size = min(chunk_size, memory_per_worker_bytes)
    
    # Round to nearest MB for simplicity
    chunk_size = int(chunk_size / (1024 ** 2)) * (1024 ** 2)
    
    # Ensure minimum reasonable size
    return max(chunk_size, 4 * 1024 * 1024)  # Minimum 4MB

def print_dataframe_info(df: Union[pd.DataFrame, dd.DataFrame]) -> None:
    """
    Print useful information about the dataframe
    
    Args:
        df: Input dataframe (Pandas or Dask)
    """
    is_dask = isinstance(df, dd.DataFrame)
    
    print("=" * 40)
    print("DATAFRAME INFORMATION")
    print("=" * 40)
    
    # Shape
    if is_dask:
        print(f"Shape: {df.shape[1]} columns x {df.shape[0].compute():,} rows (Dask DataFrame)")
    else:
        print(f"Shape: {df.shape[1]} columns x {df.shape[0]:,} rows (Pandas DataFrame)")
    
    # Memory usage
    if not is_dask:
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Column types
    col_types = get_column_types(df)
    print("\nColumn types:")
    for type_name, cols in col_types.items():
        if cols:
            print(f"  - {type_name} ({len(cols)}): {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
    
    # Missing values
    if is_dask:
        print("\nMissing values: Computing...")
    else:
        missing = df.isna().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print("\nMissing values:")
            for col, count in missing_cols.items():
                print(f"  - {col}: {count:,} ({count/len(df):.2%})")
        else:
            print("\nMissing values: None")
    
    print("=" * 40)

def get_categorical_cardinality(df: Union[pd.DataFrame, dd.DataFrame], 
                              categorical_cols: List[str]) -> Dict[str, int]:
    """
    Get cardinality (unique values count) for categorical columns
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary mapping column names to unique value counts
    """
    results = {}
    
    for col in categorical_cols:
        if col in df.columns:
            if isinstance(df, dd.DataFrame):
                # For Dask, this requires computation
                unique_count = df[col].nunique().compute()
            else:
                unique_count = df[col].nunique()
            
            results[col] = unique_count
    
    return results

def convert_bytes_to_human_readable(bytes_value: int) -> str:
    """
    Convert bytes to human-readable format
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Human-readable string
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    
    unit_index = 0
    value = float(bytes_value)
    
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    
    return f"{value:.2f} {units[unit_index]}"

def get_git_info() -> Dict[str, str]:
    """
    Get Git repository information if available
    
    Returns:
        Dictionary with git information
    """
    git_info = {
        "git_available": False,
        "branch": "",
        "commit": "",
        "commit_message": "",
        "commit_date": ""
    }
    
    try:
        import git
        
        try:
            repo = git.Repo(search_parent_directories=True)
            git_info["git_available"] = True
            git_info["branch"] = repo.active_branch.name
            git_info["commit"] = repo.head.commit.hexsha
            git_info["commit_message"] = repo.head.commit.message.strip()
            git_info["commit_date"] = repo.head.commit.committed_datetime.isoformat()
        except:
            pass
    except ImportError:
        pass
    
    return git_info