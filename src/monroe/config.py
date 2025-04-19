# config.py

import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

class AutoMLConfig:
    """
    Configuration manager for AutoML pipeline.
    
    This class handles all configuration settings including:
    - Data processing configurations
    - Model training parameters
    - Hyperparameter optimization settings
    - Evaluation metrics
    - Visualization preferences
    """
    
    DEFAULT_CONFIG = {
        # Data processing
        "data": {
            "chunk_size": 1000000,  # Chunk size for Dask
            "n_workers": -1,        # Use all available cores
            "memory_limit": "4GB",  # Memory limit per worker
            "file_types_supported": [".csv", ".parquet", ".json", ".xlsx"],
            "missing_values": {
                "strategy": "auto",  # auto, impute, drop
                "numeric_strategy": "mean",  # mean, median, constant
                "categorical_strategy": "most_frequent",  # most_frequent, constant
                "constant_value": 0
            },
            "feature_engineering": {
                "auto_encoding": True,  # One-hot encoding for categoricals
                "scaling": "standard",  # standard, minmax, robust, none
                "outlier_detection": True,
                "feature_selection": {
                    "enabled": True,
                    "method": "variance",  # variance, correlation, rfe, tree_based
                    "threshold": 0.95
                }
            }
        },
        
        # Model training
        "models": {
            "train_test_split": 0.8,
            "validation_size": 0.2,
            "cv_folds": 5,
            "stratify": True,
            "random_state": 42,
            "candidates": [
                # GLMs
                {"type": "linear", "enabled": True}, 
                {"type": "logistic", "enabled": True},
                {"type": "ridge", "enabled": True},
                {"type": "lasso", "enabled": True},
                {"type": "elasticnet", "enabled": True},
                # GBMs
                {"type": "xgboost", "enabled": True},
                {"type": "lightgbm", "enabled": True},
                {"type": "catboost", "enabled": True},
                # Other classical ML
                {"type": "random_forest", "enabled": True},
                {"type": "svm", "enabled": False},
                {"type": "knn", "enabled": False},
                # Deep Learning
                {"type": "neural_network", "enabled": True,
                 "architecture": "auto",  # auto, mlp, cnn, rnn, transformer
                 "framework": "pytorch"  # pytorch, tensorflow
                }
            ]
        },
        
        # Hyperparameter optimization
        "hyperparameter_optimization": {
            "enabled": True,
            "engine": "optuna",  # optuna, ray
            "n_trials": 100,
            "timeout": 3600,  # seconds
            "cv": 3,
            "metric": "auto",  # auto selects based on problem type
            "n_jobs": -1,
            "early_stopping": True
        },
        
        # Evaluation
        "evaluation": {
            "metrics": {
                "classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "regression": ["rmse", "mae", "r2", "mape"]
            },
            "visualizations": {
                "learning_curve": True,
                "feature_importance": True,
                "confusion_matrix": True,
                "roc_curve": True,
                "precision_recall_curve": True,
                "residual_plot": True,
                "prediction_error": True,
                "shap_values": True
            }
        },
        
        # Experiment tracking
        "experiment_tracking": {
            "enabled": True,
            "backend": "local",  # local, mlflow, wandb
            "auto_log": True,
            "save_artifacts": True,
            "log_system_metrics": True
        },
        
        # System
        "system": {
            "log_level": "INFO",
            "output_dir": "./automl_output",
            "temp_dir": "./tmp",
            "clean_tmp_after_run": True,
            "use_gpu": "auto",  # auto, True, False
            "precision": "float32"  # float16, float32, float64
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration with default values
        and optionally override with user-provided config file
        
        Args:
            config_path: Path to custom configuration file (yaml or json)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AutoMLConfig")
        
        # Load custom config if provided
        if config_path:
            self._load_config(config_path)
            
        # Create output directories
        self._setup_directories()
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file and merge with defaults
        
        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return
            
        try:
            file_ext = os.path.splitext(config_path)[1].lower()
            
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            else:
                self.logger.warning(f"Unsupported config file format: {file_ext}. Using default configuration.")
                return
                
            # Recursively update config with user values
            self._update_nested_dict(self.config, user_config)
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            self.logger.warning("Using default configuration")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Recursively update a nested dictionary with another nested dictionary
        
        Args:
            d: Target dictionary to update
            u: Source dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _setup_directories(self) -> None:
        """Create necessary directories for outputs and temporary files"""
        output_dir = self.config["system"]["output_dir"]
        temp_dir = self.config["system"]["temp_dir"]
        
        # Create timestamp-based run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Update config with run directory
        self.config["system"]["run_dir"] = self.run_dir
        self.logger.info(f"Created run directory: {self.run_dir}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation path
        
        Args:
            key_path: Dot notation path to config value (e.g., "data.chunk_size")
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation path
        
        Args:
            key_path: Dot notation path to config value (e.g., "data.chunk_size")
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        self.logger.debug(f"Set config {key_path} = {value}")
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save current configuration to a file
        
        Args:
            filepath: Path to save the configuration file
            
        Returns:
            Path to the saved configuration file
        """
        if filepath is None:
            filepath = os.path.join(self.run_dir, "config.yaml")
            
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                
            self.logger.info(f"Configuration saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return ""
    
    def validate(self) -> bool:
        """
        Validate the configuration for consistency and required values
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Implement validation logic here
        # This is a basic example - expand based on specific requirements
        try:
            # Check if supported file types are valid
            file_types = self.get("data.file_types_supported")
            if not isinstance(file_types, list) or len(file_types) == 0:
                self.logger.error("No supported file types specified")
                return False
            
            # Check if at least one model is enabled
            models = self.get("models.candidates", [])
            if not any(model.get("enabled", False) for model in models):
                self.logger.error("No models are enabled for training")
                return False
                
            # Add more validation as needed
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)
    
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def infer_problem_type(self, target_column_info: Dict[str, Any]) -> str:
        """
        Infer the problem type (classification or regression) based on target column
        
        Args:
            target_column_info: Dictionary with information about target column
            
        Returns:
            Problem type: "classification" or "regression"
        """
        dtype = target_column_info.get("dtype", "")
        unique_values = target_column_info.get("unique_values", 0)
        
        # If target has few unique values or is object/category/bool type, 
        # likely a classification problem
        if (dtype in ["object", "category", "bool"] or 
            (unique_values > 1 and unique_values <= 10)):
            return "classification"
        else:
            return "regression"
    
    def auto_configure(self, dataset_info: Dict[str, Any]) -> None:
        """
        Auto-configure based on dataset properties
        
        Args:
            dataset_info: Dictionary with dataset metadata
        """
        # Example: Adjust chunk size based on dataset size
        dataset_size = dataset_info.get("size_bytes", 0)
        if dataset_size > 1e9:  # > 1GB
            self.set("data.chunk_size", 2000000)
        
        # Example: Adjust number of workers based on dataset complexity
        n_columns = dataset_info.get("n_columns", 0)
        if n_columns > 100:
            self.set("data.n_workers", min(os.cpu_count() or 4, 16))
        
        # Example: Disable complex models for very large datasets
        if dataset_size > 5e9:  # > 5GB
            for i, model in enumerate(self.get("models.candidates", [])):
                if model["type"] in ["neural_network", "svm"]:
                    self.set(f"models.candidates.{i}.enabled", False)
        
        # Example: Adjust hyperparameter optimization based on dataset size
        if dataset_size > 2e9:  # > 2GB
            self.set("hyperparameter_optimization.n_trials", 50)
            self.set("hyperparameter_optimization.timeout", 7200)
        
        # Example: Setting problem type specific metrics
        problem_type = dataset_info.get("problem_type", "")
        if problem_type == "classification":
            if dataset_info.get("is_multiclass", False):
                self.set("evaluation.primary_metric", "f1_macro")
            else:
                self.set("evaluation.primary_metric", "roc_auc")
        else:
            self.set("evaluation.primary_metric", "rmse")
            
        self.logger.info("Auto-configured settings based on dataset properties")


# Factory function to create a configuration instance
def create_config(config_path: Optional[str] = None) -> AutoMLConfig:
    """
    Create and return a configuration instance
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized AutoMLConfig instance
    """
    config = AutoMLConfig(config_path)
    return config