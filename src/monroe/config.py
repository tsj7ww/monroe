# config.py

import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

class Config:
    DEFAULT_CONFIG = {
        # Data processing
        "data": {
            "chunk_size": 20000,  # Chunk size for Dask
            "n_workers": -1,        # Use all available cores
            "memory_limit": "4GB",  # Memory limit per worker
            # "file_types_supported": [".csv", ".parquet"],#, ".json", ".xlsx"],
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
                {"type": "linear"},
                {"type": "logistic"},
                {"type": "ridge"},
                {"type": "lasso"},
                {"type": "elasticnet"},
                # GBMs
                {"type": "xgboost"},
                {"type": "lightgbm"},
                {"type": "catboost"},
                # Other classical ML
                {"type": "random_forest"},
                {"type": "svm"},
                {"type": "knn"},
                # Deep Learning
                {"type": "neural_network",
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
        # "experiment_tracking": {
            # "enabled": True,
            # "backend": "local",  # local, mlflow, wandb
            # "auto_log": True,
            # "save_artifacts": True,
            # "log_system_metrics": True
        # },
        
        # System
        "system": {
            "log_level": "INFO",
            "output_dir": "./output",
            "temp_dir": "./tmp",
            "clean_tmp_after_run": True,
            # "use_gpu": "auto",  # auto, True, False
            # "precision": "float32"  # float16, float32, float64
        }
    }
    
    def __init__(self, data_fpath, target_column, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        self.data_fpath = data_fpath
        self.target_column = target_column
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
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _setup_directories(self) -> None:
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
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
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
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)
    
    def as_dict(self) -> Dict[str, Any]:
        return self.config.copy()