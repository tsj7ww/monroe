# neural_network_model.py

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import optuna
from optuna.trial import Trial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
from dask.distributed import Client, wait
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuralNetworkModel")

class PyTorchDataset(Dataset):
    """Custom dataset for PyTorch training"""
    
    def __init__(self, X, y):
        """
        Initialize dataset with features and targets
        
        Args:
            X: Feature array (numpy or tensor)
            y: Target array (numpy or tensor)
        """
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X
            
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y
            
        # Reshape y if needed
        if len(self.y.shape) == 1:
            # For binary/multiclass classification or single-target regression
            self.y = self.y.view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    """Flexible PyTorch neural network with configurable architecture"""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        """
        Initialize neural network
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
            config: Neural network configuration
        """
        super(NeuralNetwork, self).__init__()
        
        # Extract configuration parameters
        hidden_dims = config.get("hidden_dims", [64, 32])
        dropout_rate = config.get("dropout_rate", 0.2)
        activation = config.get("activation", "relu")
        batch_norm = config.get("batch_norm", True)
        
        # Set up activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add batch normalization if enabled
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation function
            layers.append(self.activation)
            
            # Add dropout if enabled
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

class NeuralNetworkModel:
    """Neural network wrapper for AutoML with Optuna and Dask support"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize neural network model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  config.get("use_gpu", True) else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.output_dim = None
        self.best_params = None
        self.is_classification = None
        self.classes = None
        
        logger.info(f"Neural network will use device: {self.device}")
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for neural network training
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            Tuple of processed X and y arrays
        """
        # Scale features
        if self.config.get("scale_features", True):
            if y is not None:  # Training mode
                X = self.scaler.fit_transform(X)
            else:  # Inference mode
                X = self.scaler.transform(X)
        
        # Process target for classification
        if y is not None and self.is_classification:
            if self.classes is None:
                self.classes = np.unique(y)
            
            # One-hot encode multiclass targets
            if len(self.classes) > 2 and self.output_dim > 1:
                y_encoded = np.zeros((len(y), len(self.classes)))
                for i, cls in enumerate(self.classes):
                    y_encoded[:, i] = (y == cls).astype(int)
                return X, y_encoded
        
        return (X, y) if y is not None else (X, None)
    
    def load_data_for_training(self, X: np.ndarray, y: np.ndarray
                             ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of train, validation, and test data loaders
        """
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=self.config.get("test_val_split", 0.3),
            random_state=self.config.get("random_state", 42)
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # 50% of temp goes to test
            random_state=self.config.get("random_state", 42)
        )
        
        # Preprocess data
        X_train, y_train = self.preprocess_data(X_train, y_train)
        X_val, y_val = self.preprocess_data(X_val, y_val)
        X_test, y_test = self.preprocess_data(X_test, y_test)
        
        # Create PyTorch datasets
        train_dataset = PyTorchDataset(X_train, y_train)
        val_dataset = PyTorchDataset(X_val, y_val)
        test_dataset = PyTorchDataset(X_test, y_test)
        
        # Create data loaders
        batch_size = self.config.get("batch_size", 32)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.config.get("dataloader_workers", 0)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get("dataloader_workers", 0)
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get("dataloader_workers", 0)
        )
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, params: Dict[str, Any]) -> nn.Module:
        """
        Train neural network model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            params: Model hyperparameters
            
        Returns:
            Trained PyTorch model
        """
        # Create model
        model = NeuralNetwork(self.input_dim, self.output_dim, params).to(self.device)
        
        # Define loss function
        if self.is_classification:
            if self.output_dim == 1:  # Binary classification
                criterion = nn.BCEWithLogitsLoss()
            else:  # Multiclass classification
                criterion = nn.CrossEntropyLoss()
        else:  # Regression
            criterion = nn.MSELoss()
        
        # Define optimizer
        lr = params.get("learning_rate", 0.001)
        weight_decay = params.get("weight_decay", 0.0001)
        
        if params.get("optimizer", "adam") == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif params.get("optimizer", "adam") == "sgd":
            momentum = params.get("momentum", 0.9)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if params.get("use_lr_scheduler", True):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Train model
        num_epochs = params.get("epochs", 100)
        patience = params.get("early_stopping_patience", 10)
        best_val_loss = float('inf')
        no_improvement = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Handle output shape for binary classification
                if self.is_classification and self.output_dim == 1:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Handle output shape for binary classification
                    if self.is_classification and self.output_dim == 1:
                        loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update learning rate scheduler
            if params.get("use_lr_scheduler", True):
                scheduler.step(val_loss)
            
            # Print progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                best_model_state = model.state_dict().copy()
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: PyTorch model
            data_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Convert to numpy
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        
        # Concatenate batches
        y_true = np.vstack(all_targets)
        y_pred_raw = np.vstack(all_outputs)
        
        # Process predictions based on problem type
        if self.is_classification:
            if self.output_dim == 1:  # Binary classification
                y_pred_proba = 1 / (1 + np.exp(-y_pred_raw))  # Sigmoid
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:  # Multiclass classification
                y_pred_proba = np.exp(y_pred_raw) / np.sum(np.exp(y_pred_raw), axis=1, keepdims=True)  # Softmax
                y_pred = np.argmax(y_pred_raw, axis=1)
                
                # Convert one-hot back to labels
                if y_true.shape[1] > 1:
                    y_true = np.argmax(y_true, axis=1)
        else:  # Regression
            y_pred = y_pred_raw
            y_pred_proba = None
        
        # Calculate metrics
        metrics = {}
        
        if self.is_classification:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            
            if self.output_dim == 1:  # Binary classification
                metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
                metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
                
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                except:
                    metrics["roc_auc"] = 0.0
            else:  # Multiclass classification
                metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
                metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
                metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
                metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        else:  # Regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
        
        return metrics
    
    def objective(self, trial: Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial
            X: Feature array
            y: Target array
            
        Returns:
            Validation metric to optimize
        """
        # Define hyperparameters to optimize
        params = {
            "hidden_dims": [
                trial.suggest_int(f"hidden_dim_1", 16, 256),
                trial.suggest_int(f"hidden_dim_2", 16, 128)
            ],
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "epochs": self.config.get("epochs", 100),
            "early_stopping_patience": self.config.get("early_stopping_patience", 10),
            "use_lr_scheduler": True
        }
        
        # Add momentum for SGD
        if params["optimizer"] == "sgd":
            params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
        
        # Prepare data loaders
        train_loader, val_loader, _ = self.load_data_for_training(X, y)
        
        # Train model
        model = self.train_model(train_loader, val_loader, params)
        
        # Evaluate model
        metrics = self.evaluate_model(model, val_loader)
        
        # Return appropriate metric based on problem type
        if self.is_classification:
            if self.output_dim == 1:  # Binary classification
                return 1.0 - metrics.get("roc_auc", 0)  # Minimize 1-AUC
            else:  # Multiclass classification
                return 1.0 - metrics.get("f1_macro", 0)  # Minimize 1-F1
        else:  # Regression
            return metrics.get("rmse", float('inf'))  # Minimize RMSE
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Dictionary of best parameters
        """
        # Set up Optuna study
        study_name = f"neural_network_{int(time.time())}"
        
        if self.is_classification:
            direction = "minimize"  # Minimizing 1-metric
        else:  # Regression
            direction = "minimize"  # Minimizing RMSE
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.get("random_state", 42))
        )
        
        # Run optimization
        n_trials = self.config.get("n_trials", 20)
        timeout = self.config.get("timeout", 3600)  # 1 hour
        
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout
        )
        
        # Get best parameters
        self.best_params = study.best_params
        
        # Add fixed parameters
        self.best_params["epochs"] = self.config.get("epochs", 100)
        self.best_params["early_stopping_patience"] = self.config.get("early_stopping_patience", 10)
        
        # Add hidden_dims as list
        hidden_dims = []
        for i in range(1, 10):  # Check for up to 10 hidden layers
            key = f"hidden_dim_{i}"
            if key in self.best_params:
                hidden_dims.append(self.best_params[key])
                del self.best_params[key]
        
        self.best_params["hidden_dims"] = hidden_dims
        
        # Log best parameters
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best value: {study.best_value}")
        
        return self.best_params
    
    def parallel_hyperparameter_search(self, X: np.ndarray, y: np.ndarray, n_workers: int = 4) -> Dict[str, Any]:
        """
        Run hyperparameter search in parallel using Dask
        
        Args:
            X: Feature array
            y: Target array
            n_workers: Number of Dask workers
            
        Returns:
            Dictionary of best parameters
        """
        # Set up Dask client
        client = Client()
        logger.info(f"Dask dashboard at: {client.dashboard_link}")
        
        try:
            # Create Dask arrays
            X_parts = []
            y_parts = []
            
            # Split data into parts for parallel trials
            n_parts = min(n_workers, 4)  # Don't create too many partitions
            part_size = len(X) // n_parts
            
            for i in range(n_parts):
                start = i * part_size
                end = start + part_size if i < n_parts - 1 else len(X)
                X_parts.append(X[start:end])
                y_parts.append(y[start:end])
            
            # Set up Optuna study
            study_name = f"neural_network_parallel_{int(time.time())}"
            
            if self.is_classification:
                direction = "minimize"  # Minimizing 1-metric
            else:  # Regression
                direction = "minimize"  # Minimizing RMSE
            
            # Create study with storage for sharing between workers
            storage_path = f"sqlite:///{study_name}.db"
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.config.get("random_state", 42)),
                storage=storage_path
            )
            
            # Run optimization in parallel
            n_trials = self.config.get("n_trials", 20)
            trials_per_worker = n_trials // n_workers + 1
            
            # Function for each worker
            def worker_objective(worker_id, X_part, y_part):
                # Set up local config
                local_config = self.config.copy()
                local_input_dim = self.input_dim
                local_output_dim = self.output_dim
                local_is_classification = self.is_classification
                
                # Create local study
                local_study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_path
                )
                
                # Define local objective function
                def local_objective(trial):
                    # Define hyperparameters to optimize
                    params = {
                        "hidden_dims": [
                            trial.suggest_int(f"hidden_dim_1", 16, 256),
                            trial.suggest_int(f"hidden_dim_2", 16, 128)
                        ],
                        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                        "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"]),
                        "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
                        "epochs": local_config.get("epochs", 100),
                        "early_stopping_patience": local_config.get("early_stopping_patience", 10),
                        "use_lr_scheduler": True
                    }
                    
                    # Add momentum for SGD
                    if params["optimizer"] == "sgd":
                        params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
                    
                    # Create local model
                    local_model = NeuralNetworkModel(local_config)
                    local_model.input_dim = local_input_dim
                    local_model.output_dim = local_output_dim
                    local_model.is_classification = local_is_classification
                    
                    # Prepare data loaders
                    train_loader, val_loader, _ = local_model.load_data_for_training(X_part, y_part)
                    
                    # Train model
                    nn_model = local_model.train_model(train_loader, val_loader, params)
                    
                    # Evaluate model
                    metrics = local_model.evaluate_model(nn_model, val_loader)
                    
                    # Return appropriate metric based on problem type
                    if local_is_classification:
                        if local_output_dim == 1:  # Binary classification
                            return 1.0 - metrics.get("roc_auc", 0)
                        else:  # Multiclass classification
                            return 1.0 - metrics.get("f1_macro", 0)
                    else:  # Regression
                        return metrics.get("rmse", float('inf'))
                
                # Run local optimization
                local_study.optimize(local_objective, n_trials=trials_per_worker)
                return worker_id
            
            # Submit tasks to Dask workers
            futures = []
            for i in range(n_workers):
                future = client.submit(
                    worker_objective, 
                    i, 
                    X_parts[i % len(X_parts)], 
                    y_parts[i % len(y_parts)]
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            wait(futures)
            
            # Get best parameters
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_path
            )
            
            self.best_params = study.best_params
            
            # Add fixed parameters
            self.best_params["epochs"] = self.config.get("epochs", 100)
            self.best_params["early_stopping_patience"] = self.config.get("early_stopping_patience", 10)
            
            # Add hidden_dims as list
            hidden_dims = []
            for i in range(1, 10):  # Check for up to 10 hidden layers
                key = f"hidden_dim_{i}"
                if key in self.best_params:
                    hidden_dims.append(self.best_params[key])
                    del self.best_params[key]
            
            self.best_params["hidden_dims"] = hidden_dims
            
            # Log best parameters
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best value: {study.best_value}")
            
            # Clean up
            try:
                os.remove(f"{study_name}.db")
            except:
                pass
            
            return self.best_params
        
        finally:
            # Close Dask client
            client.close()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkModel':
        """
        Fit neural network model
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Fitted model
        """
        logger.info("Fitting neural network model")
        
        # Set dimensions
        self.input_dim = X.shape[1]
        
        # Determine problem type
        unique_values = np.unique(y)
        if len(unique_values) < 10 and np.array_equal(unique_values, unique_values.astype(int)):
            # Classification problem
            self.is_classification = True
            self.classes = unique_values
            
            if len(self.classes) == 2:
                # Binary classification
                self.output_dim = 1
            else:
                # Multiclass classification
                self.output_dim = len(self.classes)
        else:
            # Regression problem
            self.is_classification = False
            self.output_dim = 1 if len(y.shape) == 1 else y.shape[1]
        
        logger.info(f"Problem type: {'Classification' if self.is_classification else 'Regression'}")
        logger.info(f"Input dimension: {self.input_dim}, Output dimension: {self.output_dim}")
        
        # Optimize hyperparameters if enabled
        if self.config.get("optimize_hyperparameters", True):
            if self.config.get("use_dask", True):
                logger.info("Running parallel hyperparameter optimization with Dask")
                self.best_params = self.parallel_hyperparameter_search(
                    X, y, n_workers=self.config.get("n_workers", 4)
                )
            else:
                logger.info("Running sequential hyperparameter optimization")
                self.best_params = self.optimize_hyperparameters(X, y)
        else:
            # Use default parameters
            self.best_params = {
                "hidden_dims": self.config.get("hidden_dims", [64, 32]),
                "dropout_rate": self.config.get("dropout_rate", 0.2),
                "learning_rate": self.config.get("learning_rate", 0.001),
                "weight_decay": self.config.get("weight_decay", 0.0001),
                "batch_size": self.config.get("batch_size", 32),
                "optimizer": self.config.get("optimizer", "adam"),
                "activation": self.config.get("activation", "relu"),
                "batch_norm": self.config.get("batch_norm", True),
                "epochs": self.config.get("epochs", 100),
                "early_stopping_patience": self.config.get("early_stopping_patience", 10),
                "use_lr_scheduler": True
            }
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self.load_data_for_training(X, y)
        
        # Train model with best parameters
        logger.info("Training final model with best parameters")
        self.model = self.train_model(train_loader, val_loader, self.best_params)
        
        # Evaluate final model
        metrics = self.evaluate_model(self.model, test_loader)
        logger.info(f"Final model metrics: {metrics}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels or values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Preprocess features
        X, _ = self.preprocess_data(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
        # Convert to numpy and process based on problem type
        outputs_np = outputs.cpu().numpy()
        
        if self.is_classification:
            if self.output_dim == 1:  # Binary classification
                y_pred = (outputs_np > 0).astype(int)
            else:  # Multiclass classification
                y_pred = np.argmax(outputs_np, axis=1)
                
                # Map back to original class labels if needed
                if self.classes is not None:
                    y_pred = np.array([self.classes[i] for i in y_pred])
        else:  # Regression
            y_pred = outputs_np
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for classification
        
        Args:
            X: Feature array
            
        Returns:
            Predicted probabilities
        """
        if not self.is_classification:
            raise ValueError("predict_proba() is only available for classification problems")
        
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Preprocess features
        X, _ = self.preprocess_data(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
        # Convert to probabilities
        outputs_np = outputs.cpu().numpy()
        
        if self.output_dim == 1:  # Binary classification
            proba = 1 / (1 + np.exp(-outputs_np))  # Sigmoid
            return np.hstack([1 - proba, proba])  # Return probabilities for both classes
        else:  # Multiclass classification
            # Apply softmax
            exp_outputs = np.exp(outputs_np)
            proba = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            return proba
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare data to save
        save_data = {
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "is_classification": self.is_classification,
            "classes": self.classes,
            "scaler": self.scaler,
            "best_params": self.best_params,
            "config": self.config
        }
        
        # Save model
        torch.save(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetworkModel':
        """
        Load model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load data
        save_data = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create new instance
        model_instance = cls(save_data["config"])
        
        # Restore attributes
        model_instance.input_dim = save_data["input_dim"]
        model_instance.output_dim = save_data["output_dim"]
        model_instance.is_classification = save_data["is_classification"]
        model_instance.classes = save_data["classes"]
        model_instance.scaler = save_data["scaler"]
        model_instance.best_params = save_data["best_params"]
        
        # Create model
        model_instance.model = NeuralNetwork(
            model_instance.input_dim,
            model_instance.output_dim,
            model_instance.best_params
        ).to(model_instance.device)
        
        # Load state
        model_instance.model.load_state_dict(save_data["model_state"])
        model_instance.model.eval()
        
        logger.info(f"Model loaded from {filepath}")
        return model_instance
    
    def get_feature_importance(self, X: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Calculate feature importance using permutation importance
        
        Args:
            X: Feature array
            n_samples: Number of samples to use
            
        Returns:
            Array of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Sample data if needed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Preprocess features
        X_processed, _ = self.preprocess_data(X_sample)
        
        # Get baseline predictions
        baseline_preds = self.predict(X_sample)
        
        # Calculate importance for each feature
        importances = np.zeros(self.input_dim)
        
        for i in range(self.input_dim):
            # Copy data and permute one feature
            X_permuted = X_processed.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get predictions with permuted feature
            permuted_preds = self.predict(X_permuted)
            
            # Calculate importance as decrease in performance
            if self.is_classification:
                # Use accuracy for classification
                importance = np.mean(baseline_preds == permuted_preds)
            else:
                # Use MSE for regression
                importance = np.mean((baseline_preds - permuted_preds) ** 2)
            
            importances[i] = importance
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances

# Example usage with Dask for data processing
def train_neural_network_with_dask(file_path: str, target_column: str, config: Dict[str, Any]) -> NeuralNetworkModel:
    """
    Train neural network model using Dask for data processing
    
    Args:
        file_path: Path to dataset file
        target_column: Name of target column
        config: Model configuration
        
    Returns:
        Trained NeuralNetworkModel
    """
    # Start Dask client
    if config.get("use_dask", True):
        client = Client()
        logger.info(f"Dask dashboard at: {client.dashboard_link}")
    
    try:
        # Load data with Dask
        logger.info(f"Loading data from {file_path}")
        
        if file_path.endswith('.csv'):
            ddf = dd.read_csv(file_path, assume_missing=True)
        elif file_path.endswith('.parquet'):
            ddf = dd.read_parquet(file_path, assume_missing=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Check if target column exists
        if target_column not in ddf.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Split features and target
        X_dask = ddf.drop(target_column, axis=1)
        y_dask = ddf[target_column]
        
        # Convert to numpy arrays
        logger.info("Converting Dask DataFrame to numpy arrays")
        X = X_dask.compute().values
        y = y_dask.compute().values
        
        # Create and train neural network model
        logger.info("Creating neural network model")
        model = NeuralNetworkModel(config)
        model.fit(X, y)
        
        return model
    
    finally:
        # Close Dask client
        if config.get("use_dask", True):
            client.close()

# Example of a complete pipeline for neural network training
def run_neural_network_pipeline(file_path: str, 
                              target_column: str = None,
                              config_path: str = None,
                              output_dir: str = "./models") -> Dict[str, Any]:
    """
    Run complete neural network training pipeline
    
    Args:
        file_path: Path to dataset file
        target_column: Name of target column (auto-detected if None)
        config_path: Path to configuration file (uses defaults if None)
        output_dir: Directory to save model and results
        
    Returns:
        Dictionary with results
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                import json
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
    else:
        # Default configuration
        config = {
            "use_dask": True,
            "n_workers": 4,
            "optimize_hyperparameters": True,
            "n_trials": 20,
            "timeout": 3600,
            "epochs": 100,
            "early_stopping_patience": 10,
            "batch_size": 32,
            "use_gpu": True,
            "scale_features": True,
            "random_state": 42
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    try:
        # Load data sample to auto-detect target if needed
        logger.info(f"Loading data sample from {file_path}")
        
        if file_path.endswith('.csv'):
            df_sample = pd.read_csv(file_path, nrows=1000)
        elif file_path.endswith('.parquet'):
            df_sample = pd.read_parquet(file_path, nrows=1000)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Auto-detect target column if not provided
        if target_column is None:
            # Try to find a suitable target column
            # This is a simplified approach - in a real system, we'd use more sophisticated detection
            num_cols = df_sample.select_dtypes(include=['number']).columns
            cat_cols = df_sample.select_dtypes(include=['category', 'object', 'bool']).columns
            
            potential_targets = []
            
            # Check categorical columns first (for classification)
            for col in cat_cols:
                unique_vals = df_sample[col].nunique()
                if 2 <= unique_vals <= 100:  # Reasonable number of classes
                    potential_targets.append((col, unique_vals))
            
            # Then check numeric columns (for regression)
            if not potential_targets:
                for col in num_cols:
                    potential_targets.append((col, df_sample[col].nunique()))
            
            if potential_targets:
                # Sort by name to ensure deterministic behavior
                potential_targets.sort(key=lambda x: x[0])
                target_column = potential_targets[0][0]
                logger.info(f"Auto-detected target column: {target_column}")
            else:
                raise ValueError("Could not auto-detect target column")
        
        # Train model
        logger.info(f"Training neural network model with target: {target_column}")
        model = train_neural_network_with_dask(file_path, target_column, config)
        
        # Save model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"neural_network_model_{timestamp}.pt")
        model.save(model_path)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            "model_path": model_path,
            "target_column": target_column,
            "is_classification": model.is_classification,
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hyperparameters": model.best_params,
            "training_time": training_time
        }
        
        # Save results
        results_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Pipeline completed successfully in {training_time:.2f} seconds")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise