"""
PyTorch Model Wrapper for ML Pipeline.
Provides a scikit-learn compatible interface for training
and evaluating PyTorch Neural Networks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MLPModule(nn.Module):
    """Multi-Layer Perceptron PyTorch Module."""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type
        
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            # Optional: Add Dropout or BatchNorm here if needed
            in_size = h_size
            
        layers.append(nn.Linear(in_size, output_size))
        # Removed Sigmoid here. We will use BCEWithLogitsLoss for numerical stability
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class PyTorchMLPWrapper:
    """
    Scikit-learn compatible wrapper for PyTorch MLP models.
    Supports both classification and regression tasks.
    """
    def __init__(self, 
                 task_type: str = "regression",
                 hidden_sizes: List[int] = [64, 32],
                 lr: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32,
                 device: str = "auto",
                 random_state: int = 42):
        self.task_type = task_type
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Set seeds for reproducibility
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.classes_ = None
        self._is_fitted = False
        
        # Target scaling parameters
        self._target_mean = 0.0
        self._target_std = 1.0
        
    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Convert input data to PyTorch Tensors."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if y is None:
            return X_tensor
            
        if isinstance(y, pd.Series):
            y = y.values
            
        if self.task_type == "regression":
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else: # classification
            if len(np.unique(y)) > 2:
                self.task_type = "multiclass"
                y_tensor = torch.tensor(y, dtype=torch.long)
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
                
        return X_tensor, y_tensor
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Train the PyTorch model."""
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        input_size = X_tensor.shape[1]
        
        if self.task_type == "regression":
            # Scale targets for regression to prevent exploding gradients
            self._target_mean = y_tensor.mean().item()
            # If target values are identical, std will be 0, causing NaNs.
            self._target_std = y_tensor.std().item()
            if self._target_std == 0:
                self._target_std = 1.0
            
            y_tensor = (y_tensor - self._target_mean) / self._target_std
            
            output_size = 1
            criterion = nn.MSELoss()
        elif self.task_type == "classification":
            self.classes_ = np.unique(y)
            output_size = 1
            criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            self.classes_ = np.unique(y)
            output_size = len(self.classes_)
            criterion = nn.CrossEntropyLoss()
            
        self.model = MLPModule(input_size, self.hidden_sizes, output_size, self.task_type).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}")
                
        self._is_fitted = True
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        X_tensor = self._prepare_data(X)
        X_tensor = X_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task_type == "regression":
                preds = outputs.cpu().numpy().flatten()
                return (preds * self._target_std) + self._target_mean
            elif self.task_type == "classification":
                # Apply sigmoid since model outputs logits
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                return preds.cpu().numpy().flatten()
            elif self.task_type == "multiclass":
                _, preds = torch.max(outputs, 1)
                return preds.cpu().numpy()
                
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Output prediction probabilities (for classification only)."""
        if self.task_type == "regression":
            raise AttributeError("predict_proba is not available for regression tasks.")
            
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        X_tensor = self._prepare_data(X)
        X_tensor = X_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task_type == "classification":
                # Apply sigmoid since model outputs logits
                probas = torch.sigmoid(outputs).cpu().numpy()
                # Scikit-learn expects [n_samples, n_classes]
                return np.hstack((1 - probas, probas))
            elif self.task_type == "multiclass":
                probas = torch.softmax(outputs, dim=1).cpu().numpy()
                return probas

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "task_type": self.task_type,
            "hidden_sizes": self.hidden_sizes,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state
        }

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
