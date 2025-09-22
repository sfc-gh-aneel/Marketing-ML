"""
PyTorch Models for Financial Services ML Pipeline
Deep learning models optimized for Snowflake integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import logging
import yaml
from typing import Dict, List, Tuple, Any
import joblib

logger = logging.getLogger(__name__)

class FinancialDataset(Dataset):
    """Custom Dataset for financial services data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DeepConversionPredictor(nn.Module):
    """Deep Neural Network for Conversion Prediction"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3, num_classes: int = 2):
        super(DeepConversionPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ChurnPredictor(nn.Module):
    """Neural Network for Churn Prediction with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3):
        super(ChurnPredictor, self).__init__()
        
        self.input_dim = input_dim
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            self.feature_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.Tanh(),
            nn.Linear(prev_dim // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        # Feature extraction
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(features), dim=1)
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        return output

class NextBestActionNetwork(nn.Module):
    """Multi-class Neural Network for Next Best Action Prediction"""
    
    def __init__(self, input_dim: int, num_actions: int, hidden_layers: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3):
        super(NextBestActionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Embedding layer for categorical features (if needed)
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers
        layers = []
        prev_dim = hidden_layers[0]
        
        for hidden_dim in hidden_layers[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for action probabilities
        self.action_head = nn.Linear(prev_dim, num_actions)
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Feature embedding
        embedded = self.feature_embedding(x)
        
        # Hidden layer processing
        hidden = self.hidden_layers(embedded)
        
        # Action prediction
        action_logits = self.action_head(hidden)
        action_probs = F.softmax(action_logits, dim=1)
        
        # Confidence estimation
        confidence = self.confidence_head(hidden)
        
        return action_probs, confidence

class FinancialMLTrainer:
    """Trainer for Financial Services ML Models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['pytorch']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize scalers and encoders
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if not col.startswith(('CLIENT_ID', 'CONVERSION_TARGET', 'CHURN_TARGET', 'NEXT_BEST_ACTION'))]
        
        # Prepare features
        X = df[feature_cols].fillna(0).values
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Prepare targets
        targets = {}
        
        # Conversion target
        if 'CONVERSION_TARGET' in df.columns:
            targets['conversion'] = df['CONVERSION_TARGET'].values
        
        # Churn target
        if 'CHURN_TARGET' in df.columns:
            targets['churn'] = df['CHURN_TARGET'].values
        
        # Next best action target
        if 'NEXT_BEST_ACTION' in df.columns:
            le_action = LabelEncoder()
            targets['next_action'] = le_action.fit_transform(df['NEXT_BEST_ACTION'])
            self.label_encoders['next_action'] = le_action
        
        logger.info(f"Prepared features: {X_scaled.shape}, Targets: {list(targets.keys())}")
        return X_scaled, targets
    
    def train_conversion_model(self, X: np.ndarray, y: np.ndarray) -> DeepConversionPredictor:
        """Train conversion prediction model"""
        logger.info("Training conversion prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.model_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        # Initialize model
        model = DeepConversionPredictor(
            input_dim=X.shape[1],
            hidden_layers=self.model_config['hidden_layers'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        for epoch in range(self.model_config['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            val_loss = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            val_acc = accuracy_score(val_targets, val_predictions)
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/conversion_model.pth')
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info(f"Conversion model training completed. Best validation accuracy: {best_val_acc:.4f}")
        return model
    
    def train_churn_model(self, X: np.ndarray, y: np.ndarray) -> ChurnPredictor:
        """Train churn prediction model"""
        logger.info("Training churn prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.model_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        # Initialize model
        model = ChurnPredictor(
            input_dim=X.shape[1],
            hidden_layers=self.model_config['hidden_layers'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        # Training setup (similar to conversion model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_auc = 0
        for epoch in range(self.model_config['epochs']):
            model.train()
            train_loss = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_probs = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of churn
                    
                    val_probs.extend(probs.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            val_auc = roc_auc_score(val_targets, val_probs)
            scheduler.step(train_loss)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'models/churn_model.pth')
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        logger.info(f"Churn model training completed. Best validation AUC: {best_val_auc:.4f}")
        return model
    
    def train_next_action_model(self, X: np.ndarray, y: np.ndarray) -> NextBestActionNetwork:
        """Train next best action model"""
        logger.info("Training next best action model...")
        
        num_actions = len(np.unique(y))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.model_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        # Initialize model
        model = NextBestActionNetwork(
            input_dim=X.shape[1],
            num_actions=num_actions,
            hidden_layers=self.model_config['hidden_layers'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        
        # Training loop
        best_val_acc = 0
        for epoch in range(self.model_config['epochs']):
            model.train()
            train_loss = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                
                optimizer.zero_grad()
                action_probs, confidence = model(batch_features)
                loss = criterion(action_probs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                    
                    action_probs, _ = model(batch_features)
                    predictions = torch.argmax(action_probs, dim=1)
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            val_acc = accuracy_score(val_targets, val_predictions)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/next_action_model.pth')
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info(f"Next action model training completed. Best validation accuracy: {best_val_acc:.4f}")
        return model
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models"""
        logger.info("Training all PyTorch models...")
        
        # Prepare data
        X, targets = self.prepare_data(df)
        
        models = {}
        
        # Train conversion model
        if 'conversion' in targets:
            models['conversion'] = self.train_conversion_model(X, targets['conversion'])
        
        # Train churn model
        if 'churn' in targets:
            models['churn'] = self.train_churn_model(X, targets['churn'])
        
        # Train next action model
        if 'next_action' in targets:
            models['next_action'] = self.train_next_action_model(X, targets['next_action'])
        
        # Save scalers and encoders
        joblib.dump(self.feature_scaler, 'models/feature_scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        logger.info("All PyTorch models trained successfully")
        return models
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, 
                      model_type: str) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        test_dataset = FinancialDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        predictions = []
        probabilities = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                
                if model_type == 'next_action':
                    action_probs, _ = model(batch_features)
                    outputs = action_probs
                else:
                    outputs = model(batch_features)
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                targets.extend(batch_targets.numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted')
        }
        
        if len(np.unique(targets)) == 2:  # Binary classification
            metrics['auc'] = roc_auc_score(targets, [p[1] for p in probabilities])
        
        return metrics

if __name__ == "__main__":
    # Example usage
    trainer = FinancialMLTrainer()
    
    # Load data
    df = pd.read_csv("data/synthetic/output/ml_features.csv")
    
    # Train all models
    models = trainer.train_all_models(df)
    
    logger.info("PyTorch models training completed")
