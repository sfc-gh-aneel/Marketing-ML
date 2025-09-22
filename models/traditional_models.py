"""
Traditional ML Models (XGBoost, Scikit-learn) for Financial Services
Optimized for Snowflake Model Registry integration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import joblib
import logging
import yaml
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class TraditionalMLTrainer:
    """Trainer for traditional ML models with Snowflake integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.xgb_config = self.config['models']['xgboost']
        self.sklearn_config = self.config['models']['sklearn']
        
        # Initialize preprocessors
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Model storage
        self.models = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """Prepare data for traditional ML models"""
        logger.info("Preparing data for traditional ML models...")
        
        # Identify feature columns
        feature_cols = [col for col in df.columns if not col.startswith(('CLIENT_ID', 'CONVERSION_TARGET', 'CHURN_TARGET', 'NEXT_BEST_ACTION', 'FEATURE_TIMESTAMP'))]
        
        # Handle categorical features
        categorical_features = []
        for col in feature_cols:
            if df[col].dtype == 'object':
                categorical_features.append(col)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        feature_names = feature_cols
        
        # Scale numerical features (except label encoded ones)
        numerical_features = [col for col in feature_cols if col not in categorical_features]
        if numerical_features:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        X = X.values
        
        # Prepare targets
        targets = {}
        
        if 'CONVERSION_TARGET' in df.columns:
            targets['conversion'] = df['CONVERSION_TARGET'].values
        
        if 'CHURN_TARGET' in df.columns:
            targets['churn'] = df['CHURN_TARGET'].values
        
        if 'NEXT_BEST_ACTION' in df.columns:
            le_action = LabelEncoder()
            targets['next_action'] = le_action.fit_transform(df['NEXT_BEST_ACTION'])
            self.label_encoders['next_action'] = le_action
        
        logger.info(f"Prepared features: {X.shape}, Feature names: {len(feature_names)}")
        return X, targets, feature_names
    
    def train_xgboost_models(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                           feature_names: List[str]) -> Dict[str, XGBClassifier]:
        """Train XGBoost models for all targets"""
        logger.info("Training XGBoost models...")
        
        xgb_models = {}
        
        for target_name, y in targets.items():
            logger.info(f"Training XGBoost model for {target_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.sklearn_config['test_size'], 
                random_state=self.sklearn_config['random_state'], stratify=y
            )
            
            # Configure XGBoost parameters
            xgb_params = {
                'max_depth': self.xgb_config['max_depth'],
                'learning_rate': self.xgb_config['learning_rate'],
                'n_estimators': self.xgb_config['n_estimators'],
                'subsample': self.xgb_config['subsample'],
                'random_state': self.sklearn_config['random_state'],
                'eval_metric': 'logloss' if len(np.unique(y)) == 2 else 'mlogloss'
            }
            
            # Handle multi-class vs binary classification
            if len(np.unique(y)) > 2:
                xgb_params['objective'] = 'multi:softprob'
                xgb_params['num_class'] = len(np.unique(y))
            else:
                xgb_params['objective'] = 'binary:logistic'
            
            # Train model
            model = XGBClassifier(**xgb_params)
            
            # Use early stopping for better generalization
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Store model and evaluate
            xgb_models[target_name] = model
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            logger.info(f"XGBoost {target_name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Store feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[f'xgb_{target_name}'] = importance_df
            
            # Save model
            model_path = f"models/xgb_{target_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved XGBoost {target_name} model to {model_path}")
        
        return xgb_models
    
    def train_random_forest_models(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                                 feature_names: List[str]) -> Dict[str, RandomForestClassifier]:
        """Train Random Forest models for all targets"""
        logger.info("Training Random Forest models...")
        
        rf_models = {}
        
        for target_name, y in targets.items():
            logger.info(f"Training Random Forest model for {target_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.sklearn_config['test_size'], 
                random_state=self.sklearn_config['random_state'], stratify=y
            )
            
            # Configure Random Forest
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.sklearn_config['random_state'],
                'n_jobs': -1
            }
            
            # Train model
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)
            
            # Store model and evaluate
            rf_models[target_name] = model
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            logger.info(f"Random Forest {target_name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Store feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[f'rf_{target_name}'] = importance_df
            
            # Save model
            model_path = f"models/rf_{target_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved Random Forest {target_name} model to {model_path}")
        
        return rf_models
    
    def train_logistic_regression_models(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                                       feature_names: List[str]) -> Dict[str, LogisticRegression]:
        """Train Logistic Regression models (baseline)"""
        logger.info("Training Logistic Regression models...")
        
        lr_models = {}
        
        for target_name, y in targets.items():
            logger.info(f"Training Logistic Regression model for {target_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.sklearn_config['test_size'], 
                random_state=self.sklearn_config['random_state'], stratify=y
            )
            
            # Configure Logistic Regression
            lr_params = {
                'random_state': self.sklearn_config['random_state'],
                'max_iter': 1000,
                'C': 1.0
            }
            
            # Handle multi-class
            if len(np.unique(y)) > 2:
                lr_params['multi_class'] = 'ovr'
                lr_params['solver'] = 'liblinear'
            else:
                lr_params['solver'] = 'liblinear'
            
            # Train model
            model = LogisticRegression(**lr_params)
            model.fit(X_train, y_train)
            
            # Store model and evaluate
            lr_models[target_name] = model
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            logger.info(f"Logistic Regression {target_name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Store feature importance (coefficients)
            if len(np.unique(y)) == 2:
                coefficients = model.coef_[0]
            else:
                coefficients = np.mean(np.abs(model.coef_), axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[f'lr_{target_name}'] = importance_df
            
            # Save model
            model_path = f"models/lr_{target_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved Logistic Regression {target_name} model to {model_path}")
        
        return lr_models
    
    def perform_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                                    model_type: str = 'xgboost') -> Dict[str, Any]:
        """Perform hyperparameter tuning for specified model type"""
        logger.info(f"Performing hyperparameter tuning for {model_type}")
        
        if model_type == 'xgboost':
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = XGBClassifier(random_state=42)
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def evaluate_all_models(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Evaluate all trained models and compare performance"""
        logger.info("Evaluating all models...")
        
        evaluation_results = []
        
        for target_name, y in targets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.sklearn_config['test_size'], 
                random_state=self.sklearn_config['random_state'], stratify=y
            )
            
            # Evaluate each model type
            model_types = ['xgb', 'rf', 'lr']
            
            for model_type in model_types:
                try:
                    model_path = f"models/{model_type}_{target_name}_model.pkl"
                    model = joblib.load(model_path)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # AUC for binary classification
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    
                    evaluation_results.append({
                        'target': target_name,
                        'model_type': model_type,
                        'accuracy': accuracy,
                        'auc': auc
                    })
                    
                except FileNotFoundError:
                    logger.warning(f"Model file not found: {model_path}")
                    continue
        
        results_df = pd.DataFrame(evaluation_results)
        
        # Save evaluation results
        results_df.to_csv("models/model_evaluation_results.csv", index=False)
        logger.info("Model evaluation results saved to models/model_evaluation_results.csv")
        
        return results_df
    
    def generate_feature_importance_report(self) -> pd.DataFrame:
        """Generate comprehensive feature importance report"""
        logger.info("Generating feature importance report...")
        
        importance_data = []
        
        for model_name, importance_df in self.feature_importance.items():
            for _, row in importance_df.head(20).iterrows():  # Top 20 features
                importance_data.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'importance': row['importance'],
                    'rank': importance_df.index.get_loc(row.name) + 1
                })
        
        importance_report = pd.DataFrame(importance_data)
        
        # Save report
        importance_report.to_csv("models/feature_importance_report.csv", index=False)
        logger.info("Feature importance report saved to models/feature_importance_report.csv")
        
        return importance_report
    
    def create_model_comparison_plots(self, evaluation_df: pd.DataFrame):
        """Create visualization plots for model comparison"""
        logger.info("Creating model comparison plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy comparison
        pivot_acc = evaluation_df.pivot(index='target', columns='model_type', values='accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0, 0], title='Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(title='Model Type')
        
        # Plot 2: AUC comparison
        pivot_auc = evaluation_df.pivot(index='target', columns='model_type', values='auc')
        pivot_auc.plot(kind='bar', ax=axes[0, 1], title='Model AUC Comparison')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend(title='Model Type')
        
        # Plot 3: Feature importance heatmap (top features)
        if self.feature_importance:
            # Get top features across all models
            all_features = set()
            for importance_df in self.feature_importance.values():
                all_features.update(importance_df.head(10)['feature'].tolist())
            
            # Create heatmap data
            heatmap_data = []
            for model_name, importance_df in list(self.feature_importance.items())[:6]:  # First 6 models
                importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
                row = [importance_dict.get(feature, 0) for feature in sorted(all_features)]
                heatmap_data.append(row)
            
            if heatmap_data:
                sns.heatmap(
                    heatmap_data, 
                    xticklabels=sorted(all_features), 
                    yticklabels=list(self.feature_importance.keys())[:6],
                    ax=axes[1, 0], 
                    cmap='YlOrRd',
                    title='Feature Importance Heatmap'
                )
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Model performance summary
        summary_stats = evaluation_df.groupby('model_type')[['accuracy', 'auc']].mean()
        summary_stats.plot(kind='bar', ax=axes[1, 1], title='Average Model Performance')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend(title='Metric')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plots saved to models/model_comparison_plots.png")
    
    def train_all_traditional_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all traditional ML models"""
        logger.info("Training all traditional ML models...")
        
        # Prepare data
        X, targets, feature_names = self.prepare_data(df)
        
        # Train models
        all_models = {}
        
        # XGBoost models
        xgb_models = self.train_xgboost_models(X, targets, feature_names)
        all_models.update({f'xgb_{k}': v for k, v in xgb_models.items()})
        
        # Random Forest models
        rf_models = self.train_random_forest_models(X, targets, feature_names)
        all_models.update({f'rf_{k}': v for k, v in rf_models.items()})
        
        # Logistic Regression models
        lr_models = self.train_logistic_regression_models(X, targets, feature_names)
        all_models.update({f'lr_{k}': v for k, v in lr_models.items()})
        
        # Evaluate models
        evaluation_df = self.evaluate_all_models(X, targets)
        
        # Generate reports and plots
        importance_report = self.generate_feature_importance_report()
        self.create_model_comparison_plots(evaluation_df)
        
        # Save preprocessors
        joblib.dump(self.scaler, 'models/traditional_scaler.pkl')
        joblib.dump(self.label_encoders, 'models/traditional_label_encoders.pkl')
        
        logger.info("All traditional ML models trained successfully")
        
        return {
            'models': all_models,
            'evaluation': evaluation_df,
            'feature_importance': importance_report,
            'preprocessors': {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }
        }

if __name__ == "__main__":
    # Example usage
    trainer = TraditionalMLTrainer()
    
    # Load data
    df = pd.read_csv("data/synthetic/output/ml_features.csv")
    
    # Train all traditional models
    results = trainer.train_all_traditional_models(df)
    
    logger.info("Traditional ML models training completed")
