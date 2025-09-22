"""
Snowflake Model Registry Integration
Register and manage ML models in Snowflake Model Registry
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import logging
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Snowflake ML imports
from snowflake.ml.registry import Registry
from snowflake.ml.model import Model as SnowflakeModel
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

# Model imports
import torch
import xgboost as xgb
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class SnowflakeModelRegistry:
    """Manage models in Snowflake Model Registry"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sf_config = self.config['snowflake']
        self.session = None
        self.registry = None
        
        self.model_metadata = {}
        
    def create_snowpark_session(self) -> Session:
        """Create Snowpark session for Model Registry"""
        connection_parameters = {
            "account": self.sf_config['account'],
            "user": self.sf_config['user'],
            "password": self.sf_config['password'],
            "role": self.sf_config['role'],
            "warehouse": self.sf_config['warehouse'],
            "database": self.sf_config['database'],
            "schema": self.sf_config['schema']
        }
        
        self.session = Session.builder.configs(connection_parameters).create()
        logger.info("Snowpark session created for Model Registry")
        return self.session
    
    def initialize_registry(self):
        """Initialize Snowflake Model Registry"""
        if not self.session:
            self.create_snowpark_session()
        
        self.registry = Registry(
            session=self.session,
            database_name=self.sf_config['database'],
            schema_name=self.sf_config['schema']
        )
        
        logger.info("Snowflake Model Registry initialized")
    
    def register_pytorch_model(self, model_path: str, model_name: str, 
                             model_type: str, version: str = "1.0",
                             sample_input: np.ndarray = None) -> str:
        """Register PyTorch model in Snowflake Model Registry"""
        logger.info(f"Registering PyTorch model: {model_name}")
        
        if not self.registry:
            self.initialize_registry()
        
        try:
            # Load the PyTorch model
            model_state = torch.load(model_path, map_location='cpu')
            
            # Create model metadata
            metadata = {
                "model_type": "pytorch",
                "task": model_type,
                "framework": "pytorch",
                "created_date": datetime.now().isoformat(),
                "model_path": model_path,
                "input_shape": str(sample_input.shape) if sample_input is not None else "unknown",
                "description": f"PyTorch {model_type} model for financial services"
            }
            
            # Create a wrapper class for PyTorch model
            class PyTorchModelWrapper:
                def __init__(self, model_state_dict, model_class):
                    self.model_state_dict = model_state_dict
                    self.model_class = model_class
                
                def predict(self, X):
                    # This would need to be implemented based on your specific PyTorch model
                    # For now, returning dummy predictions
                    return np.random.rand(len(X), 2)
            
            # Create model wrapper
            model_wrapper = PyTorchModelWrapper(model_state, None)
            
            # Register with Snowflake Model Registry
            snowflake_model = SnowflakeModel(
                model=model_wrapper,
                sample_input_data=sample_input if sample_input is not None else np.random.rand(10, 20)
            )
            
            # Log the model
            model_ref = self.registry.log_model(
                model=snowflake_model,
                model_name=model_name,
                version_name=version,
                tags={"framework": "pytorch", "task": model_type},
                comment=f"PyTorch model for {model_type} prediction"
            )
            
            self.model_metadata[model_name] = metadata
            
            logger.info(f"Successfully registered PyTorch model: {model_name} v{version}")
            return model_ref.model_name
            
        except Exception as e:
            logger.error(f"Failed to register PyTorch model {model_name}: {e}")
            raise
    
    def register_sklearn_model(self, model_path: str, model_name: str, 
                             model_type: str, version: str = "1.0",
                             sample_input: np.ndarray = None) -> str:
        """Register Scikit-learn model in Snowflake Model Registry"""
        logger.info(f"Registering Scikit-learn model: {model_name}")
        
        if not self.registry:
            self.initialize_registry()
        
        try:
            # Load the scikit-learn model
            model = joblib.load(model_path)
            
            # Create model metadata
            metadata = {
                "model_type": "sklearn",
                "task": model_type,
                "framework": "scikit-learn",
                "created_date": datetime.now().isoformat(),
                "model_path": model_path,
                "model_class": str(type(model).__name__),
                "input_shape": str(sample_input.shape) if sample_input is not None else "unknown",
                "description": f"Scikit-learn {model_type} model for financial services"
            }
            
            # Create Snowflake model
            snowflake_model = SnowflakeModel(
                model=model,
                sample_input_data=sample_input if sample_input is not None else np.random.rand(10, 20)
            )
            
            # Register with Snowflake Model Registry
            model_ref = self.registry.log_model(
                model=snowflake_model,
                model_name=model_name,
                version_name=version,
                tags={"framework": "sklearn", "task": model_type},
                comment=f"Scikit-learn model for {model_type} prediction"
            )
            
            self.model_metadata[model_name] = metadata
            
            logger.info(f"Successfully registered Scikit-learn model: {model_name} v{version}")
            return model_ref.model_name
            
        except Exception as e:
            logger.error(f"Failed to register Scikit-learn model {model_name}: {e}")
            raise
    
    def register_xgboost_model(self, model_path: str, model_name: str, 
                             model_type: str, version: str = "1.0",
                             sample_input: np.ndarray = None) -> str:
        """Register XGBoost model in Snowflake Model Registry"""
        logger.info(f"Registering XGBoost model: {model_name}")
        
        if not self.registry:
            self.initialize_registry()
        
        try:
            # Load the XGBoost model
            model = joblib.load(model_path)
            
            # Create model metadata
            metadata = {
                "model_type": "xgboost",
                "task": model_type,
                "framework": "xgboost",
                "created_date": datetime.now().isoformat(),
                "model_path": model_path,
                "model_class": "XGBClassifier",
                "input_shape": str(sample_input.shape) if sample_input is not None else "unknown",
                "description": f"XGBoost {model_type} model for financial services"
            }
            
            # Create Snowflake model
            snowflake_model = SnowflakeModel(
                model=model,
                sample_input_data=sample_input if sample_input is not None else np.random.rand(10, 20)
            )
            
            # Register with Snowflake Model Registry
            model_ref = self.registry.log_model(
                model=snowflake_model,
                model_name=model_name,
                version_name=version,
                tags={"framework": "xgboost", "task": model_type},
                comment=f"XGBoost model for {model_type} prediction"
            )
            
            self.model_metadata[model_name] = metadata
            
            logger.info(f"Successfully registered XGBoost model: {model_name} v{version}")
            return model_ref.model_name
            
        except Exception as e:
            logger.error(f"Failed to register XGBoost model {model_name}: {e}")
            raise
    
    def register_all_models(self, models_dir: str = "models", 
                          sample_data_path: str = "data/synthetic/output/ml_features.csv") -> Dict[str, str]:
        """Register all trained models in the registry"""
        logger.info("Registering all models in Snowflake Model Registry...")
        
        # Load sample data for model input schema
        if sample_data_path:
            sample_df = pd.read_csv(sample_data_path)
            feature_cols = [col for col in sample_df.columns if not col.startswith(('CLIENT_ID', 'CONVERSION_TARGET', 'CHURN_TARGET', 'NEXT_BEST_ACTION', 'FEATURE_TIMESTAMP'))]
            sample_input = sample_df[feature_cols].fillna(0).head(10).values
        else:
            sample_input = None
        
        registered_models = {}
        
        # Model registration mappings
        model_registrations = [
            # XGBoost models
            ("models/xgb_conversion_model.pkl", "XGB_Conversion_Predictor", "conversion", "xgboost"),
            ("models/xgb_churn_model.pkl", "XGB_Churn_Predictor", "churn", "xgboost"),
            ("models/xgb_next_action_model.pkl", "XGB_NextAction_Predictor", "next_action", "xgboost"),
            
            # Random Forest models
            ("models/rf_conversion_model.pkl", "RF_Conversion_Predictor", "conversion", "sklearn"),
            ("models/rf_churn_model.pkl", "RF_Churn_Predictor", "churn", "sklearn"),
            ("models/rf_next_action_model.pkl", "RF_NextAction_Predictor", "next_action", "sklearn"),
            
            # Logistic Regression models
            ("models/lr_conversion_model.pkl", "LR_Conversion_Predictor", "conversion", "sklearn"),
            ("models/lr_churn_model.pkl", "LR_Churn_Predictor", "churn", "sklearn"),
            ("models/lr_next_action_model.pkl", "LR_NextAction_Predictor", "next_action", "sklearn"),
        ]
        
        for model_path, model_name, model_type, framework in model_registrations:
            try:
                if framework == "xgboost":
                    model_ref = self.register_xgboost_model(model_path, model_name, model_type, sample_input=sample_input)
                elif framework == "sklearn":
                    model_ref = self.register_sklearn_model(model_path, model_name, model_type, sample_input=sample_input)
                elif framework == "pytorch":
                    model_ref = self.register_pytorch_model(model_path, model_name, model_type, sample_input=sample_input)
                
                registered_models[model_name] = model_ref
                
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
                continue
            except Exception as e:
                logger.error(f"Failed to register model {model_name}: {e}")
                continue
        
        logger.info(f"Successfully registered {len(registered_models)} models")
        return registered_models
    
    def list_registered_models(self) -> pd.DataFrame:
        """List all registered models in the registry"""
        if not self.registry:
            self.initialize_registry()
        
        try:
            models = self.registry.list_models()
            
            model_list = []
            for model in models:
                model_info = {
                    'model_name': model.name,
                    'latest_version': model.default_version_name,
                    'created_time': model.created_time,
                    'description': model.comment
                }
                model_list.append(model_info)
            
            return pd.DataFrame(model_list)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return pd.DataFrame()
    
    def get_model_for_inference(self, model_name: str, version: str = None) -> Any:
        """Retrieve model from registry for inference"""
        if not self.registry:
            self.initialize_registry()
        
        try:
            if version:
                model_ref = self.registry.get_model(model_name, version)
            else:
                model_ref = self.registry.get_model(model_name)
            
            # Load the actual model
            model = model_ref.load_model()
            
            logger.info(f"Successfully loaded model {model_name} for inference")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def update_model_metadata(self, model_name: str, metadata: Dict[str, Any]):
        """Update model metadata in registry"""
        if not self.registry:
            self.initialize_registry()
        
        try:
            # Get model reference
            model_ref = self.registry.get_model(model_name)
            
            # Update tags (metadata)
            for key, value in metadata.items():
                model_ref.set_tag(key, str(value))
            
            logger.info(f"Updated metadata for model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {model_name}: {e}")
            raise
    
    def create_model_performance_table(self, evaluation_results: pd.DataFrame):
        """Create table to track model performance metrics"""
        if not self.session:
            self.create_snowpark_session()
        
        try:
            # Convert evaluation results to Snowpark DataFrame
            performance_df = self.session.create_dataframe(evaluation_results)
            
            # Write to Snowflake table
            performance_df.write.mode("overwrite").save_as_table("MODEL_PERFORMANCE_TRACKING")
            
            logger.info("Model performance tracking table created")
            
        except Exception as e:
            logger.error(f"Failed to create performance tracking table: {e}")
            raise
    
    def deploy_model_for_inference(self, model_name: str, stage: str = "PRODUCTION") -> str:
        """Deploy model to specified stage for inference"""
        if not self.registry:
            self.initialize_registry()
        
        try:
            # Get model reference
            model_ref = self.registry.get_model(model_name)
            
            # Set deployment stage
            model_ref.set_tag("deployment_stage", stage)
            model_ref.set_tag("deployed_at", datetime.now().isoformat())
            
            # Create deployment record
            deployment_info = {
                "model_name": model_name,
                "stage": stage,
                "deployed_at": datetime.now(),
                "status": "ACTIVE"
            }
            
            logger.info(f"Model {model_name} deployed to {stage} stage")
            return f"{model_name}_{stage}"
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            raise
    
    def create_model_monitoring_view(self):
        """Create view for model monitoring and governance"""
        if not self.session:
            self.create_snowpark_session()
        
        monitoring_view_sql = """
        CREATE OR REPLACE VIEW MODEL_MONITORING_VIEW AS
        SELECT 
            mp.model_name,
            mp.model_version,
            mp.evaluation_date,
            mp.metric_name,
            mp.metric_value,
            mp.dataset_type,
            pr.prediction_timestamp,
            pr.client_id,
            pr.conversion_probability,
            pr.churn_probability,
            pr.recommended_action
        FROM MODEL_PERFORMANCE_METRICS mp
        LEFT JOIN MODEL_PREDICTIONS pr ON mp.model_name = SPLIT_PART(pr.model_name, '_', 1)
        WHERE mp.evaluation_date >= DATEADD(day, -30, CURRENT_DATE())
        ORDER BY mp.evaluation_date DESC, pr.prediction_timestamp DESC
        """
        
        try:
            self.session.sql(monitoring_view_sql).collect()
            logger.info("Model monitoring view created successfully")
        except Exception as e:
            logger.error(f"Failed to create monitoring view: {e}")
            raise
    
    def export_model_catalog(self) -> Dict[str, Any]:
        """Export comprehensive model catalog"""
        catalog = {
            "models": self.model_metadata,
            "registry_info": {
                "database": self.sf_config['database'],
                "schema": self.sf_config['schema'],
                "created_at": datetime.now().isoformat()
            },
            "registered_models": self.list_registered_models().to_dict('records') if self.registry else []
        }
        
        # Save catalog
        with open("models/model_catalog.json", "w") as f:
            json.dump(catalog, f, indent=2, default=str)
        
        logger.info("Model catalog exported to models/model_catalog.json")
        return catalog

if __name__ == "__main__":
    # Example usage
    registry_manager = SnowflakeModelRegistry()
    
    # Register all models
    registered_models = registry_manager.register_all_models()
    
    # Create monitoring infrastructure
    registry_manager.create_model_monitoring_view()
    
    # Export catalog
    catalog = registry_manager.export_model_catalog()
    
    logger.info("Model registry setup completed")
