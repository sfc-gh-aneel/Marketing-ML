"""
Inference Pipeline for Financial Services ML Models
Batch and real-time inference using Snowflake infrastructure
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import joblib

# Snowflake imports
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when, lit
from snowflake.ml.registry import Registry

# Model imports
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FinancialInferencePipeline:
    """Inference pipeline for financial services ML models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sf_config = self.config['snowflake']
        self.session = None
        self.registry = None
        
        # Model cache
        self.loaded_models = {}
        self.preprocessors = {}
        
        # Prediction thresholds
        self.thresholds = {
            'conversion_threshold': 0.5,
            'churn_threshold': 0.6,
            'confidence_threshold': 0.7
        }
    
    def create_snowpark_session(self) -> Session:
        """Create Snowpark session for inference"""
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
        logger.info("Snowpark session created for inference")
        return self.session
    
    def initialize_registry(self):
        """Initialize model registry for loading models"""
        if not self.session:
            self.create_snowpark_session()
        
        self.registry = Registry(
            session=self.session,
            database_name=self.sf_config['database'],
            schema_name=self.sf_config['schema']
        )
        
        logger.info("Model registry initialized for inference")
    
    def load_preprocessors(self):
        """Load data preprocessors"""
        try:
            self.preprocessors['feature_scaler'] = joblib.load('models/traditional_scaler.pkl')
            self.preprocessors['label_encoders'] = joblib.load('models/traditional_label_encoders.pkl')
            logger.info("Preprocessors loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Preprocessor file not found: {e}")
            self.preprocessors = {}
    
    def load_model_from_registry(self, model_name: str, model_type: str = "sklearn") -> Any:
        """Load model from Snowflake Model Registry"""
        if not self.registry:
            self.initialize_registry()
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        try:
            model_ref = self.registry.get_model(model_name)
            model = model_ref.load_model()
            
            self.loaded_models[model_name] = model
            logger.info(f"Loaded model {model_name} from registry")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from registry: {e}")
            return None
    
    def load_local_model(self, model_path: str, model_type: str = "sklearn") -> Any:
        """Load model from local file"""
        try:
            if model_type == "pytorch":
                model = torch.load(model_path, map_location='cpu')
                model.eval()
            else:
                model = joblib.load(model_path)
            
            logger.info(f"Loaded local model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_path}: {e}")
            return None
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess features for inference"""
        # Get feature columns (excluding metadata)
        feature_cols = [col for col in df.columns if not col.startswith(('CLIENT_ID', 'FEATURE_TIMESTAMP'))]
        
        # Handle missing values
        df_processed = df[feature_cols].fillna(0)
        
        # Apply label encoding for categorical features
        if 'label_encoders' in self.preprocessors:
            for col_name, encoder in self.preprocessors['label_encoders'].items():
                if col_name in df_processed.columns:
                    try:
                        df_processed[col_name] = encoder.transform(df_processed[col_name].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df_processed[col_name] = 0
        
        # Apply scaling
        if 'feature_scaler' in self.preprocessors:
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                df_processed[numerical_cols] = self.preprocessors['feature_scaler'].transform(df_processed[numerical_cols])
        
        return df_processed.values
    
    def predict_conversion(self, features: np.ndarray, model_name: str = "XGB_Conversion_Predictor") -> Tuple[np.ndarray, np.ndarray]:
        """Predict conversion probability"""
        model = self.load_model_from_registry(model_name, "sklearn")
        if model is None:
            model = self.load_local_model("models/xgb_conversion_model.pkl", "sklearn")
        
        if model is None:
            logger.error("No conversion model available")
            return np.zeros(len(features)), np.zeros(len(features))
        
        try:
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]  # Probability of conversion
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in conversion prediction: {e}")
            return np.zeros(len(features)), np.zeros(len(features))
    
    def predict_churn(self, features: np.ndarray, model_name: str = "XGB_Churn_Predictor") -> Tuple[np.ndarray, np.ndarray]:
        """Predict churn probability"""
        model = self.load_model_from_registry(model_name, "sklearn")
        if model is None:
            model = self.load_local_model("models/xgb_churn_model.pkl", "sklearn")
        
        if model is None:
            logger.error("No churn model available")
            return np.zeros(len(features)), np.zeros(len(features))
        
        try:
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]  # Probability of churn
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in churn prediction: {e}")
            return np.zeros(len(features)), np.zeros(len(features))
    
    def predict_next_best_action(self, features: np.ndarray, model_name: str = "XGB_NextAction_Predictor") -> Tuple[np.ndarray, np.ndarray]:
        """Predict next best action"""
        model = self.load_model_from_registry(model_name, "sklearn")
        if model is None:
            model = self.load_local_model("models/xgb_next_action_model.pkl", "sklearn")
        
        if model is None:
            logger.error("No next action model available")
            return np.array(["Unknown"] * len(features)), np.zeros(len(features))
        
        try:
            predictions = model.predict(features)
            probabilities = np.max(model.predict_proba(features), axis=1)  # Max probability
            
            # Convert numeric predictions back to action names
            if 'next_action' in self.preprocessors.get('label_encoders', {}):
                encoder = self.preprocessors['label_encoders']['next_action']
                action_names = encoder.inverse_transform(predictions)
            else:
                action_names = [f"Action_{p}" for p in predictions]
            
            return action_names, probabilities
            
        except Exception as e:
            logger.error(f"Error in next action prediction: {e}")
            return np.array(["Unknown"] * len(features)), np.zeros(len(features))
    
    def generate_comprehensive_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive predictions for all models"""
        logger.info(f"Generating predictions for {len(df)} clients...")
        
        # Preprocess features
        features = self.preprocess_features(df)
        
        # Get predictions from all models
        conv_pred, conv_prob = self.predict_conversion(features)
        churn_pred, churn_prob = self.predict_churn(features)
        action_pred, action_conf = self.predict_next_best_action(features)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'CLIENT_ID': df['CLIENT_ID'],
            'PREDICTION_TIMESTAMP': datetime.now(),
            
            # Conversion predictions
            'CONVERSION_PREDICTION': conv_pred,
            'CONVERSION_PROBABILITY': conv_prob,
            'CONVERSION_RISK_CATEGORY': self._categorize_probability(conv_prob, 'conversion'),
            
            # Churn predictions
            'CHURN_PREDICTION': churn_pred,
            'CHURN_PROBABILITY': churn_prob,
            'CHURN_RISK_CATEGORY': self._categorize_probability(churn_prob, 'churn'),
            
            # Next best action
            'RECOMMENDED_ACTION': action_pred,
            'ACTION_CONFIDENCE': action_conf,
            'ACTION_PRIORITY': self._prioritize_actions(action_pred, action_conf),
            
            # Overall scores
            'ENGAGEMENT_PRIORITY_SCORE': self._calculate_engagement_score(conv_prob, churn_prob),
            'MODEL_VERSION': '1.0'
        })
        
        # Add business rules
        results = self._apply_business_rules(results, df)
        
        logger.info("Comprehensive predictions generated successfully")
        return results
    
    def _categorize_probability(self, probabilities: np.ndarray, prediction_type: str) -> List[str]:
        """Categorize probabilities into risk categories"""
        categories = []
        
        for prob in probabilities:
            if prediction_type == 'conversion':
                if prob >= 0.7:
                    categories.append('High_Likely')
                elif prob >= 0.4:
                    categories.append('Medium_Likely')
                else:
                    categories.append('Low_Likely')
            elif prediction_type == 'churn':
                if prob >= 0.7:
                    categories.append('High_Risk')
                elif prob >= 0.4:
                    categories.append('Medium_Risk')
                else:
                    categories.append('Low_Risk')
        
        return categories
    
    def _prioritize_actions(self, actions: np.ndarray, confidences: np.ndarray) -> List[str]:
        """Assign priority levels to actions based on confidence"""
        priorities = []
        
        for action, confidence in zip(actions, confidences):
            if confidence >= 0.8:
                priorities.append('High_Priority')
            elif confidence >= 0.6:
                priorities.append('Medium_Priority')
            else:
                priorities.append('Low_Priority')
        
        return priorities
    
    def _calculate_engagement_score(self, conv_prob: np.ndarray, churn_prob: np.ndarray) -> np.ndarray:
        """Calculate overall engagement priority score"""
        # Higher conversion probability and lower churn probability = higher engagement score
        engagement_score = (conv_prob * 0.6) + ((1 - churn_prob) * 0.4)
        return np.round(engagement_score, 4)
    
    def _apply_business_rules(self, predictions_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules to refine predictions"""
        
        # If available, use client tier information to adjust recommendations
        if 'SERVICE_TIER_NUMERIC' in features_df.columns:
            # Elite clients get priority regardless of scores
            elite_mask = features_df['SERVICE_TIER_NUMERIC'] == 3
            predictions_df.loc[elite_mask, 'ACTION_PRIORITY'] = 'High_Priority'
            predictions_df.loc[elite_mask, 'ENGAGEMENT_PRIORITY_SCORE'] = np.maximum(
                predictions_df.loc[elite_mask, 'ENGAGEMENT_PRIORITY_SCORE'], 0.8
            )
        
        # High-value clients (if balance information available)
        if 'CURRENT_401K_BALANCE' in features_df.columns:
            high_value_mask = features_df['CURRENT_401K_BALANCE'] > 100000
            predictions_df.loc[high_value_mask, 'RECOMMENDED_ACTION'] = 'Wealth_Advisory_Consultation'
        
        # Near-retirement clients
        if 'YEARS_TO_RETIREMENT' in features_df.columns:
            near_retirement_mask = features_df['YEARS_TO_RETIREMENT'] <= 10
            predictions_df.loc[near_retirement_mask, 'RECOMMENDED_ACTION'] = 'Retirement_Planning_Review'
        
        return predictions_df
    
    def batch_inference(self, input_table: str = "FEATURE_STORE", 
                       output_table: str = "MODEL_PREDICTIONS") -> int:
        """Run batch inference on feature store data"""
        logger.info("Starting batch inference...")
        
        if not self.session:
            self.create_snowpark_session()
        
        # Load preprocessors
        self.load_preprocessors()
        
        # Read features from Snowflake
        features_df = self.session.table(input_table).to_pandas()
        
        if len(features_df) == 0:
            logger.warning("No data found in feature store")
            return 0
        
        # Generate predictions
        predictions_df = self.generate_comprehensive_predictions(features_df)
        
        # Write predictions back to Snowflake
        predictions_snowpark_df = self.session.create_dataframe(predictions_df)
        predictions_snowpark_df.write.mode("overwrite").save_as_table(output_table)
        
        logger.info(f"Batch inference completed. Processed {len(predictions_df)} records")
        return len(predictions_df)
    
    def real_time_inference(self, client_features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform real-time inference for a single client"""
        
        # Load preprocessors if not already loaded
        if not self.preprocessors:
            self.load_preprocessors()
        
        try:
            # Convert to DataFrame for preprocessing
            features_df = pd.DataFrame([client_features])
            
            # Generate predictions
            predictions_df = self.generate_comprehensive_predictions(features_df)
            
            # Return as dictionary
            result = predictions_df.iloc[0].to_dict()
            
            # Add response metadata
            result['inference_type'] = 'real_time'
            result['response_time_ms'] = 50  # Placeholder
            
            return result
            
        except Exception as e:
            logger.error(f"Error in real-time inference: {e}")
            return {
                'error': str(e),
                'inference_type': 'real_time',
                'success': False
            }
    
    def create_inference_monitoring_table(self):
        """Create table to monitor inference performance"""
        if not self.session:
            self.create_snowpark_session()
        
        monitoring_sql = """
        CREATE OR REPLACE TABLE INFERENCE_MONITORING (
            inference_id STRING PRIMARY KEY,
            inference_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            inference_type STRING, -- batch, real_time
            model_versions OBJECT,
            input_records INTEGER,
            successful_predictions INTEGER,
            failed_predictions INTEGER,
            avg_prediction_time_ms DECIMAL(10,2),
            performance_metrics OBJECT
        )
        """
        
        try:
            self.session.sql(monitoring_sql).collect()
            logger.info("Inference monitoring table created")
        except Exception as e:
            logger.error(f"Failed to create monitoring table: {e}")
    
    def log_inference_metrics(self, inference_type: str, input_count: int, 
                            success_count: int, avg_time_ms: float):
        """Log inference performance metrics"""
        if not self.session:
            self.create_snowpark_session()
        
        metrics = {
            'inference_id': f"{inference_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'inference_type': inference_type,
            'input_records': input_count,
            'successful_predictions': success_count,
            'failed_predictions': input_count - success_count,
            'avg_prediction_time_ms': avg_time_ms,
            'inference_timestamp': datetime.now()
        }
        
        try:
            metrics_df = self.session.create_dataframe([metrics])
            metrics_df.write.mode("append").save_as_table("INFERENCE_MONITORING")
            logger.info("Inference metrics logged")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def create_prediction_api_view(self):
        """Create view for API access to latest predictions"""
        if not self.session:
            self.create_snowpark_session()
        
        api_view_sql = """
        CREATE OR REPLACE VIEW LATEST_CLIENT_PREDICTIONS AS
        SELECT 
            p.*,
            c.CLIENT_TENURE_MONTHS,
            c.SERVICE_TIER,
            c.ANNUAL_INCOME,
            c.TOTAL_ASSETS_UNDER_MANAGEMENT,
            ROW_NUMBER() OVER (PARTITION BY p.CLIENT_ID ORDER BY p.PREDICTION_TIMESTAMP DESC) as rn
        FROM MODEL_PREDICTIONS p
        LEFT JOIN CLIENTS c ON p.CLIENT_ID = c.CLIENT_ID
        QUALIFY rn = 1
        """
        
        try:
            self.session.sql(api_view_sql).collect()
            logger.info("Prediction API view created")
        except Exception as e:
            logger.error(f"Failed to create API view: {e}")

class InferenceScheduler:
    """Schedule and manage inference jobs"""
    
    def __init__(self, inference_pipeline: FinancialInferencePipeline):
        self.pipeline = inference_pipeline
        
    def schedule_daily_batch_inference(self):
        """Schedule daily batch inference job"""
        logger.info("Scheduling daily batch inference...")
        
        # This would typically integrate with Snowflake Tasks or external scheduler
        schedule_sql = """
        CREATE OR REPLACE TASK DAILY_INFERENCE_TASK
        WAREHOUSE = 'COMPUTE_WH'
        SCHEDULE = 'USING CRON 0 6 * * * UTC'  -- Daily at 6 AM UTC
        AS
        CALL RUN_BATCH_INFERENCE();
        """
        
        try:
            self.pipeline.session.sql(schedule_sql).collect()
            logger.info("Daily inference task scheduled")
        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
    
    def run_inference_job(self, job_type: str = "batch"):
        """Run inference job manually"""
        if job_type == "batch":
            return self.pipeline.batch_inference()
        else:
            logger.error(f"Unsupported job type: {job_type}")
            return 0

if __name__ == "__main__":
    # Example usage
    pipeline = FinancialInferencePipeline()
    
    # Create monitoring infrastructure
    pipeline.create_inference_monitoring_table()
    pipeline.create_prediction_api_view()
    
    # Run batch inference
    processed_count = pipeline.batch_inference()
    
    # Example real-time inference
    sample_features = {
        'CLIENT_ID': 'test-client-001',
        'AGE': 45,
        'ANNUAL_INCOME': 75000,
        'TOTAL_EVENTS_30D': 25,
        'EMAIL_OPENS_30D': 5,
        'WEB_VISITS_30D': 10
        # ... additional features
    }
    
    real_time_result = pipeline.real_time_inference(sample_features)
    print("Real-time prediction:", real_time_result)
    
    logger.info("Inference pipeline demo completed")
