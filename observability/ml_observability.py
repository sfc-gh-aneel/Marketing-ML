"""
ML Observability Suite for Financial Services Pipeline
Implements Snowflake ML Observability for model monitoring, drift detection, and performance tracking
"""

import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import *
from snowflake.ml.model_observability import ModelMonitor
from snowflake.ml.registry import Registry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class FinancialMLObservability:
    """ML Observability suite for financial services models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sf_config = self.config['snowflake']
        self.session = None
        self.registry = None
        self.monitors = {}
        
    def create_snowpark_session(self) -> Session:
        """Create Snowpark session"""
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
        logger.info("Snowpark session created for ML Observability")
        return self.session
    
    def setup_observability_infrastructure(self):
        """Set up tables and infrastructure for ML observability"""
        if not self.session:
            self.create_snowpark_session()
        
        # Create inference logging table
        inference_table_sql = """
        CREATE OR REPLACE TABLE ml_inference_logs (
            inference_id STRING PRIMARY KEY,
            model_name STRING NOT NULL,
            model_version STRING NOT NULL,
            timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            
            -- Client information
            client_id STRING,
            
            -- Input features (for drift detection)
            total_events_30d INTEGER,
            web_visits_30d INTEGER,
            email_opens_30d INTEGER,
            engagement_frequency DECIMAL(10,4),
            annual_income INTEGER,
            current_401k_balance DECIMAL(12,2),
            age INTEGER,
            service_tier STRING,
            risk_tolerance STRING,
            
            -- Model predictions
            conversion_probability DECIMAL(5,4),
            churn_probability DECIMAL(5,4),
            recommended_action STRING,
            prediction_confidence DECIMAL(5,4),
            
            -- Ground truth (when available)
            actual_conversion BOOLEAN,
            actual_churn BOOLEAN,
            actual_outcome STRING,
            
            -- Metadata
            inference_latency_ms INTEGER,
            model_performance_score DECIMAL(5,4),
            data_quality_score DECIMAL(5,4)
        )
        """
        
        # Create model performance tracking table
        performance_table_sql = """
        CREATE OR REPLACE TABLE ml_model_performance (
            performance_id STRING PRIMARY KEY,
            model_name STRING NOT NULL,
            model_version STRING NOT NULL,
            evaluation_date TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            
            -- Performance metrics
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            auc_score DECIMAL(5,4),
            
            -- Business metrics
            conversion_rate DECIMAL(5,4),
            false_positive_rate DECIMAL(5,4),
            false_negative_rate DECIMAL(5,4),
            
            -- Drift metrics
            feature_drift_score DECIMAL(5,4),
            prediction_drift_score DECIMAL(5,4),
            data_quality_score DECIMAL(5,4),
            
            -- Volume metrics
            total_inferences INTEGER,
            successful_inferences INTEGER,
            failed_inferences INTEGER,
            
            -- Segmented performance
            performance_by_segment OBJECT
        )
        """
        
        # Create drift detection table
        drift_table_sql = """
        CREATE OR REPLACE TABLE ml_drift_detection (
            drift_id STRING PRIMARY KEY,
            model_name STRING NOT NULL,
            model_version STRING NOT NULL,
            detection_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            
            -- Drift type
            drift_type STRING, -- 'feature_drift', 'prediction_drift', 'concept_drift'
            drift_severity STRING, -- 'low', 'medium', 'high', 'critical'
            
            -- Affected features/predictions
            affected_features ARRAY,
            drift_score DECIMAL(5,4),
            
            -- Statistical measures
            ks_statistic DECIMAL(5,4),
            p_value DECIMAL(10,8),
            jensen_shannon_distance DECIMAL(5,4),
            
            -- Comparison periods
            baseline_period_start TIMESTAMP_NTZ,
            baseline_period_end TIMESTAMP_NTZ,
            current_period_start TIMESTAMP_NTZ,
            current_period_end TIMESTAMP_NTZ,
            
            -- Recommendations
            recommended_action STRING,
            action_priority STRING,
            
            -- Metadata
            detection_method STRING,
            confidence_level DECIMAL(3,2)
        )
        """
        
        # Create alerts table
        alerts_table_sql = """
        CREATE OR REPLACE TABLE ml_observability_alerts (
            alert_id STRING PRIMARY KEY,
            alert_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            alert_type STRING, -- 'performance_degradation', 'drift_detected', 'data_quality', 'volume_anomaly'
            severity STRING, -- 'info', 'warning', 'critical'
            
            -- Model information
            model_name STRING,
            model_version STRING,
            
            -- Alert details
            alert_message STRING,
            metric_name STRING,
            current_value DECIMAL(10,4),
            threshold_value DECIMAL(10,4),
            
            -- Context
            affected_segments ARRAY,
            recommended_actions ARRAY,
            
            -- Status tracking
            status STRING DEFAULT 'open', -- 'open', 'acknowledged', 'resolved'
            acknowledged_by STRING,
            acknowledged_at TIMESTAMP_NTZ,
            resolved_at TIMESTAMP_NTZ,
            resolution_notes STRING
        )
        """
        
        # Execute table creation
        tables = [
            ("ML Inference Logs", inference_table_sql),
            ("Model Performance", performance_table_sql),
            ("Drift Detection", drift_table_sql),
            ("Observability Alerts", alerts_table_sql)
        ]
        
        for table_name, sql in tables:
            try:
                self.session.sql(sql).collect()
                logger.info(f"✓ Created {table_name} table")
            except Exception as e:
                logger.error(f"Failed to create {table_name} table: {e}")
        
        logger.info("ML Observability infrastructure setup completed")
    
    def create_model_monitors(self, model_names: List[str]):
        """Create model monitors for specified models"""
        if not self.session:
            self.create_snowpark_session()
        
        if not self.registry:
            self.registry = Registry(session=self.session)
        
        for model_name in model_names:
            try:
                # Get model from registry
                model_ref = self.registry.get_model(model_name)
                
                # Create model monitor
                monitor = ModelMonitor(
                    session=self.session,
                    model_name=model_name,
                    model_version=model_ref.version_name,
                    inference_table="ml_inference_logs",
                    timestamp_column="timestamp",
                    prediction_columns=["conversion_probability", "churn_probability"],
                    label_columns=["actual_conversion", "actual_churn"],
                    feature_columns=[
                        "total_events_30d", "web_visits_30d", "email_opens_30d",
                        "annual_income", "current_401k_balance", "age"
                    ]
                )
                
                self.monitors[model_name] = monitor
                logger.info(f"✓ Created monitor for model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to create monitor for {model_name}: {e}")
    
    def log_inference_data(self, inference_data: Dict[str, Any]):
        """Log inference data for monitoring"""
        if not self.session:
            self.create_snowpark_session()
        
        try:
            # Convert to DataFrame and insert
            inference_df = self.session.create_dataframe([inference_data])
            inference_df.write.mode("append").save_as_table("ml_inference_logs")
            
        except Exception as e:
            logger.error(f"Failed to log inference data: {e}")
    
    def detect_feature_drift(self, model_name: str, baseline_days: int = 30, 
                           current_days: int = 7) -> Dict[str, Any]:
        """Detect feature drift using statistical tests"""
        if not self.session:
            self.create_snowpark_session()
        
        # Get baseline and current data
        baseline_end = datetime.now() - timedelta(days=current_days)
        baseline_start = baseline_end - timedelta(days=baseline_days)
        current_start = datetime.now() - timedelta(days=current_days)
        
        drift_results = {}
        
        # Features to monitor for drift
        features_to_monitor = [
            "total_events_30d", "web_visits_30d", "email_opens_30d",
            "annual_income", "current_401k_balance", "age"
        ]
        
        for feature in features_to_monitor:
            # Get baseline distribution
            baseline_sql = f"""
            SELECT {feature} as feature_value
            FROM ml_inference_logs 
            WHERE model_name = '{model_name}'
              AND timestamp BETWEEN '{baseline_start}' AND '{baseline_end}'
              AND {feature} IS NOT NULL
            """
            
            # Get current distribution  
            current_sql = f"""
            SELECT {feature} as feature_value
            FROM ml_inference_logs 
            WHERE model_name = '{model_name}'
              AND timestamp >= '{current_start}'
              AND {feature} IS NOT NULL
            """
            
            try:
                baseline_data = self.session.sql(baseline_sql).to_pandas()
                current_data = self.session.sql(current_sql).to_pandas()
                
                if len(baseline_data) > 50 and len(current_data) > 50:
                    # Calculate Jensen-Shannon distance
                    js_distance = self._calculate_js_distance(
                        baseline_data['FEATURE_VALUE'].values,
                        current_data['FEATURE_VALUE'].values
                    )
                    
                    # Determine drift severity
                    if js_distance > 0.5:
                        severity = 'critical'
                    elif js_distance > 0.3:
                        severity = 'high'
                    elif js_distance > 0.15:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    drift_results[feature] = {
                        'js_distance': js_distance,
                        'severity': severity,
                        'baseline_mean': float(baseline_data['FEATURE_VALUE'].mean()),
                        'current_mean': float(current_data['FEATURE_VALUE'].mean()),
                        'baseline_std': float(baseline_data['FEATURE_VALUE'].std()),
                        'current_std': float(current_data['FEATURE_VALUE'].std())
                    }
            
            except Exception as e:
                logger.error(f"Error detecting drift for {feature}: {e}")
        
        # Log significant drift
        significant_drift = {k: v for k, v in drift_results.items() 
                           if v['severity'] in ['high', 'critical']}
        
        if significant_drift:
            self._log_drift_detection(model_name, 'feature_drift', significant_drift)
        
        return drift_results
    
    def _calculate_js_distance(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon distance between two distributions"""
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, 50)
        
        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # Normalize to probabilities
        baseline_prob = baseline_hist / baseline_hist.sum()
        current_prob = current_hist / current_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_prob += epsilon
        current_prob += epsilon
        
        # Calculate JS distance
        m = 0.5 * (baseline_prob + current_prob)
        js_distance = 0.5 * self._kl_divergence(baseline_prob, m) + \
                     0.5 * self._kl_divergence(current_prob, m)
        
        return float(np.sqrt(js_distance))
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence"""
        return np.sum(p * np.log(p / q))
    
    def _log_drift_detection(self, model_name: str, drift_type: str, drift_data: Dict):
        """Log drift detection results"""
        drift_record = {
            'drift_id': f"drift_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_name': model_name,
            'model_version': '1.0',  # Get from model registry
            'drift_type': drift_type,
            'drift_severity': max([d['severity'] for d in drift_data.values()]),
            'affected_features': list(drift_data.keys()),
            'drift_score': np.mean([d['js_distance'] for d in drift_data.values()]),
            'detection_method': 'jensen_shannon_distance',
            'confidence_level': 0.95,
            'recommended_action': self._get_drift_recommendation(drift_data)
        }
        
        try:
            drift_df = self.session.create_dataframe([drift_record])
            drift_df.write.mode("append").save_as_table("ml_drift_detection")
            
            # Create alert if severe
            if drift_record['drift_severity'] in ['high', 'critical']:
                self._create_alert('drift_detected', drift_record['drift_severity'], 
                                 model_name, f"Feature drift detected: {drift_record['drift_severity']}")
                
        except Exception as e:
            logger.error(f"Failed to log drift detection: {e}")
    
    def _get_drift_recommendation(self, drift_data: Dict) -> str:
        """Get recommendation based on drift severity"""
        max_severity = max([d['severity'] for d in drift_data.values()])
        
        if max_severity == 'critical':
            return 'immediate_model_retraining'
        elif max_severity == 'high':
            return 'schedule_model_retraining'
        elif max_severity == 'medium':
            return 'investigate_data_sources'
        else:
            return 'continue_monitoring'
    
    def monitor_model_performance(self, model_name: str, evaluation_period_days: int = 7):
        """Monitor model performance over specified period"""
        if not self.session:
            self.create_snowpark_session()
        
        # Calculate performance metrics
        performance_sql = f"""
        WITH predictions AS (
            SELECT 
                conversion_probability,
                churn_probability,
                actual_conversion,
                actual_churn,
                CASE 
                    WHEN conversion_probability > 0.5 THEN TRUE 
                    ELSE FALSE 
                END as predicted_conversion,
                CASE 
                    WHEN churn_probability > 0.5 THEN TRUE 
                    ELSE FALSE 
                END as predicted_churn
            FROM ml_inference_logs
            WHERE model_name = '{model_name}'
              AND timestamp >= DATEADD(day, -{evaluation_period_days}, CURRENT_TIMESTAMP())
              AND actual_conversion IS NOT NULL
              AND actual_churn IS NOT NULL
        ),
        
        conversion_metrics AS (
            SELECT 
                COUNT(*) as total_predictions,
                AVG(CASE WHEN predicted_conversion = actual_conversion THEN 1.0 ELSE 0.0 END) as accuracy,
                AVG(CASE WHEN predicted_conversion = TRUE AND actual_conversion = TRUE THEN 1.0 ELSE 0.0 END) / 
                NULLIF(AVG(CASE WHEN predicted_conversion = TRUE THEN 1.0 ELSE 0.0 END), 0) as precision_score,
                AVG(CASE WHEN predicted_conversion = TRUE AND actual_conversion = TRUE THEN 1.0 ELSE 0.0 END) / 
                NULLIF(AVG(CASE WHEN actual_conversion = TRUE THEN 1.0 ELSE 0.0 END), 0) as recall_score
            FROM predictions
        )
        
        SELECT 
            total_predictions,
            accuracy,
            precision_score,
            recall_score,
            2 * (precision_score * recall_score) / NULLIF((precision_score + recall_score), 0) as f1_score
        FROM conversion_metrics
        """
        
        try:
            performance_results = self.session.sql(performance_sql).collect()
            
            if performance_results:
                metrics = performance_results[0]
                
                # Log performance metrics
                performance_record = {
                    'performance_id': f"perf_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'model_name': model_name,
                    'model_version': '1.0',
                    'accuracy': float(metrics['ACCURACY']) if metrics['ACCURACY'] else 0.0,
                    'precision_score': float(metrics['PRECISION_SCORE']) if metrics['PRECISION_SCORE'] else 0.0,
                    'recall_score': float(metrics['RECALL_SCORE']) if metrics['RECALL_SCORE'] else 0.0,
                    'f1_score': float(metrics['F1_SCORE']) if metrics['F1_SCORE'] else 0.0,
                    'total_inferences': int(metrics['TOTAL_PREDICTIONS']),
                    'successful_inferences': int(metrics['TOTAL_PREDICTIONS']),
                    'failed_inferences': 0
                }
                
                perf_df = self.session.create_dataframe([performance_record])
                perf_df.write.mode("append").save_as_table("ml_model_performance")
                
                # Check for performance degradation
                if performance_record['accuracy'] < 0.75:  # Threshold
                    self._create_alert('performance_degradation', 'critical', model_name,
                                     f"Model accuracy dropped to {performance_record['accuracy']:.3f}")
                
                return performance_record
                
        except Exception as e:
            logger.error(f"Failed to monitor performance for {model_name}: {e}")
            return None
    
    def _create_alert(self, alert_type: str, severity: str, model_name: str, message: str):
        """Create observability alert"""
        alert_record = {
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'alert_type': alert_type,
            'severity': severity,
            'model_name': model_name,
            'model_version': '1.0',
            'alert_message': message,
            'recommended_actions': self._get_alert_recommendations(alert_type, severity)
        }
        
        try:
            alert_df = self.session.create_dataframe([alert_record])
            alert_df.write.mode("append").save_as_table("ml_observability_alerts")
            logger.warning(f"Alert created: {alert_type} - {severity} - {message}")
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    def _get_alert_recommendations(self, alert_type: str, severity: str) -> List[str]:
        """Get recommendations based on alert type and severity"""
        recommendations = {
            'performance_degradation': {
                'critical': ['immediate_model_retraining', 'investigate_data_quality', 'rollback_to_previous_version'],
                'warning': ['schedule_model_evaluation', 'increase_monitoring_frequency']
            },
            'drift_detected': {
                'critical': ['retrain_model_with_recent_data', 'investigate_data_sources'],
                'high': ['schedule_model_retraining', 'analyze_feature_distributions'],
                'medium': ['continue_monitoring', 'investigate_specific_features']
            },
            'data_quality': {
                'critical': ['pause_inference', 'investigate_data_pipeline'],
                'warning': ['increase_data_validation', 'monitor_upstream_sources']
            }
        }
        
        return recommendations.get(alert_type, {}).get(severity, ['investigate_further'])
    
    def create_observability_dashboard(self):
        """Create SQL views for observability dashboard"""
        if not self.session:
            self.create_snowpark_session()
        
        # Model performance dashboard
        performance_dashboard_sql = """
        CREATE OR REPLACE VIEW ml_performance_dashboard AS
        SELECT 
            model_name,
            model_version,
            evaluation_date,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            auc_score,
            total_inferences,
            
            -- Performance trends
            LAG(accuracy, 1) OVER (PARTITION BY model_name ORDER BY evaluation_date) as prev_accuracy,
            accuracy - LAG(accuracy, 1) OVER (PARTITION BY model_name ORDER BY evaluation_date) as accuracy_change,
            
            -- Status indicators
            CASE 
                WHEN accuracy >= 0.85 THEN 'excellent'
                WHEN accuracy >= 0.75 THEN 'good'
                WHEN accuracy >= 0.65 THEN 'fair'
                ELSE 'poor'
            END as performance_status
            
        FROM ml_model_performance
        WHERE evaluation_date >= DATEADD(day, -30, CURRENT_DATE())
        ORDER BY model_name, evaluation_date DESC
        """
        
        # Drift monitoring dashboard
        drift_dashboard_sql = """
        CREATE OR REPLACE VIEW ml_drift_dashboard AS
        SELECT 
            model_name,
            drift_type,
            drift_severity,
            affected_features,
            drift_score,
            detection_timestamp,
            recommended_action,
            
            -- Risk assessment
            CASE 
                WHEN drift_severity = 'critical' THEN 'immediate_attention'
                WHEN drift_severity = 'high' THEN 'urgent'
                WHEN drift_severity = 'medium' THEN 'monitor'
                ELSE 'normal'
            END as risk_level,
            
            -- Days since detection
            DATEDIFF(day, detection_timestamp, CURRENT_TIMESTAMP()) as days_since_detection
            
        FROM ml_drift_detection
        WHERE detection_timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP())
        ORDER BY detection_timestamp DESC, drift_severity DESC
        """
        
        # Alerts dashboard
        alerts_dashboard_sql = """
        CREATE OR REPLACE VIEW ml_alerts_dashboard AS
        SELECT 
            alert_type,
            severity,
            model_name,
            alert_message,
            alert_timestamp,
            status,
            acknowledged_by,
            
            -- Alert metrics
            COUNT(*) OVER (PARTITION BY alert_type, severity) as similar_alerts_count,
            
            -- Time metrics
            DATEDIFF(hour, alert_timestamp, CURRENT_TIMESTAMP()) as hours_since_alert,
            CASE 
                WHEN status = 'resolved' THEN DATEDIFF(hour, alert_timestamp, resolved_at)
                ELSE NULL
            END as resolution_time_hours
            
        FROM ml_observability_alerts
        WHERE alert_timestamp >= DATEADD(day, -14, CURRENT_TIMESTAMP())
        ORDER BY alert_timestamp DESC
        """
        
        # Execute dashboard creation
        dashboards = [
            ("Performance Dashboard", performance_dashboard_sql),
            ("Drift Dashboard", drift_dashboard_sql),
            ("Alerts Dashboard", alerts_dashboard_sql)
        ]
        
        for dashboard_name, sql in dashboards:
            try:
                self.session.sql(sql).collect()
                logger.info(f"✓ Created {dashboard_name}")
            except Exception as e:
                logger.error(f"Failed to create {dashboard_name}: {e}")
    
    def run_comprehensive_monitoring(self, model_names: List[str]):
        """Run comprehensive monitoring for all models"""
        logger.info("Starting comprehensive ML monitoring...")
        
        monitoring_results = {}
        
        for model_name in model_names:
            logger.info(f"Monitoring model: {model_name}")
            
            # 1. Performance monitoring
            performance = self.monitor_model_performance(model_name)
            
            # 2. Drift detection
            drift_results = self.detect_feature_drift(model_name)
            
            # 3. Data quality checks
            # (Additional data quality monitoring can be added here)
            
            monitoring_results[model_name] = {
                'performance': performance,
                'drift': drift_results,
                'timestamp': datetime.now()
            }
        
        logger.info("Comprehensive monitoring completed")
        return monitoring_results

if __name__ == "__main__":
    # Example usage
    observability = FinancialMLObservability()
    
    # Setup infrastructure
    observability.setup_observability_infrastructure()
    
    # Create dashboards
    observability.create_observability_dashboard()
    
    # Run monitoring for models
    model_names = ["CONVERSION_PREDICTOR", "CHURN_PREDICTOR", "NEXT_ACTION_PREDICTOR"]
    results = observability.run_comprehensive_monitoring(model_names)
    
    logger.info("ML Observability suite setup completed")
