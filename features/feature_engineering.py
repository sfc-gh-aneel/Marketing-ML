"""
Feature Engineering Pipeline for Financial Services ML
Uses Snowflake Feature Store for feature management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
from typing import Dict, List, Tuple, Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when, sum as sum_, count, avg, max as max_, min as min_
from snowflake.snowpark.functions import datediff, lag, lead, row_number
from snowflake.snowpark.window import Window
from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView

logger = logging.getLogger(__name__)

class FinancialFeatureEngineer:
    """Feature engineering for financial services ML pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sf_config = self.config['snowflake']
        self.feature_config = self.config['features']
        self.session = None
        self.feature_store = None
        
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
        logger.info("Snowpark session created successfully")
        return self.session
    
    def initialize_feature_store(self):
        """Initialize Snowflake Feature Store"""
        if not self.session:
            self.create_snowpark_session()
        
        self.feature_store = FeatureStore(
            session=self.session,
            database=self.sf_config['database'],
            name="FINANCIAL_FEATURE_STORE",
            default_warehouse=self.sf_config['warehouse']
        )
        
        logger.info("Feature Store initialized")
    
    def create_engagement_features(self) -> pd.DataFrame:
        """Create client engagement features"""
        logger.info("Creating engagement features...")
        
        if not self.session:
            self.create_snowpark_session()
        
        # Get marketing events data
        events_df = self.session.table("MARKETING_EVENTS")
        
        # Define time windows
        current_time = datetime.now()
        windows = self.feature_config['recency_days']
        
        engagement_features = []
        
        for days in windows:
            cutoff_date = current_time - timedelta(days=days)
            
            # Filter events within time window
            recent_events = events_df.filter(col("EVENT_TIMESTAMP") >= cutoff_date)
            
            # Aggregate engagement metrics
            client_engagement = recent_events.group_by("CLIENT_ID").agg([
                count("EVENT_ID").alias(f"TOTAL_EVENTS_{days}D"),
                count(when(col("EVENT_TYPE") == "web_visit", col("EVENT_ID"))).alias(f"WEB_VISITS_{days}D"),
                count(when(col("EVENT_TYPE") == "email_open", col("EVENT_ID"))).alias(f"EMAIL_OPENS_{days}D"),
                count(when(col("EVENT_TYPE") == "email_click", col("EVENT_ID"))).alias(f"EMAIL_CLICKS_{days}D"),
                count(when(col("EVENT_CATEGORY") == "Personal", col("EVENT_ID"))).alias(f"PERSONAL_INTERACTIONS_{days}D"),
                avg("TIME_ON_PAGE").alias(f"AVG_TIME_ON_PAGE_{days}D"),
                sum_("TOUCHPOINT_VALUE").alias(f"TOTAL_TOUCHPOINT_VALUE_{days}D"),
                count(when(col("CONVERSION_FLAG") == True, col("EVENT_ID"))).alias(f"CONVERSIONS_{days}D")
            ])
            
            engagement_features.append(client_engagement)
        
        # Combine all time windows
        base_features = engagement_features[0]
        for features in engagement_features[1:]:
            base_features = base_features.join(features, "CLIENT_ID", "outer")
        
        return base_features.to_pandas()
    
    def create_behavioral_features(self) -> pd.DataFrame:
        """Create behavioral pattern features"""
        logger.info("Creating behavioral features...")
        
        if not self.session:
            self.create_snowpark_session()
        
        events_df = self.session.table("MARKETING_EVENTS")
        
        # Create behavioral patterns
        behavioral_features = events_df.group_by("CLIENT_ID").agg([
            # Frequency patterns
            count("EVENT_ID").alias("TOTAL_LIFETIME_EVENTS"),
            
            # Time-based patterns
            datediff("day", min_("EVENT_TIMESTAMP"), max_("EVENT_TIMESTAMP")).alias("ENGAGEMENT_SPAN_DAYS"),
            
            # Channel preferences
            count(when(col("CHANNEL") == "Website", col("EVENT_ID"))).alias("WEB_PREFERENCE_COUNT"),
            count(when(col("CHANNEL") == "Email", col("EVENT_ID"))).alias("EMAIL_PREFERENCE_COUNT"),
            count(when(col("CHANNEL") == "Phone", col("EVENT_ID"))).alias("PHONE_PREFERENCE_COUNT"),
            
            # Device preferences
            count(when(col("DEVICE_TYPE") == "Desktop", col("EVENT_ID"))).alias("DESKTOP_USAGE"),
            count(when(col("DEVICE_TYPE") == "Mobile", col("EVENT_ID"))).alias("MOBILE_USAGE"),
            
            # Content engagement
            avg("TIME_ON_PAGE").alias("AVG_SESSION_DURATION"),
            count(when(col("EVENT_TYPE") == "document_download", col("EVENT_ID"))).alias("EDUCATION_ENGAGEMENT"),
            
            # Conversion behavior
            sum_(when(col("CONVERSION_FLAG") == True, 1).otherwise(0)).alias("TOTAL_CONVERSIONS"),
            avg("TOUCHPOINT_VALUE").alias("AVG_TOUCHPOINT_VALUE")
        ])
        
        # Calculate derived behavioral metrics
        behavioral_df = behavioral_features.to_pandas()
        
        # Engagement frequency (events per day)
        behavioral_df['ENGAGEMENT_FREQUENCY'] = behavioral_df['TOTAL_LIFETIME_EVENTS'] / np.maximum(behavioral_df['ENGAGEMENT_SPAN_DAYS'], 1)
        
        # Channel preference ratios
        total_events = behavioral_df['TOTAL_LIFETIME_EVENTS']
        behavioral_df['WEB_PREFERENCE_RATIO'] = behavioral_df['WEB_PREFERENCE_COUNT'] / np.maximum(total_events, 1)
        behavioral_df['EMAIL_PREFERENCE_RATIO'] = behavioral_df['EMAIL_PREFERENCE_COUNT'] / np.maximum(total_events, 1)
        behavioral_df['PHONE_PREFERENCE_RATIO'] = behavioral_df['PHONE_PREFERENCE_COUNT'] / np.maximum(total_events, 1)
        
        # Mobile adoption score
        behavioral_df['MOBILE_ADOPTION_SCORE'] = behavioral_df['MOBILE_USAGE'] / np.maximum((behavioral_df['MOBILE_USAGE'] + behavioral_df['DESKTOP_USAGE']), 1)
        
        # Conversion rate
        behavioral_df['CONVERSION_RATE'] = behavioral_df['TOTAL_CONVERSIONS'] / np.maximum(total_events, 1)
        
        return behavioral_df
    
    def create_financial_features(self) -> pd.DataFrame:
        """Create financial profile features"""
        logger.info("Creating financial features...")
        
        if not self.session:
            self.create_snowpark_session()
        
        clients_df = self.session.table("CLIENTS")
        
        # Create financial segments and scores
        financial_features = clients_df.select([
            col("CLIENT_ID"),
            col("AGE"),
            col("ANNUAL_INCOME"),
            col("CURRENT_401K_BALANCE"),
            col("YEARS_TO_RETIREMENT"),
            col("TOTAL_ASSETS_UNDER_MANAGEMENT"),
            col("CLIENT_TENURE_MONTHS"),
            col("RISK_TOLERANCE"),
            col("INVESTMENT_EXPERIENCE"),
            col("SERVICE_TIER")
        ])
        
        # Convert to pandas for complex calculations
        financial_df = financial_features.to_pandas()
        
        # Calculate derived financial metrics
        financial_df['INCOME_TO_AGE_RATIO'] = financial_df['ANNUAL_INCOME'] / np.maximum(financial_df['AGE'], 25)
        financial_df['ASSETS_TO_INCOME_RATIO'] = financial_df['TOTAL_ASSETS_UNDER_MANAGEMENT'] / np.maximum(financial_df['ANNUAL_INCOME'], 1)
        financial_df['RETIREMENT_READINESS_SCORE'] = self._calculate_retirement_readiness(financial_df)
        financial_df['WEALTH_GROWTH_POTENTIAL'] = self._calculate_wealth_potential(financial_df)
        financial_df['PREMIUM_CLIENT_INDICATOR'] = (financial_df['TOTAL_ASSETS_UNDER_MANAGEMENT'] > 100000).astype(int)
        
        # Risk profile encoding
        risk_encoding = {'Conservative': 1, 'Moderate': 2, 'Aggressive': 3}
        financial_df['RISK_TOLERANCE_NUMERIC'] = financial_df['RISK_TOLERANCE'].map(risk_encoding)
        
        # Experience encoding
        exp_encoding = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        financial_df['INVESTMENT_EXPERIENCE_NUMERIC'] = financial_df['INVESTMENT_EXPERIENCE'].map(exp_encoding)
        
        # Service tier encoding
        tier_encoding = {'Basic': 1, 'Premium': 2, 'Elite': 3}
        financial_df['SERVICE_TIER_NUMERIC'] = financial_df['SERVICE_TIER'].map(tier_encoding)
        
        return financial_df
    
    def _calculate_retirement_readiness(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate retirement readiness score"""
        # Simplified retirement readiness calculation
        target_multiple = 10  # Target is 10x annual income for retirement
        target_balance = df['ANNUAL_INCOME'] * target_multiple
        current_ratio = df['CURRENT_401K_BALANCE'] / np.maximum(target_balance, 1)
        
        # Adjust for years to retirement
        time_factor = np.maximum(df['YEARS_TO_RETIREMENT'], 1) / 40  # Normalize by 40 years
        readiness_score = current_ratio / time_factor
        
        return np.clip(readiness_score, 0, 1)
    
    def _calculate_wealth_potential(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate wealth growth potential score"""
        # Factors: age, income, current assets, time to retirement
        age_factor = (65 - df['AGE']) / 40  # More potential for younger clients
        income_factor = np.log(df['ANNUAL_INCOME']) / np.log(200000)  # Normalize by high income
        assets_factor = np.log(df['TOTAL_ASSETS_UNDER_MANAGEMENT'] + 1) / np.log(1000000)  # Normalize by $1M
        
        wealth_potential = (age_factor * 0.3 + income_factor * 0.4 + assets_factor * 0.3)
        return np.clip(wealth_potential, 0, 1)
    
    def create_lifecycle_features(self) -> pd.DataFrame:
        """Create client lifecycle and segmentation features"""
        logger.info("Creating lifecycle features...")
        
        if not self.session:
            self.create_snowpark_session()
        
        # Get latest client data
        clients_df = self.session.table("CLIENTS")
        events_df = self.session.table("MARKETING_EVENTS")
        
        # Calculate days since last activity
        latest_activity = events_df.group_by("CLIENT_ID").agg([
            max_("EVENT_TIMESTAMP").alias("LAST_ACTIVITY_DATE")
        ])
        
        # Join with client data
        lifecycle_df = clients_df.join(latest_activity, "CLIENT_ID", "left")
        
        # Convert to pandas for complex calculations
        lifecycle_data = lifecycle_df.to_pandas()
        
        # Calculate lifecycle metrics
        current_date = datetime.now()
        lifecycle_data['LAST_ACTIVITY_DATE'] = pd.to_datetime(lifecycle_data['LAST_ACTIVITY_DATE'])
        lifecycle_data['DAYS_SINCE_LAST_ACTIVITY'] = (current_date - lifecycle_data['LAST_ACTIVITY_DATE']).dt.days
        
        # Lifecycle stage determination
        lifecycle_data['LIFECYCLE_STAGE'] = lifecycle_data.apply(self._determine_lifecycle_stage, axis=1)
        
        # Tenure-based features
        lifecycle_data['TENURE_SEGMENT'] = pd.cut(
            lifecycle_data['CLIENT_TENURE_MONTHS'], 
            bins=[0, 6, 18, 36, float('inf')], 
            labels=['New', 'Growing', 'Established', 'Mature']
        )
        
        # Age-based segments
        lifecycle_data['AGE_SEGMENT'] = pd.cut(
            lifecycle_data['AGE'],
            bins=[0, 35, 50, 60, float('inf')],
            labels=['Young', 'Mid-Career', 'Pre-Retirement', 'Near-Retirement']
        )
        
        return lifecycle_data[['CLIENT_ID', 'DAYS_SINCE_LAST_ACTIVITY', 'LIFECYCLE_STAGE', 'TENURE_SEGMENT', 'AGE_SEGMENT']]
    
    def _determine_lifecycle_stage(self, row) -> str:
        """Determine client lifecycle stage"""
        days_inactive = row['DAYS_SINCE_LAST_ACTIVITY']
        tenure = row['CLIENT_TENURE_MONTHS']
        
        if pd.isna(days_inactive):
            return 'New'
        elif days_inactive > 90:
            return 'At_Risk'
        elif days_inactive > 180:
            return 'Dormant'
        elif tenure < 6:
            return 'New'
        elif tenure < 18:
            return 'Growing'
        else:
            return 'Active'
    
    def create_comprehensive_feature_set(self) -> pd.DataFrame:
        """Create comprehensive feature set combining all feature types"""
        logger.info("Creating comprehensive feature set...")
        
        # Generate all feature types
        engagement_features = self.create_engagement_features()
        behavioral_features = self.create_behavioral_features()
        financial_features = self.create_financial_features()
        lifecycle_features = self.create_lifecycle_features()
        
        # Merge all features
        feature_set = engagement_features
        feature_set = feature_set.merge(behavioral_features, on='CLIENT_ID', how='outer')
        feature_set = feature_set.merge(financial_features, on='CLIENT_ID', how='outer')
        feature_set = feature_set.merge(lifecycle_features, on='CLIENT_ID', how='outer')
        
        # Fill missing values with appropriate defaults
        feature_set = feature_set.fillna(0)
        
        # Add feature timestamp
        feature_set['FEATURE_TIMESTAMP'] = datetime.now()
        
        logger.info(f"Created feature set with {len(feature_set)} clients and {len(feature_set.columns)} features")
        
        return feature_set
    
    def create_target_variables(self) -> pd.DataFrame:
        """Create target variables for ML models"""
        logger.info("Creating target variables...")
        
        if not self.session:
            self.create_snowpark_session()
        
        clients_df = self.session.table("CLIENTS")
        events_df = self.session.table("MARKETING_EVENTS")
        
        # Future conversion prediction (30 days forward)
        future_date = datetime.now() + timedelta(days=30)
        
        # Simulate future conversions based on current engagement
        clients_data = clients_df.to_pandas()
        
        # Create synthetic targets based on client characteristics
        np.random.seed(42)
        
        # Conversion probability based on service tier, income, and engagement
        conversion_prob = (
            (clients_data['SERVICE_TIER'] == 'Elite').astype(int) * 0.3 +
            (clients_data['SERVICE_TIER'] == 'Premium').astype(int) * 0.2 +
            (clients_data['ANNUAL_INCOME'] > 75000).astype(int) * 0.2 +
            (clients_data['TOTAL_ASSETS_UNDER_MANAGEMENT'] > 50000).astype(int) * 0.2 +
            np.random.uniform(0, 0.1, len(clients_data))  # Random component
        )
        
        # Churn probability (inverse of conversion probability with adjustments)
        churn_prob = np.maximum(0.05, 0.3 - conversion_prob + np.random.uniform(-0.1, 0.1, len(clients_data)))
        
        targets = pd.DataFrame({
            'CLIENT_ID': clients_data['CLIENT_ID'],
            'CONVERSION_TARGET': (np.random.uniform(0, 1, len(clients_data)) < conversion_prob).astype(int),
            'CHURN_TARGET': (np.random.uniform(0, 1, len(clients_data)) < churn_prob).astype(int),
            'CONVERSION_PROBABILITY': conversion_prob,
            'CHURN_PROBABILITY': churn_prob,
            'NEXT_BEST_ACTION': self._generate_next_best_actions(clients_data, conversion_prob)
        })
        
        return targets
    
    def _generate_next_best_actions(self, clients_data: pd.DataFrame, conversion_prob: np.ndarray) -> List[str]:
        """Generate next best action recommendations"""
        actions = []
        
        for i, (_, client) in enumerate(clients_data.iterrows()):
            prob = conversion_prob[i]
            
            if client['SERVICE_TIER'] == 'Basic' and prob > 0.3:
                actions.append('Upgrade_Service_Tier')
            elif client['TOTAL_ASSETS_UNDER_MANAGEMENT'] < 25000 and prob > 0.25:
                actions.append('Schedule_Planning_Session')
            elif client['AGE'] > 55 and prob > 0.2:
                actions.append('Retirement_Planning_Review')
            elif prob > 0.4:
                actions.append('Wealth_Advisory_Consultation')
            elif prob < 0.1:
                actions.append('Educational_Content')
            else:
                actions.append('Relationship_Building')
                
        return actions
    
    def save_to_feature_store(self, feature_df: pd.DataFrame):
        """Save features to Snowflake Feature Store"""
        if not self.feature_store:
            self.initialize_feature_store()
        
        # Create entity for clients
        client_entity = Entity(name="CLIENT", join_keys=["CLIENT_ID"])
        
        # Create feature view
        feature_view = FeatureView(
            name="CLIENT_FEATURES",
            entities=[client_entity],
            feature_df=self.session.create_dataframe(feature_df),
            timestamp_col="FEATURE_TIMESTAMP"
        )
        
        # Register feature view
        self.feature_store.register_feature_view(
            feature_view=feature_view,
            version="1.0"
        )
        
        logger.info("Features saved to Feature Store")

if __name__ == "__main__":
    # Example usage
    engineer = FinancialFeatureEngineer()
    
    # Create comprehensive feature set
    features = engineer.create_comprehensive_feature_set()
    targets = engineer.create_target_variables()
    
    # Merge features and targets
    ml_dataset = features.merge(targets, on='CLIENT_ID', how='inner')
    
    # Save to CSV for model training
    ml_dataset.to_csv("data/synthetic/output/ml_features.csv", index=False)
    logger.info("ML dataset saved to data/synthetic/output/ml_features.csv")
