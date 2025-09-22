-- Snowflake Setup for Financial Services ML Pipeline
-- Run these commands in Snowflake to set up the environment

-- 1. Create Database and Schema
CREATE DATABASE IF NOT EXISTS FINANCIAL_ML_DB;
USE DATABASE FINANCIAL_ML_DB;
CREATE SCHEMA IF NOT EXISTS ML_PIPELINE;
USE SCHEMA ML_PIPELINE;

-- 2. Create Warehouse for ML workloads
CREATE WAREHOUSE IF NOT EXISTS ML_WAREHOUSE 
    WITH WAREHOUSE_SIZE = 'LARGE'
    AUTO_SUSPEND = 300 
    AUTO_RESUME = TRUE
    COMMENT = 'Warehouse for ML training and inference';

-- 3. Create necessary roles and grants
CREATE ROLE IF NOT EXISTS ML_ENGINEER;
GRANT USAGE ON DATABASE FINANCIAL_ML_DB TO ROLE ML_ENGINEER;
GRANT USAGE ON SCHEMA ML_PIPELINE TO ROLE ML_ENGINEER;
GRANT USAGE ON WAREHOUSE ML_WAREHOUSE TO ROLE ML_ENGINEER;
GRANT CREATE TABLE ON SCHEMA ML_PIPELINE TO ROLE ML_ENGINEER;
GRANT CREATE VIEW ON SCHEMA ML_PIPELINE TO ROLE ML_ENGINEER;
GRANT CREATE STAGE ON SCHEMA ML_PIPELINE TO ROLE ML_ENGINEER;

-- 4. Create stages for data and model storage
CREATE STAGE IF NOT EXISTS ML_DATA_STAGE
    COMMENT = 'Stage for ML pipeline data files';

CREATE STAGE IF NOT EXISTS ML_MODEL_STAGE
    COMMENT = 'Stage for trained ML models';

-- 5. Enable Snowpark and ML features
-- (These are enabled by default in most Snowflake accounts)

-- 6. Create initial tables structure
CREATE OR REPLACE TABLE clients (
    client_id STRING PRIMARY KEY,
    created_date TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Demographics
    age INTEGER,
    gender STRING,
    marital_status STRING,
    education_level STRING,
    employment_status STRING,
    occupation STRING,
    annual_income INTEGER,
    
    -- Geographic
    state STRING,
    city STRING,
    zip_code STRING,
    
    -- Financial Profile
    current_401k_balance DECIMAL(12,2),
    years_to_retirement INTEGER,
    risk_tolerance STRING,
    investment_experience STRING,
    financial_goals ARRAY,
    
    -- Relationship Data
    client_tenure_months INTEGER,
    assigned_advisor_id STRING,
    service_tier STRING,
    total_assets_under_management DECIMAL(12,2),
    
    -- Engagement Profile
    preferred_contact_method STRING,
    last_contact_date TIMESTAMP_NTZ,
    communication_frequency_preference STRING
);

CREATE OR REPLACE TABLE advisors (
    advisor_id STRING PRIMARY KEY,
    advisor_name STRING,
    specialization STRING,
    years_experience INTEGER,
    client_count INTEGER,
    avg_client_satisfaction DECIMAL(3,2)
);

CREATE OR REPLACE TABLE marketing_events (
    event_id STRING PRIMARY KEY,
    client_id STRING,
    event_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    event_type STRING,
    event_category STRING,
    
    -- Web Events
    page_url STRING,
    session_id STRING,
    time_on_page INTEGER,
    referrer_source STRING,
    device_type STRING,
    
    -- Email Events
    campaign_id STRING,
    email_subject STRING,
    content_type STRING,
    
    -- Meeting/Call Events
    meeting_type STRING,
    duration_minutes INTEGER,
    advisor_id STRING,
    meeting_outcome STRING,
    
    -- Product Interest
    product_category STRING,
    product_specific STRING,
    action_taken STRING,
    
    -- Additional Metadata
    channel STRING,
    touchpoint_value DECIMAL(10,4),
    conversion_flag BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

-- 7. Create feature store table
CREATE OR REPLACE TABLE feature_store (
    client_id STRING,
    feature_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Engagement Features
    web_sessions_7d INTEGER,
    web_sessions_30d INTEGER,
    web_sessions_90d INTEGER,
    email_opens_30d INTEGER,
    email_clicks_30d INTEGER,
    avg_session_duration_30d DECIMAL(10,2),
    
    -- Advisor Interaction Features
    advisor_meetings_90d INTEGER,
    last_advisor_contact_days INTEGER,
    advisor_satisfaction_score DECIMAL(3,2),
    
    -- Financial Behavior Features
    portfolio_login_frequency_30d INTEGER,
    transaction_frequency_90d INTEGER,
    avg_transaction_amount_90d DECIMAL(12,2),
    
    -- Product Affinity Features
    retirement_tool_usage_30d INTEGER,
    investment_research_30d INTEGER,
    educational_content_engagement_30d INTEGER,
    
    -- Risk and Lifecycle Features
    client_lifecycle_stage STRING,
    days_since_last_activity INTEGER,
    engagement_score DECIMAL(5,4),
    churn_risk_score DECIMAL(5,4),
    conversion_propensity_score DECIMAL(5,4),
    
    -- Calculated Features
    total_touchpoints_30d INTEGER,
    digital_adoption_score DECIMAL(5,4),
    wealth_growth_potential STRING,
    
    PRIMARY KEY (client_id, feature_timestamp)
);

-- 8. Create model predictions table
CREATE OR REPLACE TABLE model_predictions (
    prediction_id STRING PRIMARY KEY,
    client_id STRING,
    model_name STRING,
    model_version STRING,
    prediction_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Next Best Action Predictions
    recommended_action STRING,
    action_probability DECIMAL(5,4),
    recommended_product STRING,
    recommendation_confidence DECIMAL(5,4),
    
    -- Conversion Predictions
    conversion_probability DECIMAL(5,4),
    expected_revenue DECIMAL(12,2),
    conversion_timeline_days INTEGER,
    
    -- Churn Predictions
    churn_probability DECIMAL(5,4),
    churn_risk_category STRING,
    days_to_churn INTEGER,
    retention_action STRING
);

-- 9. Set up model registry (using Snowflake's native ML registry)
-- This will be done programmatically in the notebooks

-- 10. Create monitoring views
CREATE OR REPLACE VIEW client_360_view AS
SELECT 
    c.*,
    a.advisor_name,
    a.specialization as advisor_specialization,
    fs.engagement_score,
    fs.churn_risk_score,
    fs.conversion_propensity_score,
    mp.recommended_action,
    mp.conversion_probability,
    COUNT(me.event_id) as total_events_30d
FROM clients c
LEFT JOIN advisors a ON c.assigned_advisor_id = a.advisor_id
LEFT JOIN feature_store fs ON c.client_id = fs.client_id 
    AND fs.feature_timestamp = (
        SELECT MAX(feature_timestamp) 
        FROM feature_store fs2 
        WHERE fs2.client_id = c.client_id
    )
LEFT JOIN model_predictions mp ON c.client_id = mp.client_id
    AND mp.prediction_timestamp = (
        SELECT MAX(prediction_timestamp)
        FROM model_predictions mp2
        WHERE mp2.client_id = c.client_id
    )
LEFT JOIN marketing_events me ON c.client_id = me.client_id
    AND me.event_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())
GROUP BY ALL;

-- Verify setup
SELECT 'Setup completed successfully!' as status;
SHOW TABLES;
