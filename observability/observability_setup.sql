-- ML Observability Setup for Financial Services Pipeline
-- Creates comprehensive monitoring infrastructure for model observability

-- ========================================
-- CORE OBSERVABILITY TABLES
-- ========================================

-- 1. Inference Logging Table (Core requirement for ML Observability)
CREATE OR REPLACE TABLE ml_inference_logs (
    inference_id STRING PRIMARY KEY,
    model_name STRING NOT NULL,
    model_version STRING NOT NULL,
    timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Client context
    client_id STRING,
    segment STRING, -- For segmented analysis
    
    -- Input features (for drift detection)
    total_events_30d INTEGER,
    web_visits_30d INTEGER,
    email_opens_30d INTEGER,
    email_clicks_30d INTEGER,
    engagement_frequency DECIMAL(10,4),
    mobile_adoption_score DECIMAL(5,4),
    
    -- Financial features
    annual_income INTEGER,
    current_401k_balance DECIMAL(12,2),
    total_assets_under_management DECIMAL(12,2),
    age INTEGER,
    years_to_retirement INTEGER,
    
    -- Categorical features
    service_tier STRING,
    risk_tolerance STRING,
    investment_experience STRING,
    lifecycle_stage STRING,
    
    -- Model predictions
    conversion_probability DECIMAL(5,4),
    churn_probability DECIMAL(5,4),
    recommended_action STRING,
    prediction_confidence DECIMAL(5,4),
    engagement_priority_score DECIMAL(5,4),
    
    -- Ground truth labels (updated when available)
    actual_conversion BOOLEAN,
    actual_churn BOOLEAN,
    actual_next_action STRING,
    outcome_timestamp TIMESTAMP_NTZ,
    
    -- Quality metrics
    inference_latency_ms INTEGER,
    data_quality_score DECIMAL(5,4),
    feature_completeness DECIMAL(3,2),
    
    -- Metadata
    inference_source STRING, -- 'batch', 'real_time', 'batch_scoring'
    model_deployment_stage STRING -- 'production', 'staging', 'canary'
);

-- 2. Model Performance Tracking
CREATE OR REPLACE TABLE ml_model_performance (
    performance_id STRING PRIMARY KEY,
    model_name STRING NOT NULL,
    model_version STRING NOT NULL,
    evaluation_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    evaluation_period_start TIMESTAMP_NTZ,
    evaluation_period_end TIMESTAMP_NTZ,
    
    -- Overall performance metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    auc_pr DECIMAL(5,4),
    
    -- Business-specific metrics
    conversion_rate DECIMAL(5,4),
    false_positive_rate DECIMAL(5,4),
    false_negative_rate DECIMAL(5,4),
    revenue_impact DECIMAL(12,2),
    cost_savings DECIMAL(12,2),
    
    -- Calibration metrics
    brier_score DECIMAL(5,4),
    log_loss DECIMAL(5,4),
    calibration_error DECIMAL(5,4),
    
    -- Volume and reliability
    total_inferences INTEGER,
    successful_inferences INTEGER,
    failed_inferences INTEGER,
    average_latency_ms DECIMAL(8,2),
    p95_latency_ms DECIMAL(8,2),
    p99_latency_ms DECIMAL(8,2),
    
    -- Segmented performance (JSON object)
    performance_by_segment OBJECT,
    
    -- Trend indicators
    performance_trend STRING, -- 'improving', 'stable', 'degrading'
    alert_threshold_breached BOOLEAN DEFAULT FALSE
);

-- 3. Drift Detection and Monitoring
CREATE OR REPLACE TABLE ml_drift_detection (
    drift_id STRING PRIMARY KEY,
    model_name STRING NOT NULL,
    model_version STRING NOT NULL,
    detection_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Drift classification
    drift_type STRING, -- 'feature_drift', 'prediction_drift', 'concept_drift', 'target_drift'
    drift_category STRING, -- 'univariate', 'multivariate', 'distributional'
    drift_severity STRING, -- 'low', 'medium', 'high', 'critical'
    
    -- Affected components
    affected_features ARRAY,
    primary_drift_feature STRING,
    drift_score DECIMAL(5,4),
    drift_magnitude DECIMAL(5,4),
    
    -- Statistical tests and measures
    test_method STRING, -- 'ks_test', 'jensen_shannon', 'chi_square', 'psi'
    test_statistic DECIMAL(10,6),
    p_value DECIMAL(15,10),
    confidence_level DECIMAL(3,2),
    
    -- Distribution metrics
    jensen_shannon_distance DECIMAL(5,4),
    kullback_leibler_divergence DECIMAL(5,4),
    population_stability_index DECIMAL(5,4),
    wasserstein_distance DECIMAL(5,4),
    
    -- Time windows
    baseline_period_start TIMESTAMP_NTZ,
    baseline_period_end TIMESTAMP_NTZ,
    current_period_start TIMESTAMP_NTZ,
    current_period_end TIMESTAMP_NTZ,
    baseline_sample_size INTEGER,
    current_sample_size INTEGER,
    
    -- Impact assessment
    performance_impact_estimated DECIMAL(5,4),
    business_impact_severity STRING,
    
    -- Recommendations and actions
    recommended_action STRING,
    action_priority STRING, -- 'immediate', 'urgent', 'normal', 'low'
    estimated_retraining_effort STRING,
    
    -- Resolution tracking
    status STRING DEFAULT 'detected', -- 'detected', 'investigating', 'resolved', 'false_positive'
    resolved_by STRING,
    resolved_timestamp TIMESTAMP_NTZ,
    resolution_method STRING
);

-- 4. Data Quality Monitoring
CREATE OR REPLACE TABLE ml_data_quality (
    quality_id STRING PRIMARY KEY,
    model_name STRING,
    check_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Quality dimensions
    completeness_score DECIMAL(5,4),
    validity_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    timeliness_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    
    -- Feature-level quality
    feature_quality_scores OBJECT,
    missing_value_rates OBJECT,
    outlier_rates OBJECT,
    
    -- Overall quality assessment
    overall_quality_score DECIMAL(5,4),
    quality_grade STRING, -- 'A', 'B', 'C', 'D', 'F'
    
    -- Issues detected
    quality_issues ARRAY,
    critical_issues_count INTEGER,
    warning_issues_count INTEGER,
    
    -- Recommendations
    quality_recommendations ARRAY
);

-- 5. Alerts and Notifications
CREATE OR REPLACE TABLE ml_observability_alerts (
    alert_id STRING PRIMARY KEY,
    alert_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    alert_type STRING, -- 'performance_degradation', 'drift_detected', 'data_quality', 'volume_anomaly', 'latency_spike'
    severity STRING, -- 'info', 'warning', 'critical', 'emergency'
    urgency STRING, -- 'low', 'medium', 'high', 'critical'
    
    -- Source information
    model_name STRING,
    model_version STRING,
    component STRING, -- 'inference', 'training', 'feature_store', 'data_pipeline'
    
    -- Alert details
    alert_title STRING,
    alert_message STRING,
    metric_name STRING,
    current_value DECIMAL(15,6),
    threshold_value DECIMAL(15,6),
    deviation_percentage DECIMAL(5,2),
    
    -- Context and impact
    affected_segments ARRAY,
    business_impact_assessment STRING,
    technical_impact_assessment STRING,
    
    -- Recommendations
    recommended_actions ARRAY,
    escalation_required BOOLEAN DEFAULT FALSE,
    estimated_resolution_time STRING,
    
    -- Tracking and resolution
    status STRING DEFAULT 'open', -- 'open', 'acknowledged', 'investigating', 'resolved', 'closed'
    assigned_to STRING,
    acknowledged_by STRING,
    acknowledged_at TIMESTAMP_NTZ,
    investigation_notes STRING,
    resolved_by STRING,
    resolved_at TIMESTAMP_NTZ,
    resolution_method STRING,
    resolution_notes STRING,
    
    -- Follow-up
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_date TIMESTAMP_NTZ,
    lessons_learned STRING
);

-- ========================================
-- MONITORING VIEWS AND DASHBOARDS
-- ========================================

-- 6. Real-time Model Health Dashboard
CREATE OR REPLACE VIEW model_health_dashboard AS
SELECT 
    mp.model_name,
    mp.model_version,
    mp.evaluation_timestamp,
    
    -- Performance indicators
    mp.accuracy,
    mp.f1_score,
    mp.auc_roc,
    CASE 
        WHEN mp.accuracy >= 0.85 THEN '🟢 Excellent'
        WHEN mp.accuracy >= 0.75 THEN '🟡 Good'
        WHEN mp.accuracy >= 0.65 THEN '🟠 Fair'
        ELSE '🔴 Poor'
    END as performance_status,
    
    -- Drift indicators
    dd.drift_severity,
    dd.drift_score,
    dd.detection_timestamp as last_drift_detection,
    
    -- Data quality
    dq.overall_quality_score,
    dq.quality_grade,
    
    -- Alert summary
    COUNT(CASE WHEN a.severity IN ('critical', 'emergency') AND a.status = 'open' THEN 1 END) as critical_alerts,
    COUNT(CASE WHEN a.severity = 'warning' AND a.status = 'open' THEN 1 END) as warning_alerts,
    
    -- Volume metrics
    mp.total_inferences,
    mp.average_latency_ms,
    
    -- Overall health score
    CASE 
        WHEN mp.accuracy >= 0.8 AND 
             COALESCE(dd.drift_severity, 'low') IN ('low', 'medium') AND 
             dq.overall_quality_score >= 0.9 AND
             COUNT(CASE WHEN a.severity IN ('critical', 'emergency') AND a.status = 'open' THEN 1 END) = 0
        THEN '🟢 Healthy'
        WHEN mp.accuracy >= 0.7 AND 
             COALESCE(dd.drift_severity, 'low') != 'critical' AND 
             dq.overall_quality_score >= 0.8
        THEN '🟡 Caution'
        ELSE '🔴 Attention Required'
    END as overall_health_status

FROM ml_model_performance mp
LEFT JOIN (
    SELECT model_name, model_version, drift_severity, drift_score, detection_timestamp,
           ROW_NUMBER() OVER (PARTITION BY model_name, model_version ORDER BY detection_timestamp DESC) as rn
    FROM ml_drift_detection
) dd ON mp.model_name = dd.model_name AND mp.model_version = dd.model_version AND dd.rn = 1
LEFT JOIN (
    SELECT model_name, overall_quality_score, quality_grade,
           ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY check_timestamp DESC) as rn
    FROM ml_data_quality
) dq ON mp.model_name = dq.model_name AND dq.rn = 1
LEFT JOIN ml_observability_alerts a ON mp.model_name = a.model_name AND a.status = 'open'
WHERE mp.evaluation_timestamp >= DATEADD(day, -1, CURRENT_TIMESTAMP())
GROUP BY ALL
ORDER BY mp.model_name, mp.evaluation_timestamp DESC;

-- 7. Drift Monitoring Dashboard
CREATE OR REPLACE VIEW drift_monitoring_dashboard AS
SELECT 
    model_name,
    model_version,
    drift_type,
    drift_severity,
    affected_features,
    drift_score,
    test_method,
    test_statistic,
    p_value,
    detection_timestamp,
    
    -- Risk assessment
    CASE 
        WHEN drift_severity = 'critical' THEN '🚨 Immediate Action Required'
        WHEN drift_severity = 'high' THEN '⚠️ Urgent Attention'
        WHEN drift_severity = 'medium' THEN '⚡ Monitor Closely'
        ELSE '✅ Normal Range'
    END as risk_assessment,
    
    -- Time since detection
    DATEDIFF(hour, detection_timestamp, CURRENT_TIMESTAMP()) as hours_since_detection,
    
    -- Recommendation priority
    CASE 
        WHEN drift_severity = 'critical' AND DATEDIFF(hour, detection_timestamp, CURRENT_TIMESTAMP()) > 24 
        THEN '🔥 Overdue Action'
        WHEN drift_severity = 'high' AND DATEDIFF(hour, detection_timestamp, CURRENT_TIMESTAMP()) > 72
        THEN '⏰ Action Needed Soon'
        ELSE recommended_action
    END as action_status,
    
    status,
    recommended_action

FROM ml_drift_detection
WHERE detection_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())
ORDER BY 
    CASE drift_severity 
        WHEN 'critical' THEN 1 
        WHEN 'high' THEN 2 
        WHEN 'medium' THEN 3 
        ELSE 4 
    END,
    detection_timestamp DESC;

-- 8. Performance Trends View
CREATE OR REPLACE VIEW model_performance_trends AS
SELECT 
    model_name,
    model_version,
    evaluation_timestamp,
    accuracy,
    f1_score,
    auc_roc,
    total_inferences,
    
    -- Moving averages
    AVG(accuracy) OVER (
        PARTITION BY model_name, model_version 
        ORDER BY evaluation_timestamp 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as accuracy_7day_avg,
    
    AVG(f1_score) OVER (
        PARTITION BY model_name, model_version 
        ORDER BY evaluation_timestamp 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as f1_7day_avg,
    
    -- Trends
    accuracy - LAG(accuracy, 1) OVER (
        PARTITION BY model_name, model_version 
        ORDER BY evaluation_timestamp
    ) as accuracy_change,
    
    accuracy - LAG(accuracy, 7) OVER (
        PARTITION BY model_name, model_version 
        ORDER BY evaluation_timestamp
    ) as accuracy_7day_change,
    
    -- Performance stability
    STDDEV(accuracy) OVER (
        PARTITION BY model_name, model_version 
        ORDER BY evaluation_timestamp 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as accuracy_volatility

FROM ml_model_performance
ORDER BY model_name, evaluation_timestamp DESC;

-- 9. Alert Summary View
CREATE OR REPLACE VIEW alerts_summary_dashboard AS
SELECT 
    DATE(alert_timestamp) as alert_date,
    alert_type,
    severity,
    model_name,
    COUNT(*) as alert_count,
    
    -- Resolution metrics
    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_count,
    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_count,
    
    -- Response times
    AVG(CASE 
        WHEN acknowledged_at IS NOT NULL 
        THEN DATEDIFF(minute, alert_timestamp, acknowledged_at) 
    END) as avg_acknowledgment_time_minutes,
    
    AVG(CASE 
        WHEN resolved_at IS NOT NULL 
        THEN DATEDIFF(hour, alert_timestamp, resolved_at) 
    END) as avg_resolution_time_hours,
    
    -- Current status
    MAX(alert_timestamp) as latest_alert_time,
    COUNT(CASE 
        WHEN status = 'open' AND severity IN ('critical', 'emergency') 
        THEN 1 
    END) as critical_open_alerts

FROM ml_observability_alerts
WHERE alert_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())
GROUP BY alert_date, alert_type, severity, model_name
ORDER BY alert_date DESC, alert_count DESC;

-- ========================================
-- AUTOMATED MONITORING PROCEDURES
-- ========================================

-- 10. Automated Performance Monitoring Procedure
CREATE OR REPLACE PROCEDURE run_performance_monitoring()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    model_cursor CURSOR FOR 
        SELECT DISTINCT model_name, model_version 
        FROM ml_inference_logs 
        WHERE timestamp >= DATEADD(day, -1, CURRENT_TIMESTAMP());
    
    model_name STRING;
    model_version STRING;
    performance_metrics OBJECT;
BEGIN
    
    FOR model_record IN model_cursor DO
        model_name := model_record.model_name;
        model_version := model_record.model_version;
        
        -- Calculate performance metrics for last 24 hours
        INSERT INTO ml_model_performance (
            performance_id, model_name, model_version,
            evaluation_period_start, evaluation_period_end,
            accuracy, precision_score, recall_score, f1_score,
            total_inferences, successful_inferences
        )
        SELECT 
            'perf_' || model_name || '_' || TO_VARCHAR(CURRENT_TIMESTAMP(), 'YYYYMMDDHH24MISS'),
            model_name,
            model_version,
            DATEADD(day, -1, CURRENT_TIMESTAMP()),
            CURRENT_TIMESTAMP(),
            
            -- Calculate metrics where ground truth is available
            AVG(CASE 
                WHEN actual_conversion IS NOT NULL 
                THEN CASE WHEN (conversion_probability > 0.5) = actual_conversion THEN 1.0 ELSE 0.0 END 
            END) as accuracy,
            
            -- Precision for conversion prediction
            SUM(CASE WHEN conversion_probability > 0.5 AND actual_conversion = TRUE THEN 1 ELSE 0 END)::DECIMAL /
            NULLIF(SUM(CASE WHEN conversion_probability > 0.5 THEN 1 ELSE 0 END), 0) as precision_score,
            
            -- Recall for conversion prediction  
            SUM(CASE WHEN conversion_probability > 0.5 AND actual_conversion = TRUE THEN 1 ELSE 0 END)::DECIMAL /
            NULLIF(SUM(CASE WHEN actual_conversion = TRUE THEN 1 ELSE 0 END), 0) as recall_score,
            
            -- F1 Score calculation
            2 * (precision_score * recall_score) / NULLIF((precision_score + recall_score), 0) as f1_score,
            
            COUNT(*) as total_inferences,
            COUNT(CASE WHEN conversion_probability IS NOT NULL THEN 1 END) as successful_inferences
            
        FROM ml_inference_logs
        WHERE model_name = model_record.model_name
          AND model_version = model_record.model_version
          AND timestamp >= DATEADD(day, -1, CURRENT_TIMESTAMP())
          AND actual_conversion IS NOT NULL;
          
    END FOR;
    
    RETURN 'Performance monitoring completed for all models';
END;
$$;

-- 11. Drift Detection Procedure
CREATE OR REPLACE PROCEDURE detect_model_drift()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    feature_drift_results OBJECT;
BEGIN
    -- Simplified drift detection - compare last 7 days vs previous 30 days
    INSERT INTO ml_drift_detection (
        drift_id, model_name, model_version, drift_type, drift_severity,
        affected_features, test_method, detection_timestamp,
        current_period_start, current_period_end,
        baseline_period_start, baseline_period_end
    )
    SELECT 
        'drift_' || model_name || '_' || TO_VARCHAR(CURRENT_TIMESTAMP(), 'YYYYMMDDHH24MISS'),
        model_name,
        model_version,
        'feature_drift',
        CASE 
            WHEN ABS(current_avg - baseline_avg) / NULLIF(baseline_stddev, 0) > 3 THEN 'critical'
            WHEN ABS(current_avg - baseline_avg) / NULLIF(baseline_stddev, 0) > 2 THEN 'high'
            WHEN ABS(current_avg - baseline_avg) / NULLIF(baseline_stddev, 0) > 1 THEN 'medium'
            ELSE 'low'
        END as drift_severity,
        ARRAY_CONSTRUCT('total_events_30d') as affected_features,
        'z_score_test',
        CURRENT_TIMESTAMP(),
        DATEADD(day, -7, CURRENT_TIMESTAMP()),
        CURRENT_TIMESTAMP(),
        DATEADD(day, -37, CURRENT_TIMESTAMP()),
        DATEADD(day, -7, CURRENT_TIMESTAMP())
        
    FROM (
        SELECT 
            model_name,
            model_version,
            -- Current period stats (last 7 days)
            AVG(CASE WHEN timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP()) 
                     THEN total_events_30d END) as current_avg,
            STDDEV(CASE WHEN timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP()) 
                        THEN total_events_30d END) as current_stddev,
            
            -- Baseline period stats (30 days before that)
            AVG(CASE WHEN timestamp BETWEEN DATEADD(day, -37, CURRENT_TIMESTAMP()) 
                                        AND DATEADD(day, -7, CURRENT_TIMESTAMP())
                     THEN total_events_30d END) as baseline_avg,
            STDDEV(CASE WHEN timestamp BETWEEN DATEADD(day, -37, CURRENT_TIMESTAMP()) 
                                           AND DATEADD(day, -7, CURRENT_TIMESTAMP())
                        THEN total_events_30d END) as baseline_stddev
                        
        FROM ml_inference_logs
        WHERE timestamp >= DATEADD(day, -37, CURRENT_TIMESTAMP())
          AND total_events_30d IS NOT NULL
        GROUP BY model_name, model_version
        HAVING COUNT(*) >= 100  -- Minimum sample size
    )
    WHERE ABS(current_avg - baseline_avg) / NULLIF(baseline_stddev, 0) > 1; -- Only flag significant drift
    
    RETURN 'Drift detection completed';
END;
$$;

-- ========================================
-- AUTOMATED TASKS
-- ========================================

-- 12. Schedule Performance Monitoring (runs every 6 hours)
CREATE OR REPLACE TASK ml_performance_monitoring_task
    WAREHOUSE = ML_WAREHOUSE
    SCHEDULE = '360 minute'  -- Every 6 hours
    COMMENT = 'Automated model performance monitoring'
AS
    CALL run_performance_monitoring();

-- 13. Schedule Drift Detection (runs daily)
CREATE OR REPLACE TASK ml_drift_detection_task
    WAREHOUSE = ML_WAREHOUSE
    SCHEDULE = '1440 minute'  -- Daily
    COMMENT = 'Automated drift detection'
AS
    CALL detect_model_drift();

-- Start the tasks
ALTER TASK ml_performance_monitoring_task RESUME;
ALTER TASK ml_drift_detection_task RESUME;

-- ========================================
-- SAMPLE QUERIES FOR TESTING
-- ========================================

-- Check current model health
SELECT * FROM model_health_dashboard;

-- View recent drift detections
SELECT * FROM drift_monitoring_dashboard 
WHERE detection_timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP());

-- Check performance trends
SELECT * FROM model_performance_trends 
WHERE evaluation_timestamp >= DATEADD(day, -14, CURRENT_TIMESTAMP());

-- Active alerts summary
SELECT * FROM alerts_summary_dashboard 
WHERE alert_date >= DATEADD(day, -7, CURRENT_DATE());

-- Show task status
SHOW TASKS LIKE '%ml_%';

SHOW TABLES LIKE '%ml_%';
