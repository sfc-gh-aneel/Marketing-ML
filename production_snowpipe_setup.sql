-- Production Snowpipe Setup for Financial Services ML Pipeline
-- Use this for real-world deployment with continuous event ingestion

-- ========================================
-- PRODUCTION SNOWPIPE CONFIGURATION
-- ========================================

-- 1. Create external stage for S3 event files
CREATE OR REPLACE STAGE marketing_events_s3_stage
    URL = 's3://your-marketing-events-bucket/events/'
    CREDENTIALS = (
        AWS_KEY_ID = 'your_aws_access_key_id' 
        AWS_SECRET_KEY = 'your_aws_secret_access_key'
    )
    FILE_FORMAT = (
        TYPE = 'JSON'
        STRIP_OUTER_ARRAY = FALSE
        DATE_FORMAT = 'AUTO'
        TIMESTAMP_FORMAT = 'AUTO'
    )
    COMMENT = 'Production stage for marketing events from S3';

-- 2. Test stage access (run this to verify S3 connection)
LIST @marketing_events_s3_stage;

-- 3. Create production Snowpipe for automatic ingestion
CREATE OR REPLACE PIPE marketing_events_production_pipe 
    AUTO_INGEST = TRUE
    COMMENT = 'Production pipe for real-time marketing event ingestion'
    AS 
    COPY INTO marketing_events (
        event_id,
        client_id, 
        event_timestamp,
        event_type,
        event_category,
        page_url,
        session_id,
        time_on_page,
        referrer_source,
        device_type,
        campaign_id,
        email_subject,
        content_type,
        meeting_type,
        duration_minutes,
        advisor_id,
        meeting_outcome,
        product_category,
        product_specific,
        action_taken,
        channel,
        touchpoint_value,
        conversion_flag
    )
    FROM (
        SELECT 
            $1:event_id::STRING as event_id,
            $1:client_id::STRING as client_id,
            $1:event_timestamp::TIMESTAMP_NTZ as event_timestamp,
            $1:event_type::STRING as event_type,
            $1:event_category::STRING as event_category,
            $1:page_url::STRING as page_url,
            $1:session_id::STRING as session_id,
            $1:time_on_page::INTEGER as time_on_page,
            $1:referrer_source::STRING as referrer_source,
            $1:device_type::STRING as device_type,
            $1:campaign_id::STRING as campaign_id,
            $1:email_subject::STRING as email_subject,
            $1:content_type::STRING as content_type,
            $1:meeting_type::STRING as meeting_type,
            $1:duration_minutes::INTEGER as duration_minutes,
            $1:advisor_id::STRING as advisor_id,
            $1:meeting_outcome::STRING as meeting_outcome,
            $1:product_category::STRING as product_category,
            $1:product_specific::STRING as product_specific,
            $1:action_taken::STRING as action_taken,
            $1:channel::STRING as channel,
            $1:touchpoint_value::DECIMAL(10,4) as touchpoint_value,
            $1:conversion_flag::BOOLEAN as conversion_flag
        FROM @marketing_events_s3_stage
    )
    FILE_FORMAT = (TYPE = 'JSON')
    ON_ERROR = 'CONTINUE';  -- Skip malformed records in production

-- 4. Get SQS queue ARN for S3 notifications
SHOW PIPES LIKE 'marketing_events_production_pipe';
-- Copy the notification_channel value to configure S3 event notifications

-- 5. Create monitoring table for pipe status
CREATE OR REPLACE TABLE snowpipe_monitoring (
    pipe_name STRING,
    check_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    files_queued INTEGER,
    files_processed INTEGER,
    bytes_processed INTEGER,
    last_load_time TIMESTAMP_NTZ,
    error_count INTEGER,
    status STRING
);

-- 6. Create monitoring procedure
CREATE OR REPLACE PROCEDURE monitor_snowpipe_health()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Insert current pipe status
    INSERT INTO snowpipe_monitoring (
        pipe_name, files_queued, files_processed, 
        bytes_processed, last_load_time, error_count, status
    )
    SELECT 
        'marketing_events_production_pipe',
        PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):pendingFileCount::INTEGER,
        PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):executedFileCount::INTEGER,
        PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):queuedBytes::INTEGER,
        PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):lastReceivedMessageTimestamp::TIMESTAMP_NTZ,
        PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):errorsSeen::INTEGER,
        CASE 
            WHEN PARSE_JSON(SYSTEM$PIPE_STATUS('marketing_events_production_pipe')):executionState::STRING = 'RUNNING' 
            THEN 'HEALTHY'
            ELSE 'ATTENTION_NEEDED'
        END;
    
    RETURN 'Monitoring data updated';
END;
$$;

-- 7. Schedule monitoring task (runs every 15 minutes)
CREATE OR REPLACE TASK monitor_snowpipe_task
    WAREHOUSE = ML_WAREHOUSE
    SCHEDULE = '15 minute'
    COMMENT = 'Monitor Snowpipe health every 15 minutes'
AS
    CALL monitor_snowpipe_health();

-- Start the monitoring task
ALTER TASK monitor_snowpipe_task RESUME;

-- ========================================
-- REAL-TIME FEATURE UPDATES
-- ========================================

-- 8. Create task for real-time feature refresh
CREATE OR REPLACE TASK realtime_feature_refresh
    WAREHOUSE = ML_WAREHOUSE
    SCHEDULE = '5 minute'  -- Update features every 5 minutes
    COMMENT = 'Refresh features with new events'
AS
    -- Refresh engagement features with new data
    MERGE INTO feature_store fs
    USING (
        SELECT 
            client_id,
            CURRENT_TIMESTAMP() as feature_timestamp,
            COUNT(*) as total_events_30d,
            COUNT(CASE WHEN event_type = 'web_visit' THEN 1 END) as web_visits_30d,
            COUNT(CASE WHEN event_type = 'email_open' THEN 1 END) as email_opens_30d,
            COUNT(CASE WHEN event_type = 'email_click' THEN 1 END) as email_clicks_30d,
            AVG(time_on_page) as avg_session_duration_30d,
            MAX(event_timestamp) as last_activity_date,
            DATEDIFF(day, MAX(event_timestamp), CURRENT_TIMESTAMP()) as days_since_last_activity
        FROM marketing_events 
        WHERE event_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())
        GROUP BY client_id
    ) new_features
    ON fs.client_id = new_features.client_id
    WHEN MATCHED THEN UPDATE SET
        fs.feature_timestamp = new_features.feature_timestamp,
        fs.web_sessions_30d = new_features.total_events_30d,
        fs.web_sessions_30d = new_features.web_visits_30d,
        fs.email_opens_30d = new_features.email_opens_30d,
        fs.avg_session_duration_30d = new_features.avg_session_duration_30d,
        fs.days_since_last_activity = new_features.days_since_last_activity
    WHEN NOT MATCHED THEN INSERT (
        client_id, feature_timestamp, web_sessions_30d, 
        email_opens_30d, avg_session_duration_30d, days_since_last_activity
    ) VALUES (
        new_features.client_id, new_features.feature_timestamp,
        new_features.web_visits_30d, new_features.email_opens_30d,
        new_features.avg_session_duration_30d, new_features.days_since_last_activity
    );

-- Start the feature refresh task
ALTER TASK realtime_feature_refresh RESUME;

-- ========================================
-- PRODUCTION MONITORING VIEWS
-- ========================================

-- 9. Create real-time dashboard views
CREATE OR REPLACE VIEW realtime_event_dashboard AS
SELECT 
    DATE_TRUNC('hour', event_timestamp) as event_hour,
    event_type,
    channel,
    COUNT(*) as event_count,
    COUNT(DISTINCT client_id) as unique_clients,
    AVG(touchpoint_value) as avg_touchpoint_value,
    SUM(CASE WHEN conversion_flag THEN 1 ELSE 0 END) as conversions
FROM marketing_events 
WHERE event_timestamp >= DATEADD(day, -1, CURRENT_TIMESTAMP())
GROUP BY event_hour, event_type, channel
ORDER BY event_hour DESC;

-- 10. Data quality monitoring
CREATE OR REPLACE VIEW data_quality_dashboard AS
SELECT 
    DATE(event_timestamp) as event_date,
    COUNT(*) as total_events,
    COUNT(CASE WHEN client_id IS NULL THEN 1 END) as missing_client_id,
    COUNT(CASE WHEN event_type IS NULL THEN 1 END) as missing_event_type,
    COUNT(CASE WHEN event_timestamp IS NULL THEN 1 END) as missing_timestamp,
    MIN(event_timestamp) as first_event,
    MAX(event_timestamp) as last_event,
    
    -- Data freshness (should be < 5 minutes for real-time)
    DATEDIFF(minute, MAX(event_timestamp), CURRENT_TIMESTAMP()) as minutes_since_last_event,
    
    -- Data quality score
    ROUND(
        (COUNT(*) - COUNT(CASE WHEN client_id IS NULL OR event_type IS NULL THEN 1 END)) 
        * 100.0 / COUNT(*), 2
    ) as data_quality_score
FROM marketing_events 
WHERE event_timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY event_date
ORDER BY event_date DESC;

-- ========================================
-- ALERTS AND NOTIFICATIONS
-- ========================================

-- 11. Create alert procedure for data issues
CREATE OR REPLACE PROCEDURE check_data_alerts()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    minutes_since_last_event INTEGER;
    data_quality_score DECIMAL(5,2);
    alert_message STRING DEFAULT '';
BEGIN
    -- Check data freshness
    SELECT DATEDIFF(minute, MAX(event_timestamp), CURRENT_TIMESTAMP())
    INTO minutes_since_last_event
    FROM marketing_events;
    
    IF (minutes_since_last_event > 10) THEN
        alert_message := alert_message || 'ALERT: No events received in ' || minutes_since_last_event || ' minutes. ';
    END IF;
    
    -- Check data quality
    SELECT 
        ROUND((COUNT(*) - COUNT(CASE WHEN client_id IS NULL OR event_type IS NULL THEN 1 END)) * 100.0 / COUNT(*), 2)
    INTO data_quality_score
    FROM marketing_events 
    WHERE event_timestamp >= DATEADD(hour, -1, CURRENT_TIMESTAMP());
    
    IF (data_quality_score < 95) THEN
        alert_message := alert_message || 'ALERT: Data quality score is ' || data_quality_score || '%. ';
    END IF;
    
    IF (alert_message = '') THEN
        RETURN 'All systems healthy';
    ELSE
        -- In production, integrate with alerting system (email, Slack, PagerDuty)
        RETURN alert_message;
    END IF;
END;
$$;

-- 12. Schedule alert checking task
CREATE OR REPLACE TASK data_quality_alerts
    WAREHOUSE = ML_WAREHOUSE
    SCHEDULE = '5 minute'
    COMMENT = 'Check for data quality issues every 5 minutes'
AS
    CALL check_data_alerts();

-- Start the alert task
ALTER TASK data_quality_alerts RESUME;

-- ========================================
-- VERIFICATION QUERIES
-- ========================================

-- Check pipe status
SELECT SYSTEM$PIPE_STATUS('marketing_events_production_pipe');

-- View recent events
SELECT * FROM marketing_events 
WHERE event_timestamp >= DATEADD(hour, -1, CURRENT_TIMESTAMP())
ORDER BY event_timestamp DESC
LIMIT 10;

-- Check monitoring data
SELECT * FROM snowpipe_monitoring 
ORDER BY check_timestamp DESC 
LIMIT 5;

-- View real-time dashboard
SELECT * FROM realtime_event_dashboard 
WHERE event_hour >= DATEADD(hour, -6, CURRENT_TIMESTAMP());

-- Check data quality
SELECT * FROM data_quality_dashboard 
WHERE event_date >= DATEADD(day, -3, CURRENT_DATE());

SHOW TASKS;
SHOW PIPES;
