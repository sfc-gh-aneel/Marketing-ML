# Snowflake-Native Event Streaming Setup

## 🎯 How It Works

Instead of generating local JSON files, events are sent directly to **Snowflake internal stages** where **Snowpipe** automatically ingests them.

## 🏗️ Architecture

```
Event Generator → Snowflake Internal Stage → Snowpipe → marketing_events table
```

## ⚙️ Setup (Already included in snowflake_setup.sql)

```sql
-- Internal stage for events
CREATE STAGE marketing_events_stage;

-- Snowpipe for automatic ingestion
CREATE PIPE marketing_events_pipe AS
COPY INTO marketing_events
FROM @marketing_events_stage
FILE_FORMAT = (TYPE = JSON);
```

## 🎬 Demo Usage

```bash
# Run the streaming demo
python demo_streaming.py

# Events are now sent to Snowflake stage (simulated in demo mode)
# In production: Events → @marketing_events_stage → Snowpipe → table
```

## 🔍 Monitor Ingestion

```sql
-- Check pipe status
SELECT SYSTEM$PIPE_STATUS('marketing_events_pipe');

-- View recent events
SELECT * FROM marketing_events 
WHERE event_timestamp >= DATEADD(minute, -10, CURRENT_TIMESTAMP())
ORDER BY event_timestamp DESC;

-- Check stage contents
LIST @marketing_events_stage;
```

## ✨ Benefits

- **Snowflake-native**: No external file system needed
- **Auto-ingestion**: Snowpipe handles everything automatically  
- **Scalable**: Handles high-volume event streams
- **Reliable**: Built-in error handling and retry logic
- **Cost-effective**: Pay only for what you ingest

Perfect for demonstrating real-time ML pipeline capabilities! 🚀
