# Event Ingestion Guide - Demo vs Production

## 🎯 Overview

The Financial Services ML Pipeline supports two ingestion modes:
1. **Demo Mode**: Simulated events for demonstration (minutes)
2. **Production Mode**: Continuous real-time ingestion (24/7)

## 📊 Demo Mode - For Demonstrations

### Purpose
- Show live streaming capabilities during demos
- Generate realistic event patterns
- Demonstrate Snowpipe functionality

### How It Works
```python
# Run for 5-minute demo
from data.streaming.event_streamer import StreamingDemo

demo = StreamingDemo(use_s3=False)  # Local for demo
demo.run_demo(duration_minutes=5)   # Stops automatically
```

### What Gets Generated
- **Event Rate**: 100-1000 events per minute
- **Event Types**: Web visits, email opens, advisor calls
- **Duration**: Configurable (1-10 minutes typical)
- **Storage**: Local files or S3 (demo purposes)

### Demo Script
```bash
# Start demo streaming in background
python run_pipeline.py --step streaming

# Shows:
# ✓ Events being generated in real-time
# ✓ Snowpipe ingestion working
# ✓ Real-time feature updates
# ✓ Model predictions updating
```

---

## 🏭 Production Mode - For Live Systems

### Purpose
- Ingest actual marketing events 24/7
- Support real business operations
- Scale to millions of events per day

### Architecture
```
Marketing Systems → Message Queue → S3 → Snowpipe → Snowflake
     ↓                  ↓           ↓        ↓         ↓
• Website          • Kafka      • Auto    • Auto    • Real-time
• Email Platform   • Kinesis     Upload   Ingest   • Features
• CRM System       • SQS                            • Predictions
```

### Production Components

#### 1. Real Event Sources
```python
# Website tracking (example)
def track_website_event(client_id, page_url, session_data):
    event = {
        'event_id': str(uuid.uuid4()),
        'client_id': client_id,
        'event_timestamp': datetime.now().isoformat(),
        'event_type': 'web_visit',
        'page_url': page_url,
        'session_id': session_data['session_id'],
        'device_type': session_data['device_type']
        # ... more fields
    }
    
    # Send to message queue
    kafka_producer.send('marketing_events', event)
```

#### 2. Message Queue Processing
```python
# Kafka/Kinesis consumer (example)
def process_event_stream():
    for message in kafka_consumer:
        event = json.loads(message.value)
        
        # Validate and enrich event
        enriched_event = enrich_event(event)
        
        # Batch and upload to S3
        s3_uploader.add_to_batch(enriched_event)
```

#### 3. Automatic S3 Upload
```python
# S3 batch uploader
class S3EventUploader:
    def __init__(self, batch_size=1000, flush_interval=30):
        self.batch = []
        self.batch_size = batch_size
        
    def add_to_batch(self, event):
        self.batch.append(event)
        if len(self.batch) >= self.batch_size:
            self.flush_to_s3()
    
    def flush_to_s3(self):
        # Upload JSONL file to S3
        filename = f"events_{datetime.now().isoformat()}.jsonl"
        s3_client.put_object(
            Bucket='marketing-events-bucket',
            Key=f'year={year}/month={month}/day={day}/{filename}',
            Body=self.format_as_jsonl()
        )
```

---

## ⚡ Snowpipe Setup (Production)

### 1. S3 Configuration
```sql
-- Create external stage pointing to S3
CREATE STAGE marketing_events_stage
    URL = 's3://your-marketing-events-bucket/events/'
    CREDENTIALS = (AWS_KEY_ID = 'your_key' AWS_SECRET_KEY = 'your_secret')
    FILE_FORMAT = (TYPE = 'JSON' STRIP_OUTER_ARRAY = FALSE);
```

### 2. Snowpipe Creation
```sql
-- Create auto-ingestion pipe
CREATE PIPE marketing_events_pipe 
    AUTO_INGEST = TRUE
    AS 
    COPY INTO marketing_events(
        event_id, client_id, event_timestamp, event_type, 
        event_category, page_url, session_id, channel
    )
    FROM (
        SELECT 
            $1:event_id::STRING,
            $1:client_id::STRING,
            $1:event_timestamp::TIMESTAMP_NTZ,
            $1:event_type::STRING,
            $1:event_category::STRING,
            $1:page_url::STRING,
            $1:session_id::STRING,
            $1:channel::STRING
        FROM @marketing_events_stage
    )
    FILE_FORMAT = (TYPE = 'JSON');
```

### 3. S3 Event Notification
```json
// S3 bucket event notification configuration
{
    "Rules": [{
        "Id": "SnowpipeAutoIngest",
        "Status": "Enabled",
        "Filter": {
            "Key": {"Prefix": "events/"}
        },
        "Events": ["s3:ObjectCreated:*"],
        "Destination": {
            "SQSConfiguration": {
                "QueueArn": "arn:aws:sqs:region:account:snowpipe-queue"
            }
        }
    }]
}
```

---

## 📋 Setup Decision Matrix

| Scenario | Ingestion Mode | Duration | Setup Required |
|----------|----------------|----------|----------------|
| **Live Demo** | Simulated | 2-10 minutes | ✅ Ready to go |
| **POC/Testing** | Hybrid | Hours/Days | S3 + basic Snowpipe |
| **Production** | Real-time | 24/7 | Full architecture |

---

## 🚀 Quick Setup Instructions

### For Demo (Ready Now)
```bash
# 1. Start demo streaming
python run_pipeline.py --step streaming

# 2. Show in Snowflake
SELECT COUNT(*) FROM marketing_events 
WHERE event_timestamp >= DATEADD(minute, -5, CURRENT_TIMESTAMP());

# 3. Stop when demo done (auto-stops)
```

### For Production (Implementation)
```bash
# 1. Set up AWS S3 bucket
aws s3 mb s3://your-marketing-events-bucket

# 2. Configure Snowpipe (SQL commands above)

# 3. Deploy event collection services
# • Website tracking pixels
# • Email engagement tracking  
# • CRM integration APIs
# • Mobile app analytics

# 4. Set up monitoring
# • Event volume alerts
# • Pipeline health checks
# • Data quality validation
```

---

## 💡 Key Differences

### Demo Mode
- **Start/Stop**: Manual control for presentations
- **Volume**: Hundreds of events (manageable for demo)
- **Purpose**: Show capabilities and potential
- **Cost**: Minimal (few minutes of compute)

### Production Mode  
- **Always On**: Continuous 24/7 operation
- **Volume**: Millions of events per day
- **Purpose**: Real business operations
- **Cost**: Ongoing (but ROI-positive)

---

## 🎯 For Your Demo

**You DON'T need constant ingestion** for the demo. Here's what to do:

1. **During Demo Setup** (once):
   ```bash
   # Generate historical data (runs once)
   python run_pipeline.py --step data
   ```

2. **During Live Demo** (5 minutes):
   ```bash
   # Show real-time streaming
   python run_pipeline.py --step streaming
   ```

3. **Demo Talking Points**:
   - "This simulates your real marketing events"
   - "In production, this runs 24/7 automatically"
   - "Snowpipe handles millions of events per day"
   - "Let me show you 5 minutes of live streaming..."

The demo streaming automatically stops after the specified duration - no manual intervention needed!
