# Live Event Streaming for Demos

## 🎯 Quick Demo Commands

### During Your Demo:
```bash
# Start the demo streaming
python demo_streaming.py

# Choose your demo length:
# 1. Quick Demo (2 minutes) - For time-constrained presentations
# 2. Full Demo (5 minutes) - For detailed demonstrations  
# 3. Interactive Demo - Choose your own duration (1-10 minutes)
```

## 📊 What It Shows

The streaming demo generates realistic financial services events in real-time:

### Event Types Generated:
- **Web Visits**: Client browsing retirement planning pages
- **Email Engagement**: Opens/clicks on financial newsletters
- **Page Views**: Portal activity and document access
- **Search Activity**: Clients searching for investment options
- **Login Events**: Portal authentication and session data

### Live Demo Display:
```
⚡ 15:52:51 | Events: + 15 (Total: 1,247) | Sample: web_visit | Client: client_12345...
📊 Progress: [████████████░░░░░░░░] 60.0% | Elapsed: 180s | Remaining: 120s
```

## 🎬 Demo Flow Recommendation

1. **Start the pipeline demo** with Snowflake notebooks
2. **When you get to real-time data ingestion**, run:
   ```bash
   python demo_streaming.py
   ```
3. **Choose option 2** (5-minute full demo) for detailed presentations
4. **Let it run in background** while you continue explaining other features
5. **Show the generated event files** in `data/streaming/output/`

## 📁 Generated Files

Events are saved locally to:
- `data/streaming/output/events_YYYYMMDD_HHMMSS.json`
- Shows realistic event streams that would feed into Snowpipe
- Perfect for demonstrating real-time capabilities without actual S3/cloud setup

## ⚡ Pro Tips for Demos

- **Run during feature engineering section** to show live data flowing
- **Point out the event variety** - web, email, search, login events
- **Mention this feeds Snowpipe** in production environments
- **Show how ML models score these events in real-time**

Perfect for showing live data ingestion without complex cloud setup!
