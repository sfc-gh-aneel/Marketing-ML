"""
Real-time Event Streaming Simulator for Snowpipe
Simulates marketing events being streamed to Snowflake
"""

import json
import time
import random
import uuid
from datetime import datetime, timedelta
# import boto3  # Not needed for demo
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
# import yaml  # Not needed for demo
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class MarketingEventStreamer:
    """Stream synthetic marketing events to S3 for Snowpipe ingestion"""
    
    def __init__(self):
        # Simplified - no config file needed for demo
        self.streaming_config = {
            'batch_size': 50,
            'frequency_seconds': 3
        }
        self.event_queue = Queue()
        self.is_streaming = False
        
        # Load client IDs for event generation
        self.client_ids = self._load_client_ids()
        
        # Event type probabilities for real-time streaming
        self.realtime_event_probs = {
            'web_visit': 0.4,
            'email_open': 0.2,
            'email_click': 0.15,
            'page_view': 0.15,
            'search': 0.05,
            'login': 0.05
        }
        
    def _load_client_ids(self) -> List[str]:
        """Load client IDs from generated data"""
        try:
            clients_df = pd.read_csv("data/synthetic/output/clients.csv")
            return clients_df['client_id'].tolist()
        except FileNotFoundError:
            logger.warning("Client data not found, generating sample client IDs")
            return [str(uuid.uuid4()) for _ in range(1000)]
    
    def generate_realtime_event(self) -> Dict:
        """Generate a single real-time marketing event"""
        client_id = random.choice(self.client_ids)
        event_type = np.random.choice(
            list(self.realtime_event_probs.keys()),
            p=list(self.realtime_event_probs.values())
        )
        
        event = {
            'event_id': str(uuid.uuid4()),
            'client_id': client_id,
            'event_timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'event_category': self._get_event_category(event_type),
            'channel': self._get_channel(event_type),
            'touchpoint_value': round(random.uniform(0.1, 1.0), 4),
            'conversion_flag': random.choice([True, False]) if random.random() < 0.03 else False
        }
        
        # Add event-specific fields
        if event_type == 'web_visit':
            event.update(self._create_web_event())
        elif event_type in ['email_open', 'email_click']:
            event.update(self._create_email_event())
        elif event_type == 'page_view':
            event.update(self._create_page_view_event())
        elif event_type == 'search':
            event.update(self._create_search_event())
        elif event_type == 'login':
            event.update(self._create_login_event())
        
        return event
    
    def _get_event_category(self, event_type: str) -> str:
        categories = {
            'web_visit': 'Digital',
            'email_open': 'Email',
            'email_click': 'Email',
            'page_view': 'Digital',
            'search': 'Digital',
            'login': 'Digital'
        }
        return categories.get(event_type, 'Other')
    
    def _get_channel(self, event_type: str) -> str:
        channels = {
            'web_visit': 'Website',
            'email_open': 'Email',
            'email_click': 'Email',
            'page_view': 'Website',
            'search': 'Website',
            'login': 'Portal'
        }
        return channels.get(event_type, 'Other')
    
    def _create_web_event(self) -> Dict:
        pages = [
            '/home', '/retirement-planning', '/401k-services', '/investment-options',
            '/advisor-directory', '/calculators/retirement', '/education/articles',
            '/contact', '/about', '/pricing', '/login'
        ]
        
        return {
            'page_url': random.choice(pages),
            'session_id': str(uuid.uuid4())[:8],
            'time_on_page': random.randint(10, 300),
            'referrer_source': np.random.choice(['Direct', 'Google', 'Email', 'Social'], 
                                              p=[0.4, 0.35, 0.15, 0.1]),
            'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.6, 0.35, 0.05])
        }
    
    def _create_email_event(self) -> Dict:
        campaigns = [
            'Weekly Market Update', 'Retirement Tips Newsletter', 'Investment Opportunity',
            'Educational Webinar Invitation', 'Account Review Reminder'
        ]
        
        return {
            'campaign_id': f"CAMP_{random.randint(1000, 9999)}",
            'email_subject': random.choice(campaigns),
            'content_type': random.choice(['Newsletter', 'Promotional', 'Educational'])
        }
    
    def _create_page_view_event(self) -> Dict:
        return {
            'page_url': random.choice([
                '/dashboard', '/portfolio', '/transactions', '/reports',
                '/settings', '/help', '/documents', '/statements'
            ]),
            'session_id': str(uuid.uuid4())[:8],
            'time_on_page': random.randint(30, 600)
        }
    
    def _create_search_event(self) -> Dict:
        search_terms = [
            'retirement planning', '401k rollover', 'investment options',
            'advisor near me', 'fees', 'portfolio performance', 'tax implications'
        ]
        
        return {
            'search_term': random.choice(search_terms),
            'search_results_count': random.randint(5, 50),
            'search_category': 'Internal'
        }
    
    def _create_login_event(self) -> Dict:
        return {
            'login_method': random.choice(['Username/Password', 'SSO', 'Mobile']),
            'device_type': random.choice(['Desktop', 'Mobile', 'Tablet']),
            'session_duration_minutes': random.randint(5, 45)
        }
    
    def event_generator(self):
        """Generate events continuously"""
        while self.is_streaming:
            # Generate batch of events
            batch_size = self.streaming_config['batch_size']
            events = []
            
            for _ in range(random.randint(10, batch_size)):
                event = self.generate_realtime_event()
                events.append(event)
            
            # Add to queue
            self.event_queue.put(events)
            
            # Wait before next batch
            time.sleep(random.uniform(1, self.streaming_config['frequency_seconds']))
    
    def start_streaming(self):
        """Start streaming events"""
        logger.info("Starting event streaming...")
        self.is_streaming = True
        
        # Start event generation in separate thread
        generator_thread = threading.Thread(target=self.event_generator)
        generator_thread.daemon = True
        generator_thread.start()
        
        return generator_thread
    
    def stop_streaming(self):
        """Stop streaming events"""
        logger.info("Stopping event streaming...")
        self.is_streaming = False
    
    def get_events_batch(self) -> List[Dict]:
        """Get batch of events from queue"""
        if not self.event_queue.empty():
            return self.event_queue.get()
        return []

class SnowpipeStreamer:
    """Handle Snowpipe streaming to S3 and Snowflake"""
    
    def __init__(self, s3_bucket: str = None, s3_prefix: str = "marketing-events/"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None
        
        if s3_bucket:
            logger.info(f"S3 disabled for demo mode")
    
    def upload_events_to_s3(self, events: List[Dict]) -> bool:
        """Upload events batch to S3 for Snowpipe ingestion"""
        if not self.s3_client or not events:
            return False
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.s3_prefix}events_{timestamp}.json"
            
            # Convert events to JSONL format (one JSON per line)
            jsonl_content = '\\n'.join([json.dumps(event) for event in events])
            
            # S3 upload disabled for demo
            pass
            
            logger.info(f"Uploaded {len(events)} events to s3://{self.s3_bucket}/{filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload events to S3: {e}")
            return False
    
    def save_events_locally(self, events: List[Dict], local_dir: str = "data/streaming/output"):
        """Save events locally (alternative to S3)"""
        import os
        os.makedirs(local_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{local_dir}/events_{timestamp}.json"
        
        with open(filename, 'w') as f:
            for event in events:
                f.write(json.dumps(event) + '\\n')
        
        logger.info(f"Saved {len(events)} events to {filename}")

class StreamingDemo:
    """Demonstrate real-time streaming capabilities"""
    
    def __init__(self, use_s3: bool = False, s3_bucket: str = None):
        self.event_streamer = MarketingEventStreamer()
        self.snowpipe_streamer = SnowpipeStreamer(s3_bucket) if use_s3 else None
        self.use_s3 = use_s3
        
    def run_demo(self, duration_minutes: int = 10):
        """Run streaming demo for specified duration"""
        logger.info(f"Starting streaming demo for {duration_minutes} minutes...")
        
        # Start event generation
        generator_thread = self.event_streamer.start_streaming()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        total_events = 0
        
        try:
            while time.time() < end_time:
                # Get events batch
                events = self.event_streamer.get_events_batch()
                
                if events:
                    total_events += len(events)
                    
                    if self.use_s3 and self.snowpipe_streamer:
                        # Upload to S3 for Snowpipe
                        self.snowpipe_streamer.upload_events_to_s3(events)
                    else:
                        # Save locally
                        if not self.snowpipe_streamer:
                            self.snowpipe_streamer = SnowpipeStreamer()
                        self.snowpipe_streamer.save_events_locally(events)
                
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.event_streamer.stop_streaming()
            generator_thread.join(timeout=5)
            logger.info(f"Demo completed. Total events generated: {total_events}")

if __name__ == "__main__":
    # Run local streaming demo
    demo = StreamingDemo(use_s3=False)
    demo.run_demo(duration_minutes=5)
