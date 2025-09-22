#!/usr/bin/env python3
"""
Demo Streaming Script for Financial Services ML Pipeline
Run this during live demos to show real-time event ingestion
"""

import sys
import time
import json
import random
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.streaming.event_streamer import StreamingDemo

class DemoController:
    """Control streaming demo for presentations"""
    
    def __init__(self):
        self.demo = StreamingDemo(use_snowflake_stage=True)  # Snowflake-native demo
        self.is_running = False
        self.total_events = 0
        
    def start_demo(self, duration_minutes=5):
        """Start demo streaming with status updates"""
        print(f"🚀 Starting {duration_minutes}-minute streaming demo...")
        print("=" * 60)
        
        self.is_running = True
        self.start_time = time.time()
        self.end_time = self.start_time + (duration_minutes * 60)
        
        # Start event generation in background
        demo_thread = threading.Thread(target=self._run_demo_background, args=(duration_minutes,))
        demo_thread.daemon = True
        demo_thread.start()
        
        # Show live status updates
        self._show_live_status()
        
        return self.total_events
    
    def _run_demo_background(self, duration_minutes):
        """Run the actual demo in background"""
        self.demo.run_demo(duration_minutes=duration_minutes)
    
    def _show_live_status(self):
        """Show live streaming status during demo"""
        while self.is_running and time.time() < self.end_time:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.end_time - time.time())
            
            # Get events from demo queue
            events = self.demo.event_streamer.get_events_batch()
            if events:
                self.total_events += len(events)
                
                # Show sample event
                if len(events) > 0:
                    sample_event = events[0]
                    print(f"⚡ {datetime.now().strftime('%H:%M:%S')} | "
                          f"Events: +{len(events):3d} (Total: {self.total_events:,}) | "
                          f"Sample: {sample_event.get('event_type', 'unknown')} | "
                          f"Client: {sample_event.get('client_id', 'unknown')[:12]}...")
            
            # Progress indicator
            progress = min(100, (elapsed / (self.end_time - self.start_time)) * 100)
            bars = int(progress / 5)
            progress_bar = "█" * bars + "░" * (20 - bars)
            
            print(f"📊 Progress: [{progress_bar}] {progress:.1f}% | "
                  f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")
            
            time.sleep(2)  # Update every 2 seconds
        
        self.is_running = False
        print("\n✅ Demo streaming completed!")
        print(f"📈 Total events generated: {self.total_events:,}")

def demo_quick_start():
    """Quick 2-minute demo for fast presentations"""
    print("🎬 QUICK DEMO MODE (2 minutes)")
    controller = DemoController()
    controller.start_demo(duration_minutes=2)

def demo_full_presentation():
    """Full 5-minute demo for detailed presentations"""
    print("🎬 FULL PRESENTATION MODE (5 minutes)")
    controller = DemoController()
    controller.start_demo(duration_minutes=5)

def demo_interactive():
    """Interactive demo where user controls duration"""
    print("🎬 INTERACTIVE DEMO MODE")
    print("How many minutes should the streaming demo run?")
    try:
        duration = int(input("Enter duration (1-10 minutes): "))
        duration = max(1, min(10, duration))  # Clamp between 1-10
        
        controller = DemoController()
        controller.start_demo(duration_minutes=duration)
    except ValueError:
        print("Invalid input. Running 3-minute demo.")
        controller = DemoController()
        controller.start_demo(duration_minutes=3)

def show_demo_menu():
    """Show demo options menu"""
    print("\n" + "="*60)
    print("🎯 FINANCIAL SERVICES ML PIPELINE - STREAMING DEMO")
    print("="*60)
    print("Choose your demo mode:")
    print("1. Quick Demo (2 minutes) - For time-constrained presentations")
    print("2. Full Demo (5 minutes) - For detailed demonstrations") 
    print("3. Interactive Demo - Choose your own duration")
    print("4. Exit")
    print("-"*60)

def main():
    """Main demo controller"""
    
    # Show header
    print("\n🏔️  SNOWFLAKE FINANCIAL SERVICES ML PIPELINE")
    print("📊 Real-time Event Streaming Demonstration")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        show_demo_menu()
        
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                demo_quick_start()
            elif choice == '2':
                demo_full_presentation()  
            elif choice == '3':
                demo_interactive()
            elif choice == '4':
                print("👋 Demo session ended. Thank you!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            
        # Ask if they want to run another demo
        if choice in ['1', '2', '3']:
            print("\n" + "="*60)
            repeat = input("Run another demo? (y/n): ").strip().lower()
            if repeat != 'y':
                print("👋 Demo session completed!")
                break

if __name__ == "__main__":
    main()
