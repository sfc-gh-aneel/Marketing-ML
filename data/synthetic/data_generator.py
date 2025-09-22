"""
Synthetic Data Generator for Financial Services ML Pipeline
Generates realistic client profiles and marketing event data
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
import json
import yaml
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataGenerator:
    """Generate synthetic financial services data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fake = Faker()
        Faker.seed(self.config['data_generation']['seed'])
        np.random.seed(self.config['data_generation']['seed'])
        random.seed(self.config['data_generation']['seed'])
        
        # Define categorical mappings
        self.risk_tolerance_map = {
            'Conservative': 0.7,
            'Moderate': 0.2, 
            'Aggressive': 0.1
        }
        
        self.service_tiers = ['Basic', 'Premium', 'Elite']
        self.education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'Other']
        self.occupations = [
            'Software Engineer', 'Teacher', 'Manager', 'Sales Representative',
            'Nurse', 'Accountant', 'Consultant', 'Executive', 'Analyst',
            'Engineer', 'Director', 'Specialist', 'Coordinator'
        ]
        
        # Marketing event types and probabilities
        self.event_types = {
            'web_visit': 0.35,
            'email_open': 0.25,
            'email_click': 0.15,
            'advisor_meeting': 0.05,
            'phone_call': 0.08,
            'document_download': 0.12
        }
        
        self.web_pages = [
            '/home', '/retirement-planning', '/401k-services', '/investment-options',
            '/advisor-directory', '/calculators/retirement', '/calculators/401k',
            '/education/articles', '/contact', '/about', '/pricing'
        ]
        
    def generate_client_profiles(self, num_clients: int) -> pd.DataFrame:
        """Generate synthetic client demographic and profile data"""
        logger.info(f"Generating {num_clients} client profiles...")
        
        clients = []
        for i in range(num_clients):
            # Basic demographics with realistic distributions
            age = np.random.normal(45, 12)
            age = max(22, min(70, int(age)))  # Clamp to reasonable range
            
            # Income correlation with age and education
            base_income = np.random.lognormal(10.8, 0.5)  # Median ~$50k
            
            # 401k balance based on age and income (simplified)
            years_contributing = max(0, age - 25)
            contribution_rate = np.random.uniform(0.03, 0.15)
            annual_contribution = min(22500, base_income * contribution_rate)
            growth_rate = 0.07
            balance_401k = annual_contribution * years_contributing * (1 + growth_rate) ** (years_contributing / 2)
            balance_401k *= np.random.uniform(0.7, 1.3)  # Add some variance
            
            client = {
                'client_id': str(uuid.uuid4()),
                'created_date': self.fake.date_between(start_date='-5y', end_date='today'),
                'age': age,
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.50, 0.02]),
                'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                                 p=[0.3, 0.55, 0.12, 0.03]),
                'education_level': np.random.choice(self.education_levels, 
                                                  p=[0.15, 0.35, 0.35, 0.10, 0.05]),
                'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Retired', 'Unemployed'],
                                                    p=[0.75, 0.15, 0.08, 0.02]),
                'occupation': np.random.choice(self.occupations),
                'annual_income': int(base_income),
                'state': self.fake.state_abbr(),
                'city': self.fake.city(),
                'zip_code': self.fake.zipcode(),
                'current_401k_balance': round(balance_401k, 2),
                'years_to_retirement': max(0, 65 - age),
                'risk_tolerance': np.random.choice(list(self.risk_tolerance_map.keys()),
                                                 p=list(self.risk_tolerance_map.values())),
                'investment_experience': np.random.choice(['Beginner', 'Intermediate', 'Advanced'],
                                                        p=[0.4, 0.45, 0.15]),
                'financial_goals': self._generate_financial_goals(),
                'client_tenure_months': np.random.randint(1, 60),
                'assigned_advisor_id': f"ADV_{np.random.randint(1, 50):03d}",
                'service_tier': np.random.choice(self.service_tiers, p=[0.6, 0.3, 0.1]),
                'total_assets_under_management': round(balance_401k * np.random.uniform(1.0, 3.5), 2),
                'preferred_contact_method': np.random.choice(['Email', 'Phone', 'Text', 'Mail'],
                                                           p=[0.5, 0.3, 0.15, 0.05]),
                'last_contact_date': self.fake.date_between(start_date='-3m', end_date='today'),
                'communication_frequency_preference': np.random.choice(['Weekly', 'Monthly', 'Quarterly'],
                                                                     p=[0.2, 0.6, 0.2])
            }
            clients.append(client)
            
        return pd.DataFrame(clients)
    
    def _generate_financial_goals(self) -> List[str]:
        """Generate realistic financial goals for a client"""
        possible_goals = [
            'Retirement Planning', 'Wealth Building', 'Tax Optimization',
            'Estate Planning', 'Education Funding', 'Emergency Fund',
            'Home Purchase', 'Debt Reduction', 'Investment Growth'
        ]
        num_goals = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.25, 0.05])
        return list(np.random.choice(possible_goals, size=num_goals, replace=False))
    
    def generate_advisors(self, num_advisors: int = 50) -> pd.DataFrame:
        """Generate advisor profiles"""
        advisors = []
        specializations = [
            'Retirement Planning', 'Investment Management', 'Estate Planning',
            'Tax Planning', 'Financial Planning', 'Wealth Management'
        ]
        
        for i in range(num_advisors):
            advisor = {
                'advisor_id': f"ADV_{i+1:03d}",
                'advisor_name': self.fake.name(),
                'specialization': np.random.choice(specializations),
                'years_experience': np.random.randint(2, 25),
                'client_count': np.random.randint(20, 150),
                'avg_client_satisfaction': round(np.random.uniform(3.5, 5.0), 2)
            }
            advisors.append(advisor)
            
        return pd.DataFrame(advisors)
    
    def generate_marketing_events(self, clients_df: pd.DataFrame, 
                                events_per_client: int = 150) -> pd.DataFrame:
        """Generate realistic marketing event data"""
        logger.info(f"Generating marketing events for {len(clients_df)} clients...")
        
        events = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['data_generation']['date_range_days'])
        
        for _, client in clients_df.iterrows():
            client_events = self._generate_client_events(
                client, events_per_client, start_date, end_date
            )
            events.extend(client_events)
            
        return pd.DataFrame(events)
    
    def _generate_client_events(self, client: pd.Series, num_events: int,
                              start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate events for a single client with realistic patterns"""
        events = []
        
        # Adjust event frequency based on client characteristics
        engagement_multiplier = self._calculate_engagement_multiplier(client)
        actual_events = int(num_events * engagement_multiplier)
        
        for _ in range(actual_events):
            event_type = np.random.choice(
                list(self.event_types.keys()),
                p=list(self.event_types.values())
            )
            
            event = self._create_event(client, event_type, start_date, end_date)
            events.append(event)
            
        return events
    
    def _calculate_engagement_multiplier(self, client: pd.Series) -> float:
        """Calculate engagement multiplier based on client characteristics"""
        multiplier = 1.0
        
        # Age factor (middle-aged more engaged)
        if 35 <= client['age'] <= 55:
            multiplier *= 1.2
        elif client['age'] > 60:
            multiplier *= 0.8
            
        # Income factor
        if client['annual_income'] > 100000:
            multiplier *= 1.3
        elif client['annual_income'] < 40000:
            multiplier *= 0.7
            
        # Service tier factor
        tier_multipliers = {'Basic': 0.8, 'Premium': 1.2, 'Elite': 1.5}
        multiplier *= tier_multipliers[client['service_tier']]
        
        return max(0.3, min(2.0, multiplier))  # Clamp between 0.3 and 2.0
    
    def _create_event(self, client: pd.Series, event_type: str,
                     start_date: datetime, end_date: datetime) -> Dict:
        """Create a single marketing event"""
        event_timestamp = self.fake.date_time_between(start_date=start_date, end_date=end_date)
        
        base_event = {
            'event_id': str(uuid.uuid4()),
            'client_id': client['client_id'],
            'event_timestamp': event_timestamp,
            'event_type': event_type,
            'event_category': self._get_event_category(event_type),
            'channel': self._get_channel(event_type),
            'touchpoint_value': round(np.random.uniform(0.1, 1.0), 4),
            'conversion_flag': np.random.choice([True, False], p=[0.05, 0.95])
        }
        
        # Add event-specific fields
        if event_type == 'web_visit':
            base_event.update(self._create_web_event())
        elif event_type in ['email_open', 'email_click']:
            base_event.update(self._create_email_event(event_type))
        elif event_type in ['advisor_meeting', 'phone_call']:
            base_event.update(self._create_advisor_event(event_type, client))
        elif event_type == 'document_download':
            base_event.update(self._create_document_event())
            
        return base_event
    
    def _get_event_category(self, event_type: str) -> str:
        """Get event category for event type"""
        categories = {
            'web_visit': 'Digital',
            'email_open': 'Email',
            'email_click': 'Email',
            'advisor_meeting': 'Personal',
            'phone_call': 'Personal',
            'document_download': 'Digital'
        }
        return categories.get(event_type, 'Other')
    
    def _get_channel(self, event_type: str) -> str:
        """Get channel for event type"""
        channels = {
            'web_visit': 'Website',
            'email_open': 'Email',
            'email_click': 'Email',
            'advisor_meeting': 'In-Person',
            'phone_call': 'Phone',
            'document_download': 'Website'
        }
        return channels.get(event_type, 'Other')
    
    def _create_web_event(self) -> Dict:
        """Create web-specific event fields"""
        return {
            'page_url': np.random.choice(self.web_pages),
            'session_id': str(uuid.uuid4())[:8],
            'time_on_page': np.random.randint(15, 600),  # 15 seconds to 10 minutes
            'referrer_source': np.random.choice(['Direct', 'Google', 'Email', 'Social'], 
                                              p=[0.4, 0.35, 0.15, 0.1]),
            'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.6, 0.35, 0.05])
        }
    
    def _create_email_event(self, event_type: str) -> Dict:
        """Create email-specific event fields"""
        campaigns = [
            'Retirement Planning Tips', 'Market Update', 'New Investment Options',
            'Advisor Spotlight', 'Educational Webinar', '401k Reminder'
        ]
        
        return {
            'campaign_id': f"CAMP_{np.random.randint(1000, 9999)}",
            'email_subject': np.random.choice(campaigns),
            'content_type': np.random.choice(['Newsletter', 'Promotional', 'Educational', 'Update'])
        }
    
    def _create_advisor_event(self, event_type: str, client: pd.Series) -> Dict:
        """Create advisor interaction event fields"""
        meeting_types = ['Initial Consultation', 'Annual Review', 'Investment Review', 
                        'Planning Session', 'Follow-up', 'Emergency Consultation']
        outcomes = ['Scheduled Follow-up', 'Action Items Assigned', 'Plan Updated', 
                   'No Action Needed', 'Referral Made']
        
        return {
            'meeting_type': np.random.choice(meeting_types),
            'duration_minutes': np.random.randint(15, 120),
            'advisor_id': client['assigned_advisor_id'],
            'meeting_outcome': np.random.choice(outcomes)
        }
    
    def _create_document_event(self) -> Dict:
        """Create document download event fields"""
        products = ['401k Guide', 'Investment Overview', 'Retirement Calculator', 
                   'Fee Schedule', 'Market Report', 'Educational Materials']
        
        return {
            'product_category': 'Educational',
            'product_specific': np.random.choice(products),
            'action_taken': 'Downloaded'
        }
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all synthetic data"""
        num_clients = self.config['data_generation']['num_clients']
        events_per_client = self.config['data_generation']['num_events_per_client']
        
        # Generate data
        clients_df = self.generate_client_profiles(num_clients)
        advisors_df = self.generate_advisors()
        events_df = self.generate_marketing_events(clients_df, events_per_client)
        
        logger.info(f"Generated {len(clients_df)} clients, {len(advisors_df)} advisors, "
                   f"{len(events_df)} events")
        
        return clients_df, advisors_df, events_df
    
    def save_data(self, clients_df: pd.DataFrame, advisors_df: pd.DataFrame, 
                  events_df: pd.DataFrame, output_dir: str = "data/synthetic/output"):
        """Save generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        clients_df.to_csv(f"{output_dir}/clients.csv", index=False)
        advisors_df.to_csv(f"{output_dir}/advisors.csv", index=False)
        events_df.to_csv(f"{output_dir}/marketing_events.csv", index=False)
        
        logger.info(f"Data saved to {output_dir}")

if __name__ == "__main__":
    generator = FinancialDataGenerator()
    clients, advisors, events = generator.generate_all_data()
    generator.save_data(clients, advisors, events)
