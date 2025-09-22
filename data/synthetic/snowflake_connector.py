"""
Snowflake Connection and Data Loading Utilities
"""

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import yaml
import logging
from typing import Dict, List, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class SnowflakeConnector:
    """Handle Snowflake connections and data operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Snowflake connection"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sf_config = self.config['snowflake']
        self.connection = None
        
    def connect(self):
        """Establish Snowflake connection"""
        try:
            self.connection = snowflake.connector.connect(
                user=self.sf_config['user'],
                password=self.sf_config['password'],
                account=self.sf_config['account'],
                warehouse=self.sf_config['warehouse'],
                database=self.sf_config['database'],
                schema=self.sf_config['schema'],
                role=self.sf_config['role']
            )
            logger.info("Successfully connected to Snowflake")
            return self.connection
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def execute_sql_file(self, sql_file_path: str):
        """Execute SQL commands from file"""
        if not self.connection:
            self.connect()
            
        with open(sql_file_path, 'r') as f:
            sql_commands = f.read()
        
        # Split by semicolon and execute each command
        commands = [cmd.strip() for cmd in sql_commands.split(';') if cmd.strip()]
        
        cursor = self.connection.cursor()
        for command in commands:
            try:
                cursor.execute(command)
                logger.info(f"Executed: {command[:50]}...")
            except Exception as e:
                logger.error(f"Failed to execute command: {e}")
                continue
        
        cursor.close()
        logger.info("Schema setup completed")
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, 
                      if_exists: str = 'replace') -> bool:
        """Load pandas DataFrame to Snowflake table"""
        if not self.connection:
            self.connect()
        
        try:
            success, nchunks, nrows, _ = write_pandas(
                conn=self.connection,
                df=df,
                table_name=table_name,
                database=self.sf_config['database'],
                schema=self.sf_config['schema'],
                auto_create_table=True,
                overwrite=(if_exists == 'replace')
            )
            
            if success:
                logger.info(f"Successfully loaded {nrows} rows to {table_name}")
                return True
            else:
                logger.error(f"Failed to load data to {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading data to {table_name}: {e}")
            return False
    
    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results and column names
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            df = pd.DataFrame(results, columns=columns)
            cursor.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def close(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")

class DataLoader:
    """Load synthetic data into Snowflake"""
    
    def __init__(self, connector: SnowflakeConnector):
        self.connector = connector
    
    def setup_database(self):
        """Set up Snowflake database and schema"""
        logger.info("Setting up Snowflake database schema...")
        self.connector.execute_sql_file("data/schema/snowflake_schema.sql")
    
    def load_synthetic_data(self, data_dir: str = "data/synthetic/output"):
        """Load all synthetic data files to Snowflake"""
        logger.info("Loading synthetic data to Snowflake...")
        
        # Load clients
        clients_df = pd.read_csv(f"{data_dir}/clients.csv")
        self.connector.load_dataframe(clients_df, "CLIENTS")
        
        # Load advisors
        advisors_df = pd.read_csv(f"{data_dir}/advisors.csv")
        self.connector.load_dataframe(advisors_df, "ADVISORS")
        
        # Load marketing events
        events_df = pd.read_csv(f"{data_dir}/marketing_events.csv")
        self.connector.load_dataframe(events_df, "MARKETING_EVENTS")
        
        logger.info("All synthetic data loaded successfully")
    
    def generate_and_load_all(self):
        """Generate synthetic data and load to Snowflake"""
        from data_generator import FinancialDataGenerator
        
        # Generate data
        generator = FinancialDataGenerator()
        clients, advisors, events = generator.generate_all_data()
        generator.save_data(clients, advisors, events)
        
        # Setup database and load data
        self.setup_database()
        self.load_synthetic_data()

if __name__ == "__main__":
    # Initialize and load data
    connector = SnowflakeConnector()
    loader = DataLoader(connector)
    
    try:
        loader.generate_and_load_all()
    finally:
        connector.close()
