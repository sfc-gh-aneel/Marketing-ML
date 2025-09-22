"""
Real-time ML Inference using Snowpark Container Services
USE ONLY when millisecond response times are required
(advisor calls, live website interactions)
"""

import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
# import yaml  # Not needed for demo
import json
from datetime import datetime

class RealtimeMLService:
    """Real-time ML service for high-frequency, low-latency inference"""
    
    def __init__(self):
        # Simplified - no config file needed
        self.config = {
            'snowflake': {
                'account': 'your_account',
                'user': 'your_user',
                'password': 'your_password', 
                'role': 'ML_ROLE',
                'warehouse': 'ML_WAREHOUSE',
                'database': 'FINANCIAL_ML',
                'schema': 'PUBLIC'
            }
        }
        
        self.session = None
    
    def setup_container_service(self, model_name: str = "CONVERSION_PREDICTOR"):
        """
        Set up Snowpark Container Service for real-time inference
        Only use when you need <100ms response times
        """
        session = self.create_session()
        
        print(f"🚀 Setting up real-time container service for {model_name}")
        print("⚠️  Use this ONLY when batch inference isn't fast enough")
        
        # 1. Create compute pool for real-time inference
        compute_pool_sql = """
        CREATE COMPUTE POOL IF NOT EXISTS ml_realtime_pool
        MIN_NODES = 1
        MAX_NODES = 3
        INSTANCE_FAMILY = CPU_X64_XS
        AUTO_RESUME = TRUE
        INITIALLY_SUSPENDED = FALSE
        """
        
        try:
            session.sql(compute_pool_sql).collect()
            print("✅ Compute pool created for real-time inference")
        except Exception as e:
            print(f"⚠️  Compute pool creation: {e}")
        
        # 2. Create the container service specification
        service_spec = self.create_service_specification(model_name)
        
        # 3. Deploy the service
        service_sql = f"""
        CREATE SERVICE IF NOT EXISTS {model_name.lower()}_realtime
        IN COMPUTE POOL ml_realtime_pool
        FROM SPECIFICATION $$
        {json.dumps(service_spec, indent=2)}
        $$
        """
        
        try:
            session.sql(service_sql).collect()
            print("✅ Real-time service deployed")
        except Exception as e:
            print(f"⚠️  Service deployment: {e}")
            return False
        
        # 4. Create endpoint function
        self.create_realtime_endpoint(session, model_name)
        
        # 5. Set up monitoring
        self.setup_realtime_monitoring(session, model_name)
        
        return True
    
    def create_service_specification(self, model_name: str) -> dict:
        """Create the container service specification"""
        
        return {
            "spec": {
                "containers": [
                    {
                        "name": "ml-inference-api",
                        "image": "/financial_ml/inference_api:latest",
                        "env": {
                            "MODEL_NAME": model_name,
                            "MODEL_VERSION": "1.0",
                            "LOG_LEVEL": "INFO"
                        },
                        "resources": {
                            "requests": {
                                "memory": "512Mi",
                                "cpu": "0.5"
                            },
                            "limits": {
                                "memory": "1Gi",
                                "cpu": "1"
                            }
                        },
                        "readinessProbe": {
                            "port": 8080,
                            "path": "/health"
                        }
                    }
                ],
                "endpoints": [
                    {
                        "name": "inference-api",
                        "port": 8080,
                        "public": True
                    }
                ]
            }
        }
    
    def create_realtime_endpoint(self, session: Session, model_name: str):
        """Create UDF that calls the real-time service"""
        
        endpoint_udf_sql = f"""
        CREATE OR REPLACE FUNCTION predict_{model_name.lower()}_realtime(
            client_id STRING,
            engagement_score FLOAT,
            annual_income FLOAT,
            current_401k_balance FLOAT,
            age FLOAT,
            service_tier_numeric FLOAT
        )
        RETURNS OBJECT
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.8'
        PACKAGES = ('requests')
        HANDLER = 'realtime_predict'
        AS $$
import requests
import json
import time

def realtime_predict(client_id, engagement_score, annual_income, current_401k_balance, age, service_tier_numeric):
    start_time = time.time()
    
    try:
        # Service endpoint URL (replace with actual service URL)
        service_url = f"https://{model_name.lower()}-realtime.snowflakecomputing.app/predict"
        
        # Prepare features
        features = {{
            "client_id": client_id,
            "engagement_score_30d": engagement_score,
            "annual_income": annual_income,
            "current_401k_balance": current_401k_balance,
            "age": age,
            "service_tier_numeric": service_tier_numeric
        }}
        
        # Call service with timeout
        response = requests.post(
            service_url, 
            json={{"features": features}},
            timeout=0.1,  # 100ms timeout
            headers={{"Content-Type": "application/json"}}
        )
        
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            result["response_time_ms"] = response_time
            result["source"] = "container_service"
            return result
        else:
            raise Exception(f"Service returned {response.status_code}")
            
    except Exception as e:
        # ALWAYS provide fallback for real-time systems
        response_time = (time.time() - start_time) * 1000
        
        # Simple fallback prediction
        if engagement_score > 0.6 and annual_income > 100000:
            probability = 0.75
            action = "Schedule_Wealth_Consultation"
        elif engagement_score > 0.3 and current_401k_balance > 50000:
            probability = 0.55
            action = "Schedule_Planning_Session"
        else:
            probability = 0.30
            action = "Educational_Content"
        
        return {{
            "conversion_probability": probability,
            "confidence": 0.7,
            "next_action": action,
            "response_time_ms": response_time,
            "source": "fallback",
            "error": str(e)
        }}
$$
        """
        
        session.sql(endpoint_udf_sql).collect()
        print("✅ Real-time endpoint UDF created with fallback")
    
    def setup_realtime_monitoring(self, session: Session, model_name: str):
        """Set up monitoring for real-time inference"""
        
        # Create real-time monitoring table
        monitoring_sql = f"""
        CREATE TABLE IF NOT EXISTS realtime_inference_log (
            inference_id STRING DEFAULT UUID_STRING(),
            client_id STRING,
            model_name STRING,
            prediction_result OBJECT,
            response_time_ms FLOAT,
            source STRING,
            timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        session.sql(monitoring_sql).collect()
        
        # Create monitoring UDF wrapper
        monitored_udf_sql = f"""
        CREATE OR REPLACE FUNCTION predict_{model_name.lower()}_monitored(
            client_id STRING,
            engagement_score FLOAT,
            annual_income FLOAT,
            current_401k_balance FLOAT,
            age FLOAT,
            service_tier_numeric FLOAT
        )
        RETURNS OBJECT
        LANGUAGE SQL
        AS $$
            SELECT prediction_result
            FROM (
                SELECT 
                    predict_{model_name.lower()}_realtime(
                        client_id, engagement_score, annual_income,
                        current_401k_balance, age, service_tier_numeric
                    ) as prediction_result
            )
            WHERE (
                INSERT INTO realtime_inference_log (
                    client_id, model_name, prediction_result, 
                    response_time_ms, source
                )
                SELECT 
                    client_id,
                    '{model_name}',
                    prediction_result,
                    prediction_result:response_time_ms::FLOAT,
                    prediction_result:source::STRING
            ) IS NOT NULL
        $$
        """
        
        session.sql(monitored_udf_sql).collect()
        print("✅ Real-time monitoring enabled")
    
    def create_session(self) -> Session:
        """Create Snowpark session"""
        if not self.session:
            connection_parameters = {
                "account": self.config['snowflake']['account'],
                "user": self.config['snowflake']['user'],
                "password": self.config['snowflake']['password'],
                "role": self.config['snowflake']['role'],
                "warehouse": self.config['snowflake']['warehouse'],
                "database": self.config['snowflake']['database'],
                "schema": self.config['snowflake']['schema']
            }
            
            self.session = Session.builder.configs(connection_parameters).create()
        
        return self.session
    
    def test_realtime_performance(self, model_name: str):
        """Test real-time inference performance"""
        session = self.create_session()
        
        print("🔍 Testing real-time inference performance...")
        
        # Test with sample data
        test_sql = f"""
        SELECT 
            predict_{model_name.lower()}_monitored(
                'test_client_' || ROW_NUMBER() OVER (ORDER BY NULL),
                UNIFORM(0.1, 0.9, RANDOM()),
                UNIFORM(30000, 150000, RANDOM()),
                UNIFORM(10000, 200000, RANDOM()),
                UNIFORM(25, 65, RANDOM()),
                UNIFORM(1, 3, RANDOM())
            ) as prediction
        FROM TABLE(GENERATOR(ROWCOUNT => 10))
        """
        
        try:
            results = session.sql(test_sql).collect()
            
            response_times = []
            sources = []
            
            for result in results:
                pred = result['PREDICTION']
                if isinstance(pred, dict):
                    response_times.append(pred.get('response_time_ms', 0))
                    sources.append(pred.get('source', 'unknown'))
            
            if response_times:
                avg_response = sum(response_times) / len(response_times)
                max_response = max(response_times)
                
                print(f"✅ Performance test completed:")
                print(f"   📊 Average response time: {avg_response:.1f}ms")
                print(f"   📊 Maximum response time: {max_response:.1f}ms")
                print(f"   📊 Success rate: {len([s for s in sources if s == 'container_service'])}/{len(sources)}")
                
                if avg_response < 100:
                    print("🎯 Performance target met (<100ms)")
                else:
                    print("⚠️  Performance target missed (>100ms)")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
    
    def create_realtime_dashboard(self):
        """Create dashboard for real-time inference monitoring"""
        session = self.create_session()
        
        dashboard_sql = """
        CREATE OR REPLACE VIEW realtime_performance_dashboard AS
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_requests,
            ROUND(AVG(response_time_ms), 2) as avg_response_time_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms), 2) as p95_response_time_ms,
            COUNT(CASE WHEN source = 'container_service' THEN 1 END) as service_requests,
            COUNT(CASE WHEN source = 'fallback' THEN 1 END) as fallback_requests,
            ROUND(
                COUNT(CASE WHEN source = 'container_service' THEN 1 END) * 100.0 / COUNT(*), 2
            ) as service_success_rate_pct
        FROM realtime_inference_log
        WHERE timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP())
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """
        
        session.sql(dashboard_sql).collect()
        print("✅ Real-time performance dashboard created")

def main():
    """Main setup for real-time inference"""
    print("⚡ Real-time ML Inference Setup")
    print("="*50)
    print("⚠️  WARNING: Use this ONLY when batch inference isn't fast enough!")
    print("   💰 Cost: Significantly higher than batch")
    print("   🔧 Maintenance: More complex")
    print("   ⚡ Use cases: Advisor calls, live website interactions")
    
    proceed = input("\nDo you really need <100ms response times? (y/N): ").lower().strip()
    
    if proceed != 'y':
        print("✅ Good choice! Stick with batch inference for most use cases.")
        return
    
    service = RealtimeMLService()
    
    # Set up container service
    print("\n1️⃣  Setting up container service...")
    success = service.setup_container_service()
    
    if success:
        # Test performance
        print("\n2️⃣  Testing performance...")
        service.test_realtime_performance("CONVERSION_PREDICTOR")
        
        # Create dashboard
        print("\n3️⃣  Creating monitoring dashboard...")
        service.create_realtime_dashboard()
        
        print("\n🎉 Real-time inference setup complete!")
        print("\n📋 Usage:")
        print("   SELECT predict_conversion_predictor_monitored(")
        print("       'client_123', 0.7, 85000, 125000, 45, 2")
        print("   );")
        
    else:
        print("\n❌ Setup failed. Consider using batch inference instead.")

if __name__ == "__main__":
    main()
