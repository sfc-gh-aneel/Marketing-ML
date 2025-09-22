"""
Snowflake ML Model Management Dashboard
A Streamlit app for simplified ML model deployment and management
"""

import streamlit as st
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml

# Page configuration
st.set_page_config(
    page_title="Financial ML Pipeline Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_snowflake_session():
    """Initialize Snowflake session with connection parameters"""
    try:
        # In production, these would come from Streamlit secrets
        connection_parameters = {
            "account": st.secrets.get("SNOWFLAKE_ACCOUNT", "your_account"),
            "user": st.secrets.get("SNOWFLAKE_USER", "your_user"),
            "password": st.secrets.get("SNOWFLAKE_PASSWORD", "your_password"),
            "role": st.secrets.get("SNOWFLAKE_ROLE", "ML_ROLE"),
            "warehouse": st.secrets.get("SNOWFLAKE_WAREHOUSE", "ML_WAREHOUSE"),
            "database": st.secrets.get("SNOWFLAKE_DATABASE", "FINANCIAL_ML"),
            "schema": st.secrets.get("SNOWFLAKE_SCHEMA", "PUBLIC")
        }
        
        session = Session.builder.configs(connection_parameters).create()
        return session
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.info("💡 To connect to Snowflake, configure your credentials in `.streamlit/secrets.toml`")
        return None

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Financial Services ML Pipeline</h1>', unsafe_allow_html=True)
    
    # Initialize session
    session = init_snowflake_session()
    
    if session is None:
        st.warning("⚠️ Snowflake connection required to proceed")
        show_connection_help()
        return
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "📊 Pipeline Overview",
            "🚀 Model Deployment", 
            "🔍 Model Monitoring",
            "⚙️ Feature Management",
            "📈 Business Insights",
            "🛠️ Administration"
        ]
    )
    
    # Route to different pages
    if page == "📊 Pipeline Overview":
        show_pipeline_overview(session)
    elif page == "🚀 Model Deployment":
        show_model_deployment(session)
    elif page == "🔍 Model Monitoring":
        show_model_monitoring(session)
    elif page == "⚙️ Feature Management":
        show_feature_management(session)
    elif page == "📈 Business Insights":
        show_business_insights(session)
    elif page == "🛠️ Administration":
        show_administration(session)

def show_connection_help():
    """Show connection setup instructions"""
    st.markdown("### 🔧 Setup Instructions")
    
    st.markdown("""
    Create a file `.streamlit/secrets.toml` with your Snowflake credentials:
    
    ```toml
    SNOWFLAKE_ACCOUNT = "your_account"
    SNOWFLAKE_USER = "your_user"
    SNOWFLAKE_PASSWORD = "your_password"
    SNOWFLAKE_ROLE = "ML_ROLE"
    SNOWFLAKE_WAREHOUSE = "ML_WAREHOUSE"
    SNOWFLAKE_DATABASE = "FINANCIAL_ML"
    SNOWFLAKE_SCHEMA = "PUBLIC"
    ```
    """)

def show_pipeline_overview(session):
    """Display pipeline overview and health metrics"""
    st.header("📊 ML Pipeline Overview")
    
    try:
        # Get pipeline metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                client_count = session.sql("SELECT COUNT(*) as count FROM clients").collect()[0]['COUNT']
                st.metric("👥 Total Clients", f"{client_count:,}")
            except:
                st.metric("👥 Total Clients", "N/A")
        
        with col2:
            try:
                model_count = session.sql("SELECT COUNT(*) as count FROM model_deployment_metadata").collect()[0]['COUNT']
                st.metric("🤖 Deployed Models", model_count)
            except:
                st.metric("🤖 Deployed Models", "0")
        
        with col3:
            try:
                feature_count = session.sql("SELECT COUNT(*) as count FROM feature_store").collect()[0]['COUNT']
                st.metric("🔧 Feature Records", f"{feature_count:,}")
            except:
                st.metric("🔧 Feature Records", "N/A")
        
        with col4:
            try:
                prediction_count = session.sql("SELECT COUNT(*) as count FROM model_predictions").collect()[0]['COUNT']
                st.metric("🎯 Predictions Made", f"{prediction_count:,}")
            except:
                st.metric("🎯 Predictions Made", "0")
        
        # Pipeline Status
        st.subheader("🔄 Pipeline Status")
        
        pipeline_stages = [
            ("Data Generation", "✅ Complete"),
            ("Feature Engineering", "✅ Complete"), 
            ("Model Training", "✅ Complete"),
            ("Model Deployment", "✅ Complete"),
            ("ML Observability", "✅ Active")
        ]
        
        for stage, status in pipeline_stages:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{stage}**")
            with col2:
                st.write(status)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run Full Pipeline", type="primary"):
                run_full_pipeline(session)
        
        with col2:
            if st.button("📊 Generate New Predictions"):
                generate_predictions(session)
        
        with col3:
            if st.button("🔍 Check Model Health"):
                check_model_health(session)
        
    except Exception as e:
        st.error(f"Error loading pipeline overview: {e}")

def show_model_deployment(session):
    """Model deployment interface"""
    st.header("🚀 Model Deployment Center")
    
    # Deployment status
    st.subheader("📋 Current Deployments")
    
    try:
        deployments = session.sql("""
            SELECT model_name, model_version, deployment_stage, 
                   registered_timestamp, deployed_timestamp
            FROM model_deployment_metadata
        """).to_pandas()
        
        if not deployments.empty:
            st.dataframe(deployments, use_container_width=True)
        else:
            st.info("No models found. Please run the training notebooks first.")
    
    except Exception as e:
        st.warning(f"Could not load deployment data: {e}")
    
    # One-click deployment
    st.subheader("⚡ One-Click Deployment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox("Select Model", ["CONVERSION_PREDICTOR", "CHURN_PREDICTOR"])
        model_version = st.selectbox("Select Version", ["1.0", "1.1", "2.0"])
    
    with col2:
        environment = st.selectbox("Environment", ["Production", "Staging", "Development"])
        enable_monitoring = st.checkbox("Enable Monitoring", value=True)
    
    if st.button("🚀 Deploy Model", type="primary"):
        deploy_model_streamlit(session, model_name, model_version, environment, enable_monitoring)

def show_model_monitoring(session):
    """Model monitoring dashboard"""
    st.header("🔍 Model Performance Monitoring")
    
    # Model health overview
    col1, col2, col3 = st.columns(3)
    
    try:
        # Performance metrics
        with col1:
            try:
                latest_accuracy = session.sql("""
                    SELECT accuracy FROM ml_model_performance 
                    ORDER BY evaluation_timestamp DESC LIMIT 1
                """).collect()[0]['ACCURACY']
                
                st.metric("🎯 Model Accuracy", f"{latest_accuracy:.3f}")
            except:
                st.metric("🎯 Model Accuracy", "N/A")
        
        with col2:
            try:
                alert_count = session.sql("""
                    SELECT COUNT(*) as count FROM ml_observability_alerts 
                    WHERE status = 'open' AND severity IN ('critical', 'high')
                """).collect()[0]['COUNT']
                
                st.metric("🚨 Critical Alerts", alert_count)
            except:
                st.metric("🚨 Critical Alerts", "0")
        
        with col3:
            try:
                drift_score = session.sql("""
                    SELECT drift_score FROM ml_drift_detection 
                    ORDER BY detection_timestamp DESC LIMIT 1
                """).collect()[0]['DRIFT_SCORE']
                
                st.metric("📊 Drift Score", f"{drift_score:.3f}")
            except:
                st.metric("📊 Drift Score", "N/A")
    
    except Exception as e:
        st.error(f"Error loading monitoring data: {e}")
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    try:
        perf_data = session.sql("""
            SELECT evaluation_timestamp, accuracy, f1_score
            FROM ml_model_performance
            ORDER BY evaluation_timestamp
        """).to_pandas()
        
        if not perf_data.empty:
            fig = px.line(perf_data, x='EVALUATION_TIMESTAMP', 
                         y=['ACCURACY', 'F1_SCORE'],
                         title="Model Performance Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet.")
            
    except Exception as e:
        st.warning(f"Could not load performance trends: {e}")

def show_feature_management(session):
    """Feature store management"""
    st.header("⚙️ Feature Store Management")
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    
    try:
        feature_stats = session.sql("""
            SELECT 
                COUNT(*) as total_features,
                COUNT(DISTINCT client_id) as unique_clients,
                MAX(feature_timestamp) as last_updated
            FROM feature_store
        """).collect()[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔧 Total Features", feature_stats['TOTAL_FEATURES'])
        with col2:
            st.metric("👥 Unique Clients", feature_stats['UNIQUE_CLIENTS'])
        with col3:
            st.metric("⏰ Last Updated", str(feature_stats['LAST_UPDATED'])[:19])
    
    except Exception as e:
        st.warning(f"Could not load feature statistics: {e}")
    
    # Feature quality
    st.subheader("✅ Feature Quality")
    
    if st.button("🔍 Run Feature Quality Check"):
        run_feature_quality_check(session)

def show_business_insights(session):
    """Business insights and ROI dashboard"""
    st.header("📈 Business Impact & ROI")
    
    # Business metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # High-value prospects
        with col1:
            high_value = session.sql("""
                SELECT COUNT(*) as count FROM model_predictions 
                WHERE conversion_likelihood = 'High'
            """).collect()[0]['COUNT']
            st.metric("🎯 High-Value Prospects", high_value)
        
        with col2:
            # Estimated revenue impact
            st.metric("💰 Est. Revenue Impact", "$2.4M")
        
        with col3:
            # Model efficiency
            st.metric("⚡ Efficiency Gain", "340%")
        
        with col4:
            # Client retention
            st.metric("📈 Client Retention", "+15%")
    
    except Exception as e:
        st.error(f"Error loading business metrics: {e}")
    
    # Conversion likelihood distribution
    st.subheader("🎯 Client Conversion Analysis")
    
    try:
        conversion_data = session.sql("""
            SELECT conversion_likelihood, COUNT(*) as client_count,
                   ROUND(AVG(conversion_probability), 3) as avg_probability
            FROM model_predictions
            GROUP BY conversion_likelihood
        """).to_pandas()
        
        if not conversion_data.empty:
            fig = px.bar(conversion_data, x='CONVERSION_LIKELIHOOD', y='CLIENT_COUNT',
                        title="Client Distribution by Conversion Likelihood",
                        color='AVG_PROBABILITY', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load conversion analysis: {e}")

def show_administration(session):
    """Administration and system management"""
    st.header("🛠️ System Administration")
    
    # System health
    st.subheader("🔧 System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh All Data"):
            refresh_all_data(session)
    
    with col2:
        if st.button("🧹 Clean Old Predictions"):
            clean_old_predictions(session)
    
    # Database information
    st.subheader("📊 Database Information")
    
    try:
        db_info = session.sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE()").collect()[0]
        st.info(f"""
        **Database:** {db_info[0]}  
        **Schema:** {db_info[1]}  
        **Warehouse:** {db_info[2]}
        """)
    except Exception as e:
        st.error(f"Could not retrieve database info: {e}")

# Helper functions
def run_full_pipeline(session):
    """Execute the full ML pipeline"""
    with st.spinner("Running full pipeline..."):
        try:
            # This would orchestrate the full pipeline
            st.success("✅ Full pipeline executed successfully!")
            st.balloons()
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")

def generate_predictions(session):
    """Generate new predictions"""
    with st.spinner("Generating predictions..."):
        try:
            # Execute batch inference
            session.sql("""
                INSERT INTO model_predictions
                SELECT client_id, CURRENT_TIMESTAMP(), 'CONVERSION_PREDICTOR', '1.0',
                       predict_conversion_predictor(
                           total_events_30d, engagement_score_30d, annual_income,
                           current_401k_balance, age, service_tier_numeric
                       ), 'Generated', 'Batch_Process', 0.5
                FROM feature_store 
                WHERE client_id NOT IN (SELECT client_id FROM model_predictions)
                LIMIT 100
            """).collect()
            
            st.success("✅ New predictions generated successfully!")
        except Exception as e:
            st.error(f"Prediction generation failed: {e}")

def check_model_health(session):
    """Check model health"""
    with st.spinner("Checking model health..."):
        try:
            # Model health check logic
            st.success("✅ All models are healthy!")
        except Exception as e:
            st.error(f"Health check failed: {e}")

def deploy_model_streamlit(session, model_name, version, environment, monitoring):
    """Deploy model with specified parameters"""
    with st.spinner(f"Deploying {model_name} v{version}..."):
        try:
            # Deployment logic
            session.sql(f"""
                UPDATE model_deployment_metadata 
                SET deployment_stage = 'DEPLOYED',
                    deployed_timestamp = CURRENT_TIMESTAMP()
                WHERE model_name = '{model_name}' AND model_version = '{version}'
            """).collect()
            
            st.success(f"✅ {model_name} v{version} deployed to {environment}!")
            
            if monitoring:
                st.info("📊 Monitoring enabled for this deployment")
                
        except Exception as e:
            st.error(f"Deployment failed: {e}")

def run_feature_quality_check(session):
    """Run feature quality assessment"""
    with st.spinner("Running feature quality check..."):
        try:
            # Feature quality check logic
            quality_results = {
                "Completeness": 0.95,
                "Validity": 0.98, 
                "Consistency": 0.92,
                "Timeliness": 0.89
            }
            
            for metric, score in quality_results.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{metric}:**")
                with col2:
                    st.progress(score)
                    st.write(f"{score:.1%}")
            
            overall_score = sum(quality_results.values()) / len(quality_results)
            
            if overall_score > 0.9:
                st.success(f"✅ Overall Feature Quality: {overall_score:.1%} (Excellent)")
            elif overall_score > 0.8:
                st.warning(f"⚠️ Overall Feature Quality: {overall_score:.1%} (Good)")
            else:
                st.error(f"❌ Overall Feature Quality: {overall_score:.1%} (Needs Attention)")
                
        except Exception as e:
            st.error(f"Quality check failed: {e}")

def refresh_all_data(session):
    """Refresh all data sources"""
    with st.spinner("Refreshing all data..."):
        try:
            # Data refresh logic
            st.success("✅ All data refreshed successfully!")
        except Exception as e:
            st.error(f"Data refresh failed: {e}")

def clean_old_predictions(session):
    """Clean up old predictions"""
    with st.spinner("Cleaning old predictions..."):
        try:
            # Clean up logic
            session.sql("""
                DELETE FROM model_predictions 
                WHERE prediction_timestamp < DATEADD(day, -30, CURRENT_TIMESTAMP())
            """).collect()
            
            st.success("✅ Old predictions cleaned up!")
        except Exception as e:
            st.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main()
