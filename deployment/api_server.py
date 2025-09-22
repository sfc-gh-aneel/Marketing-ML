"""
FastAPI Server for Real-time ML Inference
Serve financial services ML models via REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
import uvicorn
from datetime import datetime
import json
import asyncio

from inference_pipeline import FinancialInferencePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Services ML API",
    description="Real-time inference API for financial services ML models",
    version="1.0.0"
)

# Initialize inference pipeline
inference_pipeline = FinancialInferencePipeline()

# Request/Response models
class ClientFeatures(BaseModel):
    """Client features for prediction"""
    client_id: str = Field(..., description="Unique client identifier")
    age: Optional[int] = Field(None, ge=18, le=100, description="Client age")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income")
    current_401k_balance: Optional[float] = Field(None, ge=0, description="Current 401k balance")
    years_to_retirement: Optional[int] = Field(None, ge=0, le=50, description="Years to retirement")
    service_tier: Optional[str] = Field(None, description="Service tier (Basic, Premium, Elite)")
    total_events_30d: Optional[int] = Field(None, ge=0, description="Total events in last 30 days")
    email_opens_30d: Optional[int] = Field(None, ge=0, description="Email opens in last 30 days")
    web_visits_30d: Optional[int] = Field(None, ge=0, description="Web visits in last 30 days")
    engagement_score: Optional[float] = Field(None, ge=0, le=1, description="Engagement score")
    
    class Config:
        schema_extra = {
            "example": {
                "client_id": "client_123",
                "age": 45,
                "annual_income": 75000,
                "current_401k_balance": 125000,
                "years_to_retirement": 20,
                "service_tier": "Premium",
                "total_events_30d": 25,
                "email_opens_30d": 8,
                "web_visits_30d": 15,
                "engagement_score": 0.75
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    client_features: List[ClientFeatures] = Field(..., description="List of client features")
    include_explanations: bool = Field(False, description="Include prediction explanations")

class PredictionResponse(BaseModel):
    """Prediction response"""
    client_id: str
    prediction_timestamp: datetime
    conversion_probability: float = Field(..., ge=0, le=1)
    conversion_risk_category: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_risk_category: str
    recommended_action: str
    action_confidence: float = Field(..., ge=0, le=1)
    action_priority: str
    engagement_priority_score: float = Field(..., ge=0, le=1)
    model_version: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float
    success_rate: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: int
    version: str

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Services ML API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = len(inference_pipeline.loaded_models)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=models_loaded,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(client_features: ClientFeatures):
    """Single client prediction endpoint"""
    try:
        # Convert to dictionary
        features_dict = client_features.dict()
        
        # Perform inference
        result = inference_pipeline.real_time_inference(features_dict)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        start_time = datetime.now()
        
        # Convert to DataFrame
        features_list = [client.dict() for client in request.client_features]
        features_df = pd.DataFrame(features_list)
        
        # Perform batch inference
        predictions_df = inference_pipeline.generate_comprehensive_predictions(features_df)
        
        # Convert to response format
        predictions = []
        for _, row in predictions_df.iterrows():
            prediction = PredictionResponse(
                client_id=row['CLIENT_ID'],
                prediction_timestamp=row['PREDICTION_TIMESTAMP'],
                conversion_probability=row['CONVERSION_PROBABILITY'],
                conversion_risk_category=row['CONVERSION_RISK_CATEGORY'],
                churn_probability=row['CHURN_PROBABILITY'],
                churn_risk_category=row['CHURN_RISK_CATEGORY'],
                recommended_action=row['RECOMMENDED_ACTION'],
                action_confidence=row['ACTION_CONFIDENCE'],
                action_priority=row['ACTION_PRIORITY'],
                engagement_priority_score=row['ENGAGEMENT_PRIORITY_SCORE'],
                model_version=row['MODEL_VERSION']
            )
            predictions.append(prediction)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time,
            success_rate=1.0  # All successful for now
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available models"""
    try:
        if not inference_pipeline.registry:
            inference_pipeline.initialize_registry()
        
        models_df = inference_pipeline.registry.list_models() if inference_pipeline.registry else pd.DataFrame()
        
        return {
            "models": models_df.to_dict('records') if not models_df.empty else [],
            "loaded_models": list(inference_pipeline.loaded_models.keys()),
            "total_models": len(models_df)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"models": [], "loaded_models": [], "total_models": 0, "error": str(e)}

@app.get("/client/{client_id}/predictions", response_model=Dict[str, Any])
async def get_client_predictions(client_id: str, limit: int = 10):
    """Get historical predictions for a specific client"""
    try:
        if not inference_pipeline.session:
            inference_pipeline.create_snowpark_session()
        
        query = f"""
        SELECT * FROM MODEL_PREDICTIONS 
        WHERE CLIENT_ID = '{client_id}' 
        ORDER BY PREDICTION_TIMESTAMP DESC 
        LIMIT {limit}
        """
        
        predictions_df = inference_pipeline.session.sql(query).to_pandas()
        
        return {
            "client_id": client_id,
            "predictions": predictions_df.to_dict('records'),
            "total_predictions": len(predictions_df)
        }
        
    except Exception as e:
        logger.error(f"Error fetching client predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/batch-job")
async def trigger_batch_inference(background_tasks: BackgroundTasks):
    """Trigger batch inference job"""
    try:
        def run_batch_job():
            processed_count = inference_pipeline.batch_inference()
            logger.info(f"Batch job completed. Processed {processed_count} records")
        
        background_tasks.add_task(run_batch_job)
        
        return {
            "message": "Batch inference job triggered",
            "status": "running",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error triggering batch job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/model-performance", response_model=Dict[str, Any])
async def get_model_performance():
    """Get model performance analytics"""
    try:
        if not inference_pipeline.session:
            inference_pipeline.create_snowpark_session()
        
        # Get recent model performance metrics
        performance_query = """
        SELECT 
            model_name,
            model_version,
            metric_name,
            metric_value,
            evaluation_date
        FROM MODEL_PERFORMANCE_METRICS 
        WHERE evaluation_date >= DATEADD(day, -30, CURRENT_DATE())
        ORDER BY evaluation_date DESC
        """
        
        performance_df = inference_pipeline.session.sql(performance_query).to_pandas()
        
        # Get prediction volume statistics
        volume_query = """
        SELECT 
            DATE(prediction_timestamp) as prediction_date,
            COUNT(*) as prediction_count,
            AVG(conversion_probability) as avg_conversion_prob,
            AVG(churn_probability) as avg_churn_prob
        FROM MODEL_PREDICTIONS 
        WHERE prediction_timestamp >= DATEADD(day, -7, CURRENT_DATE())
        GROUP BY DATE(prediction_timestamp)
        ORDER BY prediction_date DESC
        """
        
        volume_df = inference_pipeline.session.sql(volume_query).to_pandas()
        
        return {
            "model_performance": performance_df.to_dict('records'),
            "prediction_volumes": volume_df.to_dict('records'),
            "last_updated": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return {"error": str(e), "model_performance": [], "prediction_volumes": []}

@app.get("/explain/{client_id}")
async def explain_prediction(client_id: str):
    """Get prediction explanation for a specific client"""
    try:
        # This would integrate with SHAP or LIME for model explanations
        # For now, returning mock explanation
        
        explanation = {
            "client_id": client_id,
            "explanation_type": "feature_importance",
            "top_features": [
                {"feature": "engagement_score", "importance": 0.25, "value": 0.75},
                {"feature": "annual_income", "importance": 0.20, "value": 75000},
                {"feature": "web_visits_30d", "importance": 0.18, "value": 15},
                {"feature": "email_opens_30d", "importance": 0.15, "value": 8},
                {"feature": "age", "importance": 0.12, "value": 45}
            ],
            "model_version": "1.0",
            "explanation_confidence": 0.85
        }
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Financial Services ML API...")
    
    try:
        # Load preprocessors
        inference_pipeline.load_preprocessors()
        
        # Create monitoring tables if they don't exist
        inference_pipeline.create_inference_monitoring_table()
        inference_pipeline.create_prediction_api_view()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Financial Services ML API...")
    
    try:
        if inference_pipeline.session:
            inference_pipeline.session.close()
        
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Custom middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
