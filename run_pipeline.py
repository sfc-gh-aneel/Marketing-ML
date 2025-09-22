#!/usr/bin/env python3
"""
End-to-End Financial Services ML Pipeline Runner
Orchestrates the complete pipeline from data generation to model deployment
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinancialMLPipelineRunner:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            sys.exit(1)
    
    def run_data_generation(self):
        """Step 1: Generate synthetic data"""
        logger.info("=== STEP 1: Data Generation ===")
        
        try:
            from data.synthetic.data_generator import FinancialDataGenerator
            from data.synthetic.snowflake_connector import SnowflakeConnector, DataLoader
            
            # Generate synthetic data
            generator = FinancialDataGenerator(self.config_path)
            clients, advisors, events = generator.generate_all_data()
            generator.save_data(clients, advisors, events)
            
            logger.info(f"Generated {len(clients)} clients, {len(advisors)} advisors, {len(events)} events")
            
            # Load to Snowflake (if connection available)
            try:
                connector = SnowflakeConnector(self.config_path)
                loader = DataLoader(connector)
                loader.setup_database()
                loader.load_synthetic_data()
                logger.info("Data loaded to Snowflake successfully")
            except Exception as e:
                logger.warning(f"Snowflake loading failed (continuing with local files): {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            return False
    
    def run_feature_engineering(self):
        """Step 2: Feature engineering"""
        logger.info("=== STEP 2: Feature Engineering ===")
        
        try:
            from features.feature_engineering import FinancialFeatureEngineer
            
            engineer = FinancialFeatureEngineer(self.config_path)
            
            # Create comprehensive feature set
            features = engineer.create_comprehensive_feature_set()
            targets = engineer.create_target_variables()
            
            # Merge features and targets
            ml_dataset = features.merge(targets, on='CLIENT_ID', how='inner')
            
            # Save feature dataset
            output_path = "data/synthetic/output/ml_features.csv"
            ml_dataset.to_csv(output_path, index=False)
            
            logger.info(f"Feature engineering completed: {ml_dataset.shape}")
            logger.info(f"Features saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return False
    
    def run_model_training(self):
        """Step 3: Model training"""
        logger.info("=== STEP 3: Model Training ===")
        
        try:
            import pandas as pd
            from models.pytorch_models import FinancialMLTrainer as PyTorchTrainer
            from models.traditional_models import TraditionalMLTrainer
            
            # Load feature data
            df = pd.read_csv("data/synthetic/output/ml_features.csv")
            logger.info(f"Loaded feature data: {df.shape}")
            
            # Train PyTorch models
            try:
                pytorch_trainer = PyTorchTrainer(self.config_path)
                pytorch_models = pytorch_trainer.train_all_models(df)
                logger.info(f"PyTorch models trained: {list(pytorch_models.keys())}")
            except Exception as e:
                logger.warning(f"PyTorch training failed: {e}")
            
            # Train traditional ML models
            traditional_trainer = TraditionalMLTrainer(self.config_path)
            traditional_results = traditional_trainer.train_all_traditional_models(df)
            
            logger.info(f"Traditional models trained: {list(traditional_results['models'].keys())}")
            logger.info("Model evaluation completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def run_model_registry(self):
        """Step 4: Model registry"""
        logger.info("=== STEP 4: Model Registry ===")
        
        try:
            from models.model_registry import SnowflakeModelRegistry
            
            registry_manager = SnowflakeModelRegistry(self.config_path)
            
            # Register all models
            registered_models = registry_manager.register_all_models()
            logger.info(f"Registered {len(registered_models)} models in Snowflake Model Registry")
            
            # Create monitoring infrastructure
            registry_manager.create_model_monitoring_view()
            
            # Export model catalog
            catalog = registry_manager.export_model_catalog()
            logger.info("Model catalog exported")
            
            return True
            
        except Exception as e:
            logger.warning(f"Model registry failed (requires Snowflake connection): {e}")
            logger.info("Models are saved locally and ready for registry when connected")
            return True  # Continue pipeline
    
    def run_inference_pipeline(self):
        """Step 5: Inference pipeline setup"""
        logger.info("=== STEP 5: Inference Pipeline ===")
        
        try:
            from deployment.inference_pipeline import FinancialInferencePipeline
            
            pipeline = FinancialInferencePipeline(self.config_path)
            
            # Create monitoring infrastructure
            try:
                pipeline.create_inference_monitoring_table()
                pipeline.create_prediction_api_view()
                logger.info("Inference monitoring infrastructure created")
            except Exception as e:
                logger.warning(f"Monitoring setup failed (requires Snowflake): {e}")
            
            # Test batch inference
            try:
                import pandas as pd
                sample_data = pd.read_csv("data/synthetic/output/ml_features.csv").head(100)
                
                pipeline.load_preprocessors()
                predictions = pipeline.generate_comprehensive_predictions(sample_data)
                
                logger.info(f"Batch inference test completed: {len(predictions)} predictions")
            except Exception as e:
                logger.warning(f"Batch inference test failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Inference pipeline setup failed: {e}")
            return False
    
    def run_streaming_demo(self, duration_minutes: int = 2):
        """Step 6: Streaming demonstration"""
        logger.info("=== STEP 6: Streaming Demo ===")
        
        try:
            from data.streaming.event_streamer import StreamingDemo
            
            demo = StreamingDemo(use_s3=False)  # Local streaming for demo
            demo.run_demo(duration_minutes=duration_minutes)
            
            logger.info(f"Streaming demo completed ({duration_minutes} minutes)")
            return True
            
        except Exception as e:
            logger.error(f"Streaming demo failed: {e}")
            return False
    
    def run_full_pipeline(self, include_streaming: bool = True):
        """Run the complete end-to-end pipeline"""
        logger.info("Starting Financial Services ML Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Pipeline steps
        steps = [
            ("Data Generation", self.run_data_generation),
            ("Feature Engineering", self.run_feature_engineering),
            ("Model Training", self.run_model_training),
            ("Model Registry", self.run_model_registry),
            ("Inference Pipeline", self.run_inference_pipeline),
        ]
        
        if include_streaming:
            steps.append(("Streaming Demo", lambda: self.run_streaming_demo(2)))
        
        # Execute pipeline steps
        failed_steps = []
        for step_name, step_func in steps:
            try:
                success = step_func()
                if success:
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.error(f"✗ {step_name} failed")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"✗ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        # Pipeline summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Steps completed: {len(steps) - len(failed_steps)}/{len(steps)}")
        
        if failed_steps:
            logger.warning(f"Failed steps: {', '.join(failed_steps)}")
        else:
            logger.info("✓ All steps completed successfully!")
        
        # Generated artifacts
        artifacts = [
            "data/synthetic/output/clients.csv",
            "data/synthetic/output/advisors.csv", 
            "data/synthetic/output/marketing_events.csv",
            "data/synthetic/output/ml_features.csv",
            "models/",
            "notebooks/"
        ]
        
        logger.info("\nGenerated artifacts:")
        for artifact in artifacts:
            if os.path.exists(artifact):
                logger.info(f"✓ {artifact}")
            else:
                logger.info(f"✗ {artifact} (missing)")
        
        logger.info("\nNext steps:")
        logger.info("1. Review generated data and features")
        logger.info("2. Examine trained models and performance metrics")
        logger.info("3. Test inference pipeline with real data")
        logger.info("4. Deploy to production environment")
        logger.info("5. Set up monitoring and alerts")
        
        return len(failed_steps) == 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Financial Services ML Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--step", choices=[
        "data", "features", "models", "registry", "inference", "streaming", "all"
    ], default="all", help="Pipeline step to run")
    parser.add_argument("--no-streaming", action="store_true", help="Skip streaming demo")
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = FinancialMLPipelineRunner(args.config)
    
    # Run specific step or full pipeline
    if args.step == "all":
        success = runner.run_full_pipeline(include_streaming=not args.no_streaming)
    elif args.step == "data":
        success = runner.run_data_generation()
    elif args.step == "features":
        success = runner.run_feature_engineering()
    elif args.step == "models":
        success = runner.run_model_training()
    elif args.step == "registry":
        success = runner.run_model_registry()
    elif args.step == "inference":
        success = runner.run_inference_pipeline()
    elif args.step == "streaming":
        success = runner.run_streaming_demo(5)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
