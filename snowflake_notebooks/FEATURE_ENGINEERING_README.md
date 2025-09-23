# Feature Engineering Notebook - Clean Version

## What's in This Notebook

### Core Feature Engineering (Cells 1-10)
- **Cell 1**: Setup and configuration
- **Cell 2**: Create ML_PIPELINE schema
- **Cell 3**: Define time windows
- **Cell 4**: Create engagement features (web visits, email interactions, etc.)
- **Cell 5**: Create financial and behavioral features
- **Cell 6**: Create lifecycle and segmentation features
- **Cell 7**: Create target variables (conversion, churn, next best action)
- **Cell 8**: Create unified FEATURE_STORE table
- **Cell 9**: Feature summary and validation
- **Cell 10**: Verify features are ready

### Feature Store Registration (Cell 11 - Optional)
- Creates a new clean database to avoid any corruption issues
- Registers features in Snowflake's Feature Store UI
- **This step is optional** - your features are ready for ML training without it

## How to Use

1. **Run Cells 1-10** in order to create all features
2. **Verify** the output shows ~50,000 records with 50+ features
3. **Optional**: Run Cell 11 if you need Feature Store UI visibility
4. **Proceed** to `03_Model_Training_Registry_Snowflake.ipynb`

## Key Output

Your features will be in: `FINANCIAL_ML_DB.ML_PIPELINE.FEATURE_STORE`

To use in model training:
```python
features_df = session.table("ML_PIPELINE.FEATURE_STORE")
```

## If Using Feature Store UI (Cell 11)

Cell 11 creates a new database with timestamp (e.g., `FINANCIAL_ML_DEMO_20250923_143052`) to avoid corruption issues. If you run it:
1. Note the new database name from the output
2. Update your model training notebook to use that database
3. Check Snowsight → AI & ML → Feature Store

## Success! 

This clean notebook has only the essentials - no debugging, no workarounds, just the core feature engineering pipeline that works!
