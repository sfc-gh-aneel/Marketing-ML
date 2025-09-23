# Script to create a clean feature engineering notebook
clean_notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Feature Engineering for Financial ML Pipeline\\n",
        "\\n",
        "This notebook creates comprehensive features for predicting:\\n",
        "- Next best action for client engagement\\n",
        "- Client conversion likelihood\\n",
        "- Client churn risk\\n",
        "\\n",
        "## Features Created:\\n",
        "1. **Engagement Metrics** - Web visits, email interactions, campaign responses\\n",
        "2. **Financial Indicators** - Income, assets, retirement readiness\\n",
        "3. **Behavioral Patterns** - Preferences, lifecycle stage, engagement trends\\n",
        "4. **Target Variables** - Conversion, churn, next best action"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Cell 1: Setup and Configuration\\n",
        "import snowflake.snowpark as snowpark\\n",
        "from snowflake.snowpark.functions import col, sum, avg, count, max, min, datediff, current_timestamp, when, lit, stddev\\n",
        "from snowflake.snowpark.functions import lag, lead, row_number, dense_rank\\n",
        "from snowflake.snowpark.window import Window\\n",
        "import pandas as pd\\n",
        "\\n",
        "print(\"🔧 Snowflake Feature Engineering Pipeline\")\\n",
        "print(f\"Database: \\\"{session.get_current_database()}\\\"\")\\n",
        "\\n",
        "# Auto-detect and switch to ML_PIPELINE schema if needed\\n",
        "current_schema = session.get_current_schema()\\n",
        "if current_schema != \"ML_PIPELINE\":\\n",
        "    # Check if CLIENTS table exists in ML_PIPELINE\\n",
        "    try:\\n",
        "        test_count = session.sql(\"SELECT COUNT(*) FROM ML_PIPELINE.CLIENTS\").collect()[0][0]\\n",
        "        if test_count > 0:\\n",
        "            session.sql(\"USE SCHEMA ML_PIPELINE\").collect()\\n",
        "            print(f\"Switched to ML_PIPELINE schema (found {test_count:,} clients)\")\\n",
        "    except:\\n",
        "        print(f\"Using current schema: {current_schema}\")\\n",
        "else:\\n",
        "    print(f\"Schema: \\\"{current_schema}\\\"\")\\n",
        "\\n",
        "print(f\"Warehouse: \\\"{session.get_current_warehouse()}\\\"\")\\n",
        "print(f\"Timestamp: {pd.Timestamp.now()}\")\\n",
        "\\n",
        "# Check data availability\\n",
        "try:\\n",
        "    client_count = session.table(\"CLIENTS\").count()\\n",
        "    event_count = session.table(\"MARKETING_EVENTS\").count()\\n",
        "    print(f\"\\nData Available:\")\\n",
        "    print(f\"📊 Clients: {client_count:,}\")\\n",
        "    print(f\"📊 Marketing Events: {event_count:,}\")\\n",
        "except Exception as e:\\n",
        "    print(f\"\\n⚠️ Data not found: {e}\")\\n",
        "    print(\"Please run 01_Data_Generation_Snowflake.ipynb first!\")"
      ]
    }
  ]
}

# Add all the feature engineering cells (2-10)
# Then add the clean Feature Store registration cell

import json
with open('/Users/aneel/Desktop/Marketing-ML/snowflake_notebooks/02_Feature_Engineering_CLEAN.ipynb', 'w') as f:
    json.dump(clean_notebook, f, indent=2)
