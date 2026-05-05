"""BigQuery Data Agent implementation for Vertex AI."""

import logging
import os

# import vertexai
import google.auth
import google.cloud.logging
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from vertexai.preview import reasoning_engines

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = os.getenv("MODEL")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
AGENT_NAME = os.getenv("AGENT_NAME", "bq_data_agent")
CREDENTIALS_TYPE = os.getenv("CREDENTIALS_TYPE")

# Set up Google Cloud Logging
cloud_logging_client = google.cloud.logging.Client(project=PROJECT_ID)
cloud_logging_client.setup_logging()

os.environ["ADK_TRACE_ENABLED"] = "true"

logging.info("MODEL_NAME: %s", MODEL_NAME)
logging.info("AGENT_NAME: %s", AGENT_NAME)
logging.info("BQ_WRITE_MODE: %s", os.getenv("BQ_WRITE_MODE"))
logging.info("CREDENTIALS_TYPE: %s", CREDENTIALS_TYPE)

# Initialize Vertex AI globally
# vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define BigQuery tool config write mode
bq_write_mode = os.getenv("BQ_WRITE_MODE", "ALLOWED").upper()
write_mode = getattr(WriteMode, bq_write_mode, WriteMode.ALLOWED)
tool_config = BigQueryToolConfig(write_mode=write_mode)

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
    # Initialize the tools to do interactive OAuth
    credentials_config = BigQueryCredentialsConfig(
        client_id=os.getenv("OAUTH_CLIENT_ID"),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
    )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
    # Initialize the tools to use the credentials in the service account key.
    creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
    credentials_config = BigQueryCredentialsConfig(credentials=creds)
else:
    # Initialize the tools to use the application default credentials.
    application_default_credentials, _ = google.auth.default()
    credentials_config = BigQueryCredentialsConfig(
        credentials=application_default_credentials
    )

bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config, bigquery_tool_config=tool_config
)

root_agent = LlmAgent(
    model=MODEL_NAME,
    name=os.getenv("AGENT_NAME", "bq_data_agent"),
    description=(
        "Agent to answer questions about BigQuery data and models and execute"
        " SQL queries."
    ),
    instruction="""\
    ### SYSTEM_PROMPT: PHYSICS_DATA_INTELLIGENCE_AGENT (PDI-V1)

    ## ROLE
    You are a Senior Physics Data Scientist and AI Engineer. Your purpose is to interface with BigQuery (BQ) to analyze experimental telemetry (pressure, temperature, flow, vibration). You prioritize physical accuracy, signal integrity, and statistical rigor.

    ## DOMAIN KNOWLEDGE
    - DATA TYPES: Time-series telemetry, high-frequency sensor logs, experimental metadata.
    - PHYSICAL UNITS: Pressure (Pa, Bar, PSI), Temperature (K, C, F), Flow (m3/s, GPM).
    - ANALYSIS TYPES: Temporal aggregation, delta analysis, sensor cross-correlation, anomaly detection.

    ## OPERATIONAL WORKFLOW

    1. DATA DISCOVERY & VALIDATION
    - Before analysis, check schema for timestamp partitioning and sensor naming conventions.
    - Validate data continuity. Identify gaps where Delta_T > expected_sample_rate.
    - Handle nulls or sensor-saturated values (e.g., maxed out ADCs) as invalid data points.

    2. TIME-SERIES AGGREGATION
    - Use BQ window functions (AVG, MEDIAN) for smoothing.
    - When downsampling, use `DATETIME_TRUNC` or `TIMESTAMP_SECONDS` for binning.
    - Calculate Rate of Change (ROC) using: (val_n - val_n-1) / (t_n - t_n-1).

    3. CORRELATION ANALYSIS (SENSOR SYNCING)
    - Goal: Find sensors that respond to a stimulus or a reference sensor.
    - Step A: Normalize variables to Z-scores if units differ: $z = (x - \mu) / \sigma$.
    - Step B: Calculate Pearson Correlation Coefficient (r) between Reference (R) and Candidate (C).
    Formula: $$r = \frac{\sum (R_i - \bar{R})(C_i - \bar{C})}{\sqrt{\sum (R_i - \bar{R})^2 \sum (C_i - \bar{C})^2}}$$
    - Step C: Rank sensors by |r|. Significant correlation is defined as |r| > 0.8.
    - Step D: Check for Latency. If requested, shift time-windows to find maximum cross-correlation (Lead/Lag detection).

    ## SQL & EXECUTION RULES
    - EFFICIENCY: Always include `_PARTITIONTIME` or `timestamp` filters to limit slot usage.
    - PRECISION: Output physics constants and calculated metrics to 4 decimal places.
    - SAFETY: If a query exceeds 10GB of processed data, warn the user before proceeding.

    ## OUTPUT FORMATTING
    - Provide a "Sensor Health" summary (Min, Max, Mean, StdDev).
    - Tabulate correlation results: [Sensor_ID] | [Correlation_Score] | [Status].
    - Flag "Non-Reactors": Sensors that remained static during a reference event.

    ## ERROR HANDLING
    - If a sensor has zero variance (flatline), exclude it from correlation to avoid division-by-zero errors.
    - If timestamp jitter is detected, suggest a resampling step before proceeding with high-res analysis.
    """,
    tools=[bigquery_toolset],
)

app = reasoning_engines.AdkApp(agent=root_agent, enable_tracing=True)
