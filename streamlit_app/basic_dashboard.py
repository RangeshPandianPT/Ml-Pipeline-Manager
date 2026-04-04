import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# -----------------------------------------------------------------------------
# Config & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="📊",
    layout="wide"
)

DB_PATH = "pipeline_metadata.db"

# -----------------------------------------------------------------------------
# Database Connection
# -----------------------------------------------------------------------------
@st.cache_resource
def get_connection():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_connection()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_data(table_name, limit=50):
    if not conn:
        return pd.DataFrame()
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def parse_json_col(df, col_name):
    """Parses a stringified JSON column into a proper dict/list"""
    if col_name in df.columns:
        df[col_name] = df[col_name].apply(lambda x: json.loads(x) if x else {})
    return df

# -----------------------------------------------------------------------------
# Sidebar & Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("MLOps Dashboard 🚀")
page = st.sidebar.radio("Navigate", ["Overview", "Drift Monitoring", "Ingestion Logs", "Models"])

st.sidebar.markdown("---")
st.sidebar.info(f"Database: `{DB_PATH}`")

if not conn:
    st.error(f"Database file not found at `{DB_PATH}`. Run the pipeline first to generate data.")
    st.stop()

# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("Pipeline Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Quick Stats
    ingest_count = pd.read_sql("SELECT COUNT(*) FROM ingestion_logs", conn).iloc[0,0]
    drift_count = pd.read_sql("SELECT COUNT(*) FROM drift_logs WHERE drift_detected=1", conn).iloc[0,0]
    latest_run = pd.read_sql("SELECT MAX(timestamp) FROM ingestion_logs", conn).iloc[0,0]
    
    col1.metric("Total Ingestions", ingest_count)
    col2.metric("Drift Alerts", drift_count)
    col3.metric("Last Run", latest_run if latest_run else "N/A")
    
    st.markdown("### Recent Activity")
    recent = load_data("ingestion_logs", 10)
    st.dataframe(recent[["timestamp", "status", "rows_ingested", "source_path"]])

# -----------------------------------------------------------------------------
# Page: Drift Monitoring
# -----------------------------------------------------------------------------
elif page == "Drift Monitoring":
    st.title("🔍 Data Drift Monitoring")
    
    drift_df = load_data("drift_logs", 100)
    
    if drift_df.empty:
        st.info("No drift logs found.")
    else:
        # Parse JSON columns for details
        drift_df = parse_json_col(drift_df, "p_values")
        drift_df = parse_json_col(drift_df, "ks_statistics")
        
        # 1. Drift Trend
        st.subheader("Drift Detection Timeline")
        # specific 0/1 casting for plot
        drift_df['drift_detected_int'] = drift_df['drift_detected'].astype(int)
        
        fig = px.scatter(
            drift_df, 
            x="timestamp", 
            y="drift_detected_int",
            color="drift_detected_int",
            title="Drift Events (1=Detected, 0=Stable)",
            color_continuous_scale=["green", "red"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Detailed View
        st.subheader("Latest Drift Analysis")
        selected_run = st.selectbox("Select Run", drift_df['drift_id'].unique())
        
        run_data = drift_df[drift_df['drift_id'] == selected_run].iloc[0]
        st.write(f"**Timestamp:** {run_data['timestamp']}")
        st.write(f"**Drift Detected:** {'🔴 YES' if run_data['drift_detected'] else '🟢 NO'}")
        
        # Show P-Values bar chart if available
        if run_data['p_values']:
            p_vals = run_data['p_values']
            p_df = pd.DataFrame(list(p_vals.items()), columns=['Feature', 'P-Value'])
            
            # Threshold line
            threshold = 0.05 # Default assumption
            
            fig_bar = px.bar(p_df, x='Feature', y='P-Value', title="K-S Test P-Values (Lower = More Drift)")
            fig_bar.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Significance Threshold (0.05)")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.json(run_data['drift_columns'])

# -----------------------------------------------------------------------------
# Page: Ingestion Logs
# -----------------------------------------------------------------------------
elif page == "Ingestion Logs":
    st.title("📥 Data Ingestion History")
    
    logs = load_data("ingestion_logs", 100)
    st.dataframe(logs)

# -----------------------------------------------------------------------------
# Page: Models
# -----------------------------------------------------------------------------
elif page == "Models":
    st.title("🤖 Model Registry")
    
    # We might not have a model_logs table if database.py didn't create it in my previous read
    # checking database.py again, I didn't see model_logs creation in the first 50 lines 
    # of _initialize_database but I saw the dataclass.
    # I'll try to query it, if fail, show error.
    
    try:
        models = load_data("model_logs", 50)
        if not models.empty:
            models = parse_json_col(models, "metrics")
            
            st.dataframe(models[['timestamp', 'model_type', 'model_id', 'status']])
            
            st.subheader("Latest Model Metrics")
            latest_model = models.iloc[0]
            st.json(latest_model['metrics'])
        else:
            st.info("No model logs found in database.")
            
    except Exception as e:
        st.warning("Model logs table might not exist yet or is empty.")
        st.error(str(e))
    
    st.markdown("### Physical Models on Disk")
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    st.write(model_files)
