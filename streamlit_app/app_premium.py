"""
Premium ML Pipeline Dashboard
Modern, interactive Streamlit application for ML pipeline management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
from datetime import datetime
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline_manager import MLPipeline, PipelineConfig
from monitor import DriftMonitor
from feature_eng import FeatureEngineer, get_available_transformations

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="ML Pipeline Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
        font-size: 1.3rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Initialize Session State ====================

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = MLPipeline()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'drift_report' not in st.session_state:
    st.session_state.drift_report = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# ==================== Helper Functions ====================

def create_metric_card(label, value, delta=None):
    """Create a styled metric card"""
    delta_html = f"<div style='font-size: 1rem; margin-top: 0.5rem;'>{delta}</div>" if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def plot_feature_importance(feature_names, importances, top_n=15):
    """Create interactive feature importance plot"""
    # Get top N features
    indices = np.argsort(importances)[-top_n:]
    
    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importances[indices],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=500,
        template="plotly_white",
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def plot_drift_results(drift_report):
    """Create drift visualization"""
    if not drift_report or not drift_report.column_results:
        return None
    
    columns = []
    p_values = []
    drift_status = []
    
    for col, result in drift_report.column_results.items():
        columns.append(col)
        p_values.append(result.p_value)
        drift_status.append("Drift Detected" if result.drift_detected else "No Drift")
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=columns,
        y=p_values,
        marker=dict(
            color=['#f56565' if status == "Drift Detected" else '#48bb78' for status in drift_status],
        ),
        text=[f"{p:.4f}" for p in p_values],
        textposition='outside'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="red",
        annotation_text="Significance Threshold (0.05)"
    )
    
    fig.update_layout(
        title="Data Drift Analysis (K-S Test P-Values)",
        xaxis_title="Features",
        yaxis_title="P-Value",
        height=500,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    
    return fig

def plot_model_metrics(metrics):
    """Create model metrics visualization"""
    if not metrics:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=list(metrics.keys()),
        specs=[[{"type": "indicator"}] * len(metrics)]
    )
    
    for i, (metric_name, value) in enumerate(metrics.items(), 1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                title={'text': metric_name},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#f56565"},
                        {'range': [0.5, 0.75], 'color': "#ed8936"},
                        {'range': [0.75, 1], 'color': "#48bb78"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        height=300,
        template="plotly_white",
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

# ==================== Sidebar ====================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # Model settings
    st.subheader("🤖 Model Settings")
    model_type = st.selectbox(
        "Model Type",
        ["random_forest", "gradient_boosting", "xgboost", "linear", "logistic_regression"],
        help="Select the machine learning algorithm"
    )
    
    validation_split = st.slider(
        "Validation Split",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Proportion of data for validation"
    )
    
    st.markdown("---")
    
    # Feature engineering settings
    st.subheader("🔧 Feature Engineering")
    auto_features = st.checkbox("Auto Feature Engineering", value=True, help="Automatically create features")
    
    if not auto_features:
        available_transforms = get_available_transformations()
        selected_transforms = st.multiselect(
            "Select Transformations",
            available_transforms,
            default=["impute_median", "standardize"]
        )
    
    st.markdown("---")
    
    # Drift monitoring settings
    st.subheader("📊 Drift Monitoring")
    check_drift = st.checkbox("Enable Drift Detection", value=True)
    
    if check_drift:
        drift_threshold = st.slider(
            "Drift Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Percentage of columns with drift to trigger alert"
        )
    
    st.markdown("---")
    
    # Pipeline state
    st.subheader("📈 Pipeline State")
    st.info(f"**Status:** {st.session_state.pipeline.state.value}")
    st.info(f"**Model Trained:** {'✅ Yes' if st.session_state.model_trained else '❌ No'}")

# ==================== Main Content ====================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 ML Pipeline Pro")
    st.markdown("**Production-Ready MLOps Platform** | Automated Feature Engineering & Drift Monitoring")
with col2:
    st.image("https://img.icons8.com/fluency/96/000000/machine-learning.png", width=100)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Upload",
    "🔧 Feature Engineering",
    "🤖 Model Training",
    "📊 Drift Monitoring",
    "📈 Results & Analytics"
])

# ==================== Tab 1: Data Upload ====================

with tab1:
    st.header("📁 Data Upload & Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Supported formats: CSV, JSON, Excel"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.data is not None:
            create_metric_card("Rows", f"{st.session_state.data.shape[0]:,}")
            create_metric_card("Columns", f"{st.session_state.data.shape[1]:,}")
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            
            st.success(f"✅ Successfully loaded {df.shape[0]:,} rows and {df.shape[1]:,} columns!")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Column Types")
                type_counts = df.dtypes.value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index.astype(str),
                    title="Data Types Distribution",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Missing Values")
                missing = df.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=False)
                
                if len(missing) > 0:
                    fig = px.bar(
                        x=missing.values,
                        y=missing.index,
                        orientation='h',
                        title="Missing Values by Column",
                        labels={'x': 'Count', 'y': 'Column'},
                        color=missing.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values detected!")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# ==================== Tab 2: Feature Engineering ====================

with tab2:
    st.header("🔧 Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
    else:
        st.subheader("🎯 Select Target Column")
        target_column = st.selectbox(
            "Target Column",
            options=st.session_state.data.columns.tolist(),
            help="Select the column you want to predict"
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🔨 Transformation Pipeline")
            if auto_features:
                st.info("✨ **Auto Mode Enabled** - Pipeline will automatically select and apply optimal transformations")
            else:
                st.info(f"🎛️ **Manual Mode** - Selected transformations: {', '.join(selected_transforms)}")
        
        with col2:
            if st.button("▶️ Run Feature Engineering", use_container_width=True):
                with st.spinner("🔄 Engineering features..."):
                    try:
                        # Ingest data
                        st.session_state.pipeline.ingest(st.session_state.data)
                        
                        # Engineer features
                        result = st.session_state.pipeline.engineer_features(
                            auto=auto_features,
                            target_column=target_column
                        )
                        
                        st.session_state.processed_data = result['transformed_data']
                        
                        st.success(f"✅ Feature engineering completed! Created {result['features_created']} new features")
                        
                        # Show transformation summary
                        st.subheader("📊 Transformation Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            create_metric_card("Original Features", st.session_state.data.shape[1] - 1)
                        with col2:
                            create_metric_card("New Features", result['features_created'])
                        with col3:
                            create_metric_card("Total Features", st.session_state.processed_data.shape[1] - 1)
                    
                    except Exception as e:
                        st.error(f"❌ Feature engineering failed: {str(e)}")
        
        # Show processed data preview
        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.subheader("📋 Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)

# ==================== Tab 3: Model Training ====================

with tab3:
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🎯 Training Configuration")
            st.info(f"**Model:** {model_type} | **Validation Split:** {validation_split:.0%}")
        
        with col2:
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("🔄 Training model..."):
                    try:
                        # Update config
                        st.session_state.pipeline.config.model_type = model_type
                        st.session_state.pipeline.config.validation_split = validation_split
                        
                        # Get target column
                        if 'target_column' not in st.session_state:
                            st.error("❌ Please run feature engineering first to select target column")
                        else:
                            # Run full pipeline
                            results = st.session_state.pipeline.run_full_pipeline(
                                source=st.session_state.data,
                                target_column=target_column,
                                auto_features=auto_features,
                                check_drift=False,
                                train_model=True
                            )
                            
                            st.session_state.training_results = results
                            st.session_state.model_trained = True
                            
                            st.success("✅ Model training completed successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        # Show training results
        if st.session_state.training_results is not None:
            st.markdown("---")
            st.subheader("📊 Training Results")
            
            metrics = st.session_state.training_results.get('model_metrics', {})
            
            if metrics:
                # Metrics visualization
                fig = plot_model_metrics(metrics)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Metrics table
                st.subheader("📈 Detailed Metrics")
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['Value']
                metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
                st.dataframe(metrics_df, use_container_width=True)
                
                # Feature importance
                if hasattr(st.session_state.pipeline.model, 'feature_importances_'):
                    st.markdown("---")
                    st.subheader("🎯 Feature Importance")
                    
                    importances = st.session_state.pipeline.model.feature_importances_
                    feature_names = st.session_state.processed_data.drop(columns=[target_column]).columns.tolist()
                    
                    fig = plot_feature_importance(feature_names, importances)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== Tab 4: Drift Monitoring ====================

with tab4:
    st.header("📊 Data Drift Monitoring")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 Upload New Data for Drift Check")
            new_data_file = st.file_uploader(
                "Upload new dataset to check for drift",
                type=['csv', 'json', 'xlsx', 'xls'],
                key="drift_upload"
            )
        
        with col2:
            set_reference = st.checkbox("Set as Reference Data", value=False)
            
            if st.button("🔍 Check Drift", use_container_width=True):
                if new_data_file is None:
                    st.error("❌ Please upload a dataset first")
                else:
                    with st.spinner("🔄 Analyzing drift..."):
                        try:
                            # Load new data
                            if new_data_file.name.endswith('.csv'):
                                new_df = pd.read_csv(new_data_file)
                            elif new_data_file.name.endswith('.json'):
                                new_df = pd.read_json(new_data_file)
                            else:
                                new_df = pd.read_excel(new_data_file)
                            
                            # Check drift
                            drift_report = st.session_state.pipeline.monitor_drift(
                                current_data=new_df,
                                set_as_reference=set_reference
                            )
                            
                            st.session_state.drift_report = drift_report
                            
                            if drift_report.overall_drift_detected:
                                st.error("🚨 **Drift Detected!** Model retraining recommended")
                            else:
                                st.success("✅ **No Significant Drift** Data distribution is stable")
                        
                        except Exception as e:
                            st.error(f"❌ Drift check failed: {str(e)}")
        
        # Show drift results
        if st.session_state.drift_report is not None:
            st.markdown("---")
            
            drift_report = st.session_state.drift_report
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            drifted_cols = [col for col, result in drift_report.column_results.items() if result.drift_detected]
            
            with col1:
                create_metric_card("Columns Checked", len(drift_report.column_results))
            with col2:
                create_metric_card("Drifted Columns", len(drifted_cols))
            with col3:
                drift_pct = len(drifted_cols) / len(drift_report.column_results) * 100
                create_metric_card("Drift %", f"{drift_pct:.1f}%")
            with col4:
                status = "⚠️ Retrain" if st.session_state.pipeline.drift_monitor.should_retrain() else "✅ OK"
                create_metric_card("Status", status)
            
            # Drift visualization
            st.markdown("---")
            st.subheader("📊 Drift Analysis")
            
            fig = plot_drift_results(drift_report)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Drifted columns details
            if drifted_cols:
                st.markdown("---")
                st.subheader("🚨 Drifted Columns Details")
                
                drift_details = []
                for col in drifted_cols:
                    result = drift_report.column_results[col]
                    drift_details.append({
                        "Column": col,
                        "P-Value": f"{result.p_value:.6f}",
                        "Statistic": f"{result.statistic:.4f}",
                        "Method": result.test_method
                    })
                
                st.dataframe(pd.DataFrame(drift_details), use_container_width=True)

# ==================== Tab 5: Results & Analytics ====================

with tab5:
    st.header("📈 Results & Analytics")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")
    else:
        # Overall summary
        st.subheader("📊 Pipeline Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Dataset Size", f"{st.session_state.data.shape[0]:,}")
        with col2:
            create_metric_card("Features", f"{st.session_state.processed_data.shape[1] - 1}")
        with col3:
            create_metric_card("Model Type", model_type.replace('_', ' ').title())
        with col4:
            if st.session_state.training_results:
                best_metric = max(st.session_state.training_results['model_metrics'].values())
                create_metric_card("Best Score", f"{best_metric:.3f}")
        
        st.markdown("---")
        
        # Download section
        st.subheader("💾 Export & Download")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Processed Data", use_container_width=True):
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download Model Report", use_container_width=True):
                report = {
                    "model_type": model_type,
                    "metrics": st.session_state.training_results['model_metrics'],
                    "timestamp": datetime.now().isoformat()
                }
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(report, indent=2),
                    file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.session_state.drift_report:
                if st.button("📥 Download Drift Report", use_container_width=True):
                    drift_data = {
                        "overall_drift": st.session_state.drift_report.overall_drift_detected,
                        "recommendation": st.session_state.drift_report.recommendation,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.download_button(
                        label="💾 Download JSON",
                        data=json.dumps(drift_data, indent=2),
                        file_name=f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

# ==================== Footer ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <h3>🚀 ML Pipeline Pro v2.0</h3>
    <p>Production-Ready MLOps Platform | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
