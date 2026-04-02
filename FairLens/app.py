"""
FairLens Streamlit Dashboard

İnteractive fairness denetim arayüzü. Kullanıcılar CSV dosyaları yükleyebilir,
korumalı öznitelikleri seçebilir ve adaletsizlik metriklerini gerçek zamanda
görselleri ile görebilir.

Yazarlar: FairLens Geliştirme Ekibi
Sürüm: 1.0.0
Lisans: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import traceback
from PIL import Image
import io

# Import FairLens modules
from fairness_engine import FairnessAuditor, MetricThresholds
from visualization_utils import (
    create_outcome_rates_chart,
    create_bias_trend_chart,
    create_disparate_impact_gauge,
    create_equity_heatmap,
    create_risk_summary_table,
    format_metric_for_display,
    get_risk_color
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FairLens | Bias Detection Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-low { background-color: #d4edda; border-left: 4px solid #28a745; }
    .risk-medium { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .risk-high { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    .header-section {
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'auditor' not in st.session_state:
        st.session_state.auditor = None
    if 'audit_result' not in st.session_state:
        st.session_state.audit_result = None


init_session_state()


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render the sidebar with file upload and configuration options."""
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # File uploader
    st.sidebar.markdown("#### 📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your dataset. Must contain columns for the target label and protected attributes."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = df
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            return None, None, None, None, None
    
    if st.session_state.df is None:
        st.sidebar.warning("Please upload a CSV file to begin")
        return None, None, None, None, None
    
    df = st.session_state.df
    
    # Column selectors
    st.sidebar.markdown("#### 🎯 Column Selection")
    
    available_cols = list(df.columns)
    
    # Target/Label column
    label_col = st.sidebar.selectbox(
        "Target Variable (Label/Outcome)",
        options=available_cols,
        help="The outcome variable you want to audit (e.g., 'hired', 'approved')"
    )
    
    # Protected Attribute column
    protected_attrs = [col for col in available_cols if col != label_col]
    protected_attr = st.sidebar.selectbox(
        "Protected Attribute",
        options=protected_attrs,
        help="The sensitive attribute (e.g., 'gender', 'race')"
    )
    
    # Time column (optional)
    time_cols = ['None'] + available_cols
    time_col = st.sidebar.selectbox(
        "Time Column (Optional)",
        options=time_cols,
        help="Select for temporal/panel data analysis (e.g., 'year', 'quarter')"
    )
    time_col = None if time_col == 'None' else time_col
    
    # Metric Thresholds (optional advanced settings)
    with st.sidebar.expander("📊 Advanced Metric Thresholds"):
        dpd_threshold = st.slider(
            "Demographic Parity Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Lower DPD is better (closer to 0)"
        )
        eod_threshold = st.slider(
            "Equalized Odds Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Lower EOD is better (closer to 0)"
        )
        dir_threshold = st.slider(
            "Disparate Impact Threshold (80% Rule)",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Higher DIR is better (≥ 0.8 passes 80% rule)"
        )
        
        thresholds = MetricThresholds(
            demographic_parity_threshold=dpd_threshold,
            equalized_odds_threshold=eod_threshold,
            disparate_impact_threshold=dir_threshold
        )
    
    # Dataset info
    st.sidebar.markdown("#### 📈 Dataset Info")
    st.sidebar.metric("Rows", len(df))
    st.sidebar.metric("Columns", len(df.columns))
    
    # Display sample data
    if st.sidebar.checkbox("Show Sample Data", value=False):
        st.sidebar.markdown("**Sample Data (first 5 rows):**")
        st.sidebar.dataframe(df.head(), use_container_width=True)
    
    return df, label_col, protected_attr, time_col, thresholds


# ============================================================================
# MAIN AUDIT LOGIC
# ============================================================================

def run_audit(
    df: pd.DataFrame,
    label_col: str,
    protected_attr: str,
    thresholds: MetricThresholds
) -> Tuple[Optional[FairnessAuditor], Optional[dict], Optional[Exception], Optional[List[str]]]:
    """Run the fairness audit and return auditor, result, any error, and preprocessing warnings."""
    try:
        # Validate inputs
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset")
        if protected_attr not in df.columns:
            raise ValueError(f"Protected attribute '{protected_attr}' not found in dataset")
        
        # Create auditor (preprocessing happens during __init__)
        auditor = FairnessAuditor(
            df=df,
            label_name=label_col,
            protected_attributes=[protected_attr],
            thresholds=thresholds
        )
        
        # Get preprocessing warnings before running audit
        preprocessing_warnings = auditor.get_preprocessing_warnings()
        
        # Run audit
        result = auditor.generate_report(as_json=False)
        
        return auditor, result, None, preprocessing_warnings
    
    except Exception as e:
        return None, None, e, None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application logic."""
    
    # Header
    st.markdown("<h1>⚖️ FairLens: Algorithmic Bias Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "Detect and audit algorithmic bias in your datasets before deployment. "
        "FairLens provides industry-standard fairness metrics and risk assessment.",
        help="Powered by AIF360 and cutting-edge fairness research"
    )
    st.divider()
    
    # Render sidebar
    df, label_col, protected_attr, time_col, thresholds = render_sidebar()
    
    # Main content
    if df is None:
        st.info(
            "📁 **Please upload a CSV file in the sidebar** to begin your fairness audit.\n\n"
            "**Example Dataset Structure:**\n"
            "| gender | age | experience | test_score | hired |\n"
            "|--------|-----|------------|------------|-------|\n"
            "| M      | 35  | 5          | 85         | 1     |\n"
            "| F      | 32  | 6          | 88         | 0     |"
        )
        return
    
    # Run audit
    st.markdown("<div class='header-section'><h3>🧪 Running Fairness Audit...</h3></div>", unsafe_allow_html=True)
    
    auditor, audit_result, error, preprocessing_warnings = run_audit(df, label_col, protected_attr, thresholds)
    
    if error:
        st.error(f"❌ **Audit Error**: {str(error)}")
        st.info(
            "**Troubleshooting Tips:**\n"
            "- Ensure the label column contains binary values (0/1 or Yes/No)\n"
            "- Ensure the protected attribute column is categorical (e.g., gender, race)\n"
            "- Remove any special characters from column names"
        )
        return
    
    st.session_state.audit_result = audit_result
    st.session_state.auditor = auditor
    
    # ====================================================================
    # DISPLAY PREPROCESSING INFORMATION
    # ====================================================================
    
    if preprocessing_warnings:
        st.divider()
        with st.expander("📋 Data Preprocessing Summary", expanded=True):
            st.info("**Data transformations applied:**")
            for warning in preprocessing_warnings:
                st.write(f"• {warning}")
    
    # ====================================================================
    # DISPLAY METRICS
    # ====================================================================
    
    st.success("✅ Audit completed successfully!")
    st.divider()
    
    st.markdown("<h2>📊 Fairness Metrics</h2>", unsafe_allow_html=True)
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dpd = audit_result['demographic_parity_difference']
        st.metric(
            label="Demographic Parity",
            value=f"{dpd:.4f}",
            delta=f"Threshold: {thresholds.demographic_parity_threshold:.4f}",
            delta_color="inverse"
        )
        if dpd > thresholds.demographic_parity_threshold:
            st.warning("⚠️ Exceeds threshold")
        else:
            st.success("✓ Within threshold")
    
    with col2:
        eod = audit_result['equalized_odds_difference']
        st.metric(
            label="Equalized Odds",
            value=f"{eod:.4f}",
            delta=f"Threshold: {thresholds.equalized_odds_threshold:.4f}",
            delta_color="inverse"
        )
        if eod > thresholds.equalized_odds_threshold:
            st.warning("⚠️ Exceeds threshold")
        else:
            st.success("✓ Within threshold")
    
    with col3:
        dir_score = audit_result['disparate_impact_ratio']
        st.metric(
            label="Disparate Impact Ratio",
            value=f"{dir_score:.4f}",
            delta=f"Threshold: {thresholds.disparate_impact_threshold:.4f}"
        )
        if dir_score < thresholds.disparate_impact_threshold:
            st.warning("⚠️ Below 80% rule threshold")
        else:
            st.success("✓ Passes 80% rule")
    
    with col4:
        risk_level = audit_result['risk_level']
        risk_color = get_risk_color(risk_level)
        st.metric(label="Risk Level", value=risk_level)
        if 'High' in risk_level or 'Yüksek' in risk_level:
            st.error("🔴 **HIGH RISK** - Significant bias detected")
        elif 'Medium' in risk_level or 'Orta' in risk_level:
            st.warning("🟡 **MEDIUM RISK** - Some fairness concerns")
        else:
            st.success("🟢 **LOW RISK** - Dataset appears fair")
    
    st.divider()
    
    # ====================================================================
    # VISUALIZATIONS
    # ====================================================================
    
    st.markdown("<h2>📈 Visualizations</h2>", unsafe_allow_html=True)
    
    # Row 1: Outcome Rates & Disparate Impact Gauge
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Outcome Rates by Group")
        try:
            fig_outcomes = create_outcome_rates_chart(df, label_col, protected_attr)
            st.plotly_chart(fig_outcomes, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating outcome rates chart: {str(e)}")
    
    with viz_col2:
        st.subheader("Disparate Impact Gauge (80% Rule)")
        try:
            fig_gauge = create_disparate_impact_gauge(dir_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating gauge chart: {str(e)}")
    
    # Row 2: Equity Heatmap
    st.subheader("Equity Analysis Heatmap")
    try:
        fig_heatmap = create_equity_heatmap(df, label_col, protected_attr)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
    
    # ====================================================================
    # TEMPORAL ANALYSIS (if time column is provided)
    # ====================================================================
    
    if time_col:
        st.divider()
        st.markdown("<h2>⏱️ Temporal Analysis (Panel Data)</h2>", unsafe_allow_html=True)
        
        try:
            temporal_df = auditor.temporal_bias_analysis(df, time_col)
            
            st.info(
                f"📊 Analyzing {len(temporal_df)} time periods. "
                "This shows how fairness metrics have evolved over time."
            )
            
            # Temporal trend chart
            st.subheader("Bias Metrics Over Time")
            fig_trend = create_bias_trend_chart(temporal_df)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Temporal data table
            st.subheader("Temporal Data Table")
            st.dataframe(temporal_df, use_container_width=True)
            
            # Download temporal data
            csv = temporal_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Temporal Data (CSV)",
                data=csv,
                file_name=f"fairlens_temporal_analysis_{time_col}.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Error in temporal analysis: {str(e)}")
            st.info(
                "**Troubleshooting:**\n"
                f"- Ensure '{time_col}' contains valid time period values\n"
                "- Each time period should have sufficient samples for metrics calculation\n"
                f"- Current data shape: {df.shape}"
            )
    
    st.divider()
    
    # ====================================================================
    # METRICS SUMMARY TABLE
    # ====================================================================
    
    st.markdown("<h2>📋 Metrics Summary</h2>", unsafe_allow_html=True)
    
    try:
        summary_table = create_risk_summary_table(audit_result['metric_details'], temporal_df if time_col else None)
        st.dataframe(summary_table, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating summary table: {str(e)}")
    
    # ====================================================================
    # DOWNLOAD REPORT
    # ====================================================================
    
    st.divider()
    st.markdown("<h2>📄 Export & Download</h2>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # JSON export
        import json
        json_report = json.dumps(audit_result, indent=2, default=str)
        st.download_button(
            label="📥 Download Report (JSON)",
            data=json_report,
            file_name="fairlens_audit_report.json",
            mime="application/json"
        )
    
    with col_d2:
        # CSV export of processed data
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name="fairlens_processed_data.csv",
            mime="text/csv"
        )
    
    # ====================================================================
    # RECOMMENDATIONS
    # ====================================================================
    
    st.divider()
    st.markdown("<h2>💡 Recommendations</h2>", unsafe_allow_html=True)
    
    recommendations = []
    
    if audit_result['demographic_parity_difference'] > thresholds.demographic_parity_threshold:
        recommendations.append(
            "🔴 **High Demographic Parity Difference**: The selection rates between groups are significantly different. "
            "Consider reviewing your decision-making criteria or collecting more representative data."
        )
    
    if audit_result['equalized_odds_difference'] > thresholds.equalized_odds_threshold:
        recommendations.append(
            "🔴 **High Equalized Odds Difference**: The model has different error rates for different groups. "
            "This suggests the algorithm performs differently for protected groups."
        )
    
    if audit_result['disparate_impact_ratio'] < 0.8:
        recommendations.append(
            "🔴 **Disparate Impact Violation (80% Rule)**: The selection rate for the protected group is less than "
            "80% of the privileged group. This may violate employment law in some jurisdictions."
        )
    
    if not recommendations:
        recommendations.append(
            "🟢 **No major fairness concerns detected**: Your dataset appears to pass standard fairness audits. "
            "However, continue monitoring for emerging bias patterns."
        )
    
    for rec in recommendations:
        if "🔴" in rec:
            st.error(rec)
        elif "🟡" in rec:
            st.warning(rec)
        else:
            st.success(rec)
    
    # ====================================================================
    # FOOTER
    # ====================================================================
    
    st.divider()
    st.markdown(
        "---\n"
        "**FairLens v1.0** | Built with ⚖️ for responsible AI\n"
        "[GitHub](https://github.com) | [Documentation](https://fairlens.dev) | [License: MIT](https://opensource.org/licenses/MIT)"
    )


# ============================================================================
# APP ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
