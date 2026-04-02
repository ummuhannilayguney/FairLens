#!/usr/bin/env python
"""
FairLens Quick Start Examples

This script demonstrates the new features added in v2.0.0:
1. Temporal Bias Analysis (Panel Data)
2. Streamlit Dashboard Usage
3. Enhanced Visualizations

Run this script to verify everything is working:
    python examples.py
"""

import sys
import pandas as pd
import numpy as np
from fairness_engine import FairnessAuditor, MetricThresholds
from visualization_utils import (
    create_outcome_rates_chart,
    create_disparate_impact_gauge,
    create_bias_trend_chart,
    create_equity_heatmap
)

# Set seed for reproducibility
np.random.seed(42)


def create_sample_dataset_with_time(n_samples=1000, n_years=3):
    """
    Create a sample hiring dataset with temporal dimension for demonstration.
    
    Features:
    - Multiple time periods (years)
    - Gender as protected attribute
    - Hiring outcomes
    """
    data = []
    
    # Slightly increasing bias over time to demonstrate trend
    bias_factors = [1.0, 1.15, 1.3]  # Bias increases over years
    
    for year_idx, year in enumerate(range(2021, 2021 + n_years)):
        bias_factor = bias_factors[year_idx]
        
        for _ in range(n_samples // n_years):
            gender = np.random.choice([0, 1])  # 0=Female, 1=Male
            age = np.random.normal(40, 10)
            experience = np.random.exponential(5)
            test_score = np.random.normal(75, 15)
            
            # Hiring decision with embedded bias
            base_score = (age / 100 + experience / 20 + test_score / 100)
            
            # Apply gender bias (males get preference in later years)
            if gender == 0:  # Female
                adjusted_score = base_score / bias_factor
            else:  # Male
                adjusted_score = base_score * (1 + (bias_factor - 1) * 0.5)
            
            hired = 1 if adjusted_score > np.percentile(base_score, 40) else 0
            
            data.append({
                'gender': gender,
                'age': age,
                'experience': experience,
                'test_score': test_score,
                'hired': hired,
                'year': year
            })
    
    return pd.DataFrame(data)


def example_1_temporal_analysis():
    """
    Example 1: Temporal Bias Analysis (NEW in v2.0.0)
    
    Learn how bias metrics evolve over time using Panel Data approach.
    Useful for monitoring - Are we improving or getting worse?
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: TEMPORAL BIAS ANALYSIS (Panel Data)")
    print("=" * 70)
    
    # Create sample data with time dimension
    df = create_sample_dataset_with_time(n_samples=1000, n_years=3)
    
    print(f"\nDataset: {len(df)} hiring decisions across {df['year'].nunique()} years")
    print(f"Columns: {list(df.columns)}\n")
    
    # Create auditor
    auditor = FairnessAuditor(
        df=df,
        label_name='hired',
        protected_attributes=['gender']
    )
    
    # Run temporal analysis
    print("Running temporal analysis...")
    temporal_results = auditor.temporal_bias_analysis(df, 'year')
    
    # Display results
    print("\nTemporal Bias Analysis Results:")
    print("-" * 70)
    print(temporal_results.to_string(index=False))
    print("-" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    for _, row in temporal_results.iterrows():
        print(f"\nYear {row['time_period']} (n={row['sample_size']}):")
        print(f"  • Disparate Impact: {row['disparate_impact_ratio']:.4f} ", end="")
        if row['disparate_impact_ratio'] >= 0.8:
            print("✓ PASS (≥ 80% Rule)")
        else:
            print("✗ FAIL (Violates 80% Rule)")
        
        print(f"  • Demographic Parity: {row['demographic_parity_difference']:.4f}", end="")
        print(" (" + ("✓ Good" if row['demographic_parity_difference'] < 0.1 else "✗ High") + ")")
        print(f"  • Risk Level: {row['risk_level']}")
    
    # Save for later visualization
    temporal_results_file = 'temporal_analysis_results.csv'
    temporal_results.to_csv(temporal_results_file, index=False)
    print(f"\n💾 Results saved to: {temporal_results_file}")
    
    return df, auditor, temporal_results


def example_2_dashboard_walkthrough():
    """
    Example 2: Streamlit Dashboard Walkthrough (NEW in v2.0.0)
    
    This explains how to use the new interactive web dashboard.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: STREAMLIT DASHBOARD WALKTHROUGH")
    print("=" * 70)
    
    print("\n📊 How to Use the FairLens Interactive Dashboard:\n")
    
    print("1. INSTALLATION")
    print("   " + "-" * 60)
    print("   pip install -r requirements.txt")
    print()
    
    print("2. START THE APP")
    print("   " + "-" * 60)
    print("   streamlit run app.py")
    print("   → Opens automatically in your browser at http://localhost:8501")
    print()
    
    print("3. UPLOAD DATA (Sidebar)")
    print("   " + "-" * 60)
    print("   • Click 'Choose a CSV file'")
    print("   • Select your hiring/approval data")
    print("   • Required columns: at least target label + protected attribute")
    print()
    
    print("4. CONFIGURE COLUMNS (Sidebar)")
    print("   " + "-" * 60)
    print("   • Select 'Target Variable' (e.g., 'hired', 'approved')")
    print("   • Select 'Protected Attribute' (e.g., 'gender', 'race')")
    print("   • (Optional) Select 'Time Column' for temporal analysis (e.g., 'year')")
    print()
    
    print("5. ADJUST THRESHOLDS (Optional - Sidebar)")
    print("   " + "-" * 60)
    print("   • Expand 'Advanced Metric Thresholds'")
    print("   • Customize DPD, EOD, DIR thresholds")
    print("   • Default values use industry standards")
    print()
    
    print("6. VIEW RESULTS (Main Panel)")
    print("   " + "-" * 60)
    print("   ✓ 4 Key Metrics with Color Coding:")
    print("     - 🟢 Green = Within threshold (LOW RISK)")
    print("     - 🟡 Yellow = Marginal violation")
    print("     - 🔴 Red = High violation (HIGH RISK)")
    print()
    print("   ✓ 3 Interactive Charts:")
    print("     - Bar Chart: Outcome rates by group")
    print("     - Gauge Chart: Disparate Impact score")
    print("     - Heatmap: Hiring decision distribution")
    print()
    print("   ✓ Temporal Trend (if time column selected):")
    print("     - Line chart tracking bias over time periods")
    print("     - Reference line at 80% rule threshold")
    print()
    
    print("7. DOWNLOAD REPORTS")
    print("   " + "-" * 60)
    print("   • JSON Report: Full audit with all details")
    print("   • CSV Dataset: Processed data for analysis")
    print("   • CSV Temporal: Time-series metrics (if applicable)")
    print()
    
    print("8. RECOMMENDATIONS")
    print("   " + "-" * 60)
    print("   The dashboard provides actionable suggestions:")
    print("   • 🔴 Critical: Significant fairness violations")
    print("   • 🟡 Warning: Potential bias concerns")
    print("   • 🟢 Clear: Dataset passes fairness audit")
    print()


def example_3_visualizations(df, auditor):
    """
    Example 3: Visualization Functions (NEW in v2.0.0)
    
    Demonstrates how to create Plotly visualizations programmatically.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: VISUALIZATION FUNCTIONS")
    print("=" * 70)
    
    print("\nAvailable visualization functions:\n")
    
    print("1. create_outcome_rates_chart()")
    print("   Shows hiring/approval rates by protected group")
    try:
        fig = create_outcome_rates_chart(df, 'hired', 'gender')
        print("   ✓ Generated successfully")
        print("   → Show with: fig.show() or st.plotly_chart(fig)")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n2. create_disparate_impact_gauge()")
    print("   Needle gauge showing Disparate Impact Ratio")
    try:
        # Get the actual DIR from audit
        audit_result = auditor.generate_report(as_json=False)
        dir_score = audit_result['disparate_impact_ratio']
        fig = create_disparate_impact_gauge(dir_score)
        print(f"   ✓ Generated successfully (DIR={dir_score:.4f})")
        print("   → Show with: fig.show() or st.plotly_chart(fig)")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n3. create_equity_heatmap()")
    print("   Heatmap showing hiring decisions by gender")
    try:
        fig = create_equity_heatmap(df, 'hired', 'gender')
        print("   ✓ Generated successfully")
        print("   → Show with: fig.show() or st.plotly_chart(fig)")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n4. create_bias_trend_chart()")
    print("   Multi-axis line chart for temporal trends")
    try:
        temporal_df = auditor.temporal_bias_analysis(df, 'year')
        fig = create_bias_trend_chart(temporal_df)
        print("   ✓ Generated successfully")
        print("   → Show with: fig.show() or st.plotly_chart(fig)")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n💡 All charts are interactive:")
    print("   • Hover for exact values")
    print("   • Zoom/pan with mouse")
    print("   • Download plots as PNG")
    print("   • Toggle series on/off")


def example_4_programmatic_audit():
    """
    Example 4: Programmatic Audit (Traditional Usage)
    
    Shows how to use FairLens programmatically without the dashboard.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: PROGRAMMATIC AUDIT (Command Line)")
    print("=" * 70)
    
    # Create data
    df = create_sample_dataset_with_time(n_samples=1000, n_years=1)
    df = df[df['year'] == 2021].reset_index(drop=True)  # Just one year
    
    # Standard audit
    print("\nRunning standard fairness audit...")
    auditor = FairnessAuditor(
        df=df,
        label_name='hired',
        protected_attributes=['gender']
    )
    
    report = auditor.generate_report(as_json=False)
    
    print("\n📊 AUDIT REPORT:")
    print("-" * 70)
    print(f"Risk Level: {report['risk_level']}")
    print(f"Demographic Parity Difference: {report['demographic_parity_difference']:.4f}")
    print(f"Equalized Odds Difference: {report['equalized_odds_difference']:.4f}")
    print(f"Disparate Impact Ratio: {report['disparate_impact_ratio']:.4f}")
    print(f"Sample Size: {report['metric_details']['sample_size']}")
    
    # Quick audit
    print("\n⚡ QUICK AUDIT (Essential metrics only):")
    print("-" * 70)
    quick_result = auditor.quick_audit()
    for key, value in quick_result.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")


def example_5_custom_thresholds():
    """
    Example 5: Custom Thresholds
    
    Shows how to adjust fairness thresholds for your specific context.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: CUSTOM METRIC THRESHOLDS")
    print("=" * 70)
    
    df = create_sample_dataset_with_time(n_samples=500, n_years=1)
    df = df[df['year'] == 2021].reset_index(drop=True)
    
    print("\n📋 THRESHOLD SCENARIOS:\n")
    
    scenarios = [
        {
            "name": "Strict (Research Standard)",
            "dpd": 0.05,
            "eod": 0.05,
            "dir": 0.95
        },
        {
            "name": "Moderate (Industry Standard)",
            "dpd": 0.10,
            "eod": 0.10,
            "dir": 0.80
        },
        {
            "name": "Lenient (Policy Minimums)",
            "dpd": 0.15,
            "eod": 0.15,
            "dir": 0.70
        }
    ]
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  DPD threshold:  {scenario['dpd']}")
        print(f"  EOD threshold:  {scenario['eod']}")
        print(f"  DIR threshold:  {scenario['dir']} (≥ {scenario['dir']*100:.0f}% Rule)")
        
        thresholds = MetricThresholds(
            demographic_parity_threshold=scenario['dpd'],
            equalized_odds_threshold=scenario['eod'],
            disparate_impact_threshold=scenario['dir']
        )
        
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender'],
            thresholds=thresholds
        )
        
        report = auditor.generate_report(as_json=False)
        print(f"  Result: {report['risk_level']}\n")


def main():
    """Main entry point for examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "FairLens v2.0.0 - Quick Start Examples" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Run examples
        df, auditor, temporal_results = example_1_temporal_analysis()
        example_2_dashboard_walkthrough()
        example_3_visualizations(df, auditor)
        example_4_programmatic_audit()
        example_5_custom_thresholds()
        
        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 NEXT STEPS:")
        print("1. Run 'streamlit run app.py' to start the interactive dashboard")
        print("2. Integrate FairLens into your ML pipeline: from fairness_engine import FairnessAuditor")
        print("3. Monitor fairness over time with temporal_bias_analysis()")
        print("4. Check .github/workflows/python-tests.yml for CI/CD integration")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
