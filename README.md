# FairLens: Proactive Algorithmic Bias Auditing Framework

**Version:** 2.0.0 | **Last Updated:** April 2, 2026 | **Status:** ✅ Production Ready

![FairLens Badge](https://img.shields.io/badge/Python-3.9%2B-blue) ![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen) ![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary

**FairLens** is a comprehensive Python framework for detecting and analyzing algorithmic bias in tabular datasets. With both programmatic and interactive web interfaces, FairLens enables responsible AI teams to:

- 🔍 **Detect bias early** at the data-sourcing stage before model training
- 📊 **Visualize fairness metrics** interactively through a web dashboard
- 📈 **Track bias trends** over time with temporal analysis capabilities
- 📋 **Generate compliance reports** with industry-standard fairness metrics
- 🎯 **Assess risk levels** with automated recommendations

### What's New in v2.0.0
- ✨ **Interactive Streamlit Dashboard** with real-time bias visualization
- ⏱️ **Temporal Bias Analysis** for tracking fairness metrics over time periods
- 📊 **Advanced Visualizations** with Plotly interactive charts
- 🔄 **Automated CI/CD** with GitHub Actions testing pipeline
- 📚 **Comprehensive Documentation** with step-by-step implementation guide

---

## 1. Architecture Overview

### 1.1 System Design

```
┌────────────────────────────────────────────────────────────────┐
│                    FairLens Ecosystem v2.0                     │
└────────────────────────────┬───────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐       ┌──────────────┐
   │Core      │         │Dashboard │       │Visualizations│
   │Framework │         │(Streamlit)       │(Plotly)      │
   └──────────┘         └──────────┘       └──────────────┘
        │                    │                    │
        │         ┌──────────┴────────────┐       │
        │         │                       │       │
        ▼         ▼                       ▼       ▼
   ┌─────────────────────────────────────────────────────┐
   │           Fairness Auditor Engine                   │
   │  - Temporal Bias Analysis                           │
   │  - Risk Assessment & Classification                 │
   │  - Multi-metric Evaluation                          │
   └─────────────────────────────────────────────────────┘
        │           │                    │
        ▼           ▼                    ▼
   ┌─────────┐ ┌──────────┐         ┌─────────┐
   │Tabular  │ │Temporal  │         │Metrics  │
   │Auditor  │ │Analysis  │         │Calc.    │
   └─────────┘ └──────────┘         └─────────┘
        │
        ▼
   ┌──────────────────────────────┐
   │ aif360 StandardDataset       │
   │ Logistic Regression Proxy    │
   └──────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **FairnessAuditor** | Public API & Orchestration | Coordinates auditing; temporal analysis; JSON reports |
| **TabularAuditor** | Core bias detection | Pandas DataFrames; aif360 integration; validation |
| **BaseAuditor** | Abstract contract | Interface definition; extensibility |
| **MetricsCalculator** | Fairness computation | Static methods; DPD, EOD, DIR metrics |
| **Temporal Analysis** | Time-series tracking | Period-based bias trends; risk evolution |
| **Dashboard (app.py)** | Interactive web UI | Real-time visualization; file upload; export |
| **Visualization Utils** | Chart generation | Outcome rates; disparate impact gauges; heatmaps |
| **RiskAssessment** | Risk classification | Automated risk mapping; recommendations |

---

## 2. Quick Start Guide

### 2.1 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FairLens.git
cd FairLens

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Launch Interactive Dashboard

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` with:
- 📁 CSV file upload and validation
- 🎯 Column selection for analysis
- 📊 Real-time bias metrics visualization
- 📈 Temporal trend analysis (optional)
- 📋 JSON report generation
- 💾 Export options (CSV, JSON)

### 2.3 Programmatic Usage

```python
from fairness_engine import FairnessAuditor
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create auditor
auditor = FairnessAuditor(
    df=df,
    label_name='outcome_column',
    protected_attributes=['protected_group_column']
)

# Generate comprehensive report
report = auditor.generate_report(as_json=False)
print(f"Risk Level: {report['risk_level']}")

# Or export temporal analysis
temporal_data = auditor.temporal_bias_analysis(df, time_column='year')
temporal_data.to_csv('bias_trends.csv', index=False)
```

### 2.4 Code Examples

Explore [QUICKSTART_EXAMPLES.py](QUICKSTART_EXAMPLES.py) for 5 runnable examples:
1. Basic bias audit
2. Custom thresholds configuration
3. Multiple protected attributes
4. Temporal trend analysis
5. JSON report generation

---

## 3. Fairness Metrics Implementation

### 3.1 Demographic Parity Difference (DPD)

**Formula:** $|P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)|$

**Interpretation:**
- Measures the difference in positive prediction rates between protected and privileged groups
- **Lower is better**: 0 means perfect parity
- **Threshold**: < 0.1 is considered LOW risk
- **Use case**: Loan approvals, hiring decisions

**Example:**
```
Group 0 (protected): 60% approval rate
Group 1 (privileged): 70% approval rate
DPD = |0.60 - 0.70| = 0.10 (at threshold)
```

### 3.2 Equalized Odds Difference (EOD)

**Formula:** $\max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)$

**Interpretation:**
- Measures max difference in True Positive Rate (TPR) and False Positive Rate (FPR)
- Ensures equal error rates across groups
- **Lower is better**: 0 means perfectly equalized odds
- **Threshold**: < 0.1 is considered LOW risk
- **Use case**: Credit scoring, recidivism prediction

**Example:**
```
Group 0: TPR = 0.85, FPR = 0.15
Group 1: TPR = 0.90, FPR = 0.10
EOD = max(|0.85-0.90|, |0.15-0.10|) = 0.05 (LOW risk)
```

### 3.3 Disparate Impact Ratio (DIR) - The 80% Rule

**Formula:** $\frac{\text{Selection Rate (Protected)}}{\text{Selection Rate (Privileged)}}$

**Interpretation:**
- US EEOC standard: ratio ≥ 0.8 is legally acceptable
- **Ratio = 1.0**: Perfect parity
- **Ratio < 0.8**: Disparate impact (HIGH risk)
- **Ratio < 0.6**: Severe disparate impact (automatic HIGH risk)

**Example:**
```
Group 0 hiring rate: 40%
Group 1 hiring rate: 50%
DIR = 0.40 / 0.50 = 0.80 (exactly at threshold)
```

---

## 4. Risk Classification Logic

| Risk Level | Criteria |
|-----------|----------|
| **LOW** | All metrics pass thresholds |
| **MEDIUM** | 1 metric exceeds threshold |
| **HIGH** | 2+ metrics exceed thresholds OR DIR < 0.6 |

---

## 5. Software Architecture Details

### 4.1 Object-Oriented Design

#### BaseAuditor (Abstract Base Class)

```python
class BaseAuditor(ABC):
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate data integrity"""
        pass
    
    @abstractmethod
    def audit(self) -> AuditResult:
        """Execute fairness audit"""
        pass
```

#### TabularAuditor (Concrete Implementation)

- Validates DataFrame schema
- Handles categorical encoding
- Wraps data with aif360 StandardDataset
- Trains proxy Logistic Regression classifier
- Computes all fairness metrics

#### MetricsCalculator (Static Utility)

All methods are static, enabling independent metric calculation without state.

### 4.2 Exception Handling

Comprehensive validation catches:
- Missing label columns
- Missing protected attributes
- Null values in critical columns
- Non-binary labels

### 4.3 Type Hints & PEP 8 Compliance

- Full type hints on all function signatures
- Docstrings follow NumPy/Google style
- 4-space indentation
- 79-character line limit compliance

---

## 5. Usage Guide

### 5.1 Basic Usage

```python
from fairness_engine import FairnessAuditor
import pandas as pd

# Load data
df = pd.read_csv('hiring_data.csv')

# Create auditor
auditor = FairnessAuditor(
    df=df,
    label_name='hired',
    protected_attributes=['gender']
)

# Generate report (JSON)
report_json = auditor.generate_report(as_json=True)
print(report_json)

# Generate report (Dictionary)
report_dict = auditor.generate_report(as_json=False)
print(f"Risk Level: {report_dict['risk_level']}")
```

### 5.2 Quick Audit

```python
# Get summary with just metrics
summary = auditor.quick_audit()
# Output: {'risk_level': 'Low', 'demographic_parity_difference': 0.025, ...}
```

### 5.3 Custom Thresholds

```python
from fairness_engine import MetricThresholds

custom_thresholds = MetricThresholds(
    demographic_parity_threshold=0.05,
    equalized_odds_threshold=0.05,
    disparate_impact_threshold=0.85
)

auditor = FairnessAuditor(
    df=df,
    label_name='hired',
    protected_attributes=['gender'],
    thresholds=custom_thresholds
)
```

### 5.4 Multiple Protected Attributes

```python
auditor = FairnessAuditor(
    df=df,
    label_name='hired',
    protected_attributes=['gender', 'age_group', 'race']
)
```

---

## 6. Report Structure

### Sample JSON Output

```json
{
  "demographic_parity_difference": 0.0256,
  "equalized_odds_difference": 0.0278,
  "disparate_impact_ratio": 1.0373,
  "risk_level": "Low",
  "metric_details": {
    "demographic_parity": {
      "score": 0.0256,
      "threshold": 0.1,
      "interpretation": "Lower is better (closer to 0)"
    },
    "equalized_odds": {
      "score": 0.0278,
      "threshold": 0.1,
      "interpretation": "Lower is better (closer to 0)"
    },
    "disparate_impact": {
      "score": 1.0373,
      "threshold": 0.8,
      "interpretation": "Higher is better (≥ 0.8 is acceptable)",
      "rule": "80% Rule"
    },
    "sample_size": 500,
    "protected_attributes": ["gender"]
  }
}
```

---

## 7. Testing & Validation

### 7.1 Test Coverage

The framework includes **8 comprehensive test cases**:

1. **Unbiased Dataset**: Validates LOW risk classification
2. **Moderately Biased Dataset**: Validates MEDIUM/HIGH risk detection
3. **Severely Biased Dataset**: Validates severe bias detection
4. **Missing Label Error**: Exception handling
5. **Missing Protected Attribute Error**: Exception handling
6. **Null Values Error**: Exception handling
7. **JSON Report Format**: Report structure validation
8. **Quick Audit Method**: Summary report validation

### 7.2 Running Tests

```bash
python test_fairness_engine.py
```

**Expected Output:**
```
######################################################################
# TEST SUMMARY
######################################################################
Total Tests: 8
Passed: 8
Failed: 0
Success Rate: 100.0%
######################################################################
```

### 7.3 Test Datasets

#### Unbiased Dataset
- Groups have ~70% hiring rate
- Selection rates differ by <3%
- DIR ≈ 1.04
- **Expected Risk**: LOW

#### Moderately Biased Dataset
- Group 0: ~44% hired
- Group 1: ~66% hired
- DIR ≈ 0.66
- **Expected Risk**: HIGH (DIR violation)

#### Severely Biased Dataset
- Group 0: ~20% hired
- Group 1: ~75% hired
- DIR ≈ 0.28
- **Expected Risk**: HIGH (severe DIR violation)

---

## 9. Dependencies

```
aif360>=0.5.0        # AI Fairness 360 toolkit
numpy>=1.20.0        # Numerical computing
pandas>=1.3.0        # Data manipulation
scikit-learn>=0.24.0 # Machine learning utilities
streamlit>=1.28.0    # Interactive web framework (NEW)
plotly>=5.17.0       # Interactive visualizations (NEW)
Pillow>=9.0.0        # Image processing (NEW)
```

### Installation

```bash
pip install -r requirements.txt
```

---

## 10. New Features in v2.0.0

### 10.1 Interactive Dashboard (Streamlit)

The new web dashboard (`app.py`) provides:

**Features:**
- 📁 Drag-and-drop CSV file upload with validation
- 🎯 Interactive column selection for analysis
- ⚙️ Customizable fairness thresholds
- 📊 Real-time metric visualization
- 📈 Color-coded risk indicators (🟢 LOW, 🟡 MEDIUM, 🔴 HIGH)
- 📉 Temporal trend analysis with interactive charts
- 📋 Automated recommendations based on bias detection
- 💾 Export capabilities (JSON reports, CSV data)

**Usage:**
```bash
streamlit run app.py
```

### 10.2 Temporal Bias Analysis

Track fairness metrics evolution across time periods:

```python
# Analyze bias trends by year
temporal_results = auditor.temporal_bias_analysis(df, time_column='year')

# Returns DataFrame with:
# - time_period: Year/month identifier
# - disparate_impact_ratio: DIR per period
# - demographic_parity_difference: DPD per period
# - equalized_odds_difference: EOD per period
# - sample_size: Records per period
# - risk_level: Risk classification per period
```

**Use Cases:**
- Monitor fairness improvements over time
- Identify when bias issues emerged
- Track effectiveness of fairness interventions
- Generate audit trails for compliance

### 10.3 Advanced Visualizations

7 reusable visualization functions via `visualization_utils.py`:

| Function | Output |
|----------|--------|
| `create_outcome_rates_chart()` | Bar chart comparing group approval/outcome rates |
| `create_disparate_impact_gauge()` | Gauge chart with 80% Rule pass/fail zones |
| `create_equity_heatmap()` | Heatmap of hiring/rejection distributions |
| `create_bias_trend_chart()` | Multi-axis line chart for temporal trends |
| `create_risk_summary_table()` | Formatted metric summary table |
| `format_metric_for_display()` | Human-readable metric formatting |
| `get_risk_color()` | Hex color by risk level |

All return Plotly figures compatible with Streamlit.

### 10.4 Automated Testing Pipeline

GitHub Actions CI/CD configuration (`.github/workflows/python-tests.yml`):

**Triggers:** On push to `main`/`develop` or Pull Requests  
**Matrix Testing:** Python 3.9, 3.10, 3.11  
**Pipeline:**
1. Environment setup with dependency caching
2. Code linting (flake8)
3. Full test suite execution
4. Code coverage reporting
5. Syntax validation for all modules
6. Code style check (black)

**Status:** ✅ All 8 original tests pass

---

## 11. Design Decisions & Rationale

### 11.1 Why Logistic Regression as Proxy Classifier?

- **Simple & Interpretable**: No hyperparameter tuning needed
- **Fast**: Trains quickly on large datasets
- **Stable**: Low variance across different random seeds
- **Assumption**: We're auditing data bias, not model performance

### 11.2 Why aif360 StandardDataset?

- **Industry Standard**: Widely used in fairness research
- **Mature**: Well-documented and battle-tested
- **Extensible**: Can add new metrics easily
- **Compatibility**: Ecosystem of fairness tools

### 11.3 Why JSON Reports?

- **Interoperability**: Works across all platforms/languages
- **Parseable**: Programmatic report generation
- **Human-Readable**: Easy to understand and audit
- **Integration**: Supports downstream systems/dashboards

### 11.4 Why Add Streamlit Dashboard?

- **Low-code**: Quick UI without frontend expertise
- **Interactive**: Real-time feedback on analysis changes
- **Production-ready**: Built-in deployment support
- **Accessible**: Non-technical users can run analyses
- **Shareable**: Easy to deploy and share reports

### 11.5 Why Temporal Analysis?

- **Compliance**: Document fairness over audit periods
- **Flexibility**: Detect emerging or improving bias patterns
- **Actionable**: Measure intervention effectiveness
- **Comprehensive**: Combines trend analysis with risk assessment

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

1. **Binary Labels Only**: Multi-class classification not supported
2. **Single First Protected Attribute**: Multiple attributes not fully tested
3. **Proxy Classifier Assumptions**: May not reflect actual model behavior
4. **Fixed Favorable Classes**: Assumes 1 is favorable for all columns

### 12.2 Future Enhancements

- [ ] Multi-class support
- [ ] Calibration metrics
- [ ] Individual fairness measures
- [ ] Intersectional fairness analysis
- [ ] Counterfactual fairness
- [ ] Custom classifier support
- [ ] HTML report generation
- [ ] Advanced ML model integration

---

## 13. Responsible AI Considerations

### 13.1 When to Use FairLens

✅ **Appropriate Use Cases:**
- Screening datasets before training
- Identifying bias sources in data pipelines
- Regulatory compliance checks (EEOC, GDPR)
- Building fairness-aware models

❌ **Inappropriate Use Cases:**
- Replacing human judgment
- Making final hiring/lending decisions alone
- Assuming perfect fairness metrics mean no bias

### 13.2 Interpretation Guidelines

- **Metrics are correlations, not causation**
- **Multiple metrics should be considered together**
- **Context matters**: Industry, legal, ethical standards vary
- **Transparent communication**: Share findings with stakeholders

---

## 14. File Structure & Modules

```
FairLens/
├── app.py                           # Streamlit web dashboard (NEW)
├── fairness_engine.py               # Core framework
├── visualization_utils.py           # Plotly visualization functions (NEW)
├── test_fairness_engine.py          # Comprehensive test suite
├── examples.py                      # Usage examples
├── QUICKSTART_EXAMPLES.py           # 5 runnable quick-start examples (NEW)
├── IMPLEMENTATION_GUIDE.md          # Step-by-step implementation guide (NEW)
├── README.md                        # This documentation
├── requirements.txt                 # Python dependencies
├── verify.py                        # Verification/demo script
├── 00_START_HERE.md                 # Project entry point
└── .github/
    └── workflows/
        └── python-tests.yml         # GitHub Actions CI/CD (NEW)
```

### Key Modules Description

| Module | Purpose | Lines |
|--------|---------|-------|
| `fairness_engine.py` | Core fairness audit framework | 700+ |
| `app.py` | Interactive Streamlit dashboard | 400+ |
| `visualization_utils.py` | 7 reusable Plotly chart functions | 300+ |
| `test_fairness_engine.py` | 8 comprehensive test cases | 250+ |

---

## 15. Example Workflows

### 15.1 Pre-Training Audit

```python
# Load raw data
raw_df = pd.read_csv('raw_data.csv')

# Audit for bias
auditor = FairnessAuditor(
    df=raw_df,
    label_name='target',
    protected_attributes=['gender', 'race']
)

# Check risk
risk = auditor.quick_audit()
if risk['risk_level'] == 'High':
    print("⚠️ HIGH BIAS DETECTED - Review data collection")
    # Investigate root causes
    # Apply fairness interventions (reweighting, stratification, etc.)
```

### 15.2 Compliance Documentation

```python
# Generate formal audit trail
full_report = auditor.generate_report(as_json=True)

# Save for compliance
with open('audit_report.json', 'w') as f:
    f.write(full_report)

# Timestamp for documentation
import datetime
metadata = {
    'timestamp': datetime.datetime.now().isoformat(),
    'dataset': 'hiring_data_v2',
    'auditor': 'FairLens v1.0',
    'audit_result': json.loads(full_report)
}
```

---

---

## 16. FAQ

**Q: What's the difference between DPD and DIR?**
A: DPD measures absolute difference in rates; DIR measures ratio. DIR is more commonly used in legal contexts.

**Q: Can I use this on regression tasks?**
A: Not directly. Convert continuous targets to binary first (e.g., income > median).

**Q: What if my data has more than 2 groups?**
A: Current version compares group 0 vs. others. Extend TabularAuditor for pairwise comparisons.

**Q: How do I fix detected bias?**
A: FairLens identifies bias; fixing requires fairness interventions (reweighting, stratification, synthetic data, etc.).

**Q: How do I use the Streamlit dashboard?**
A: Run `streamlit run app.py` and upload your CSV file. The dashboard will guide you through the analysis.

**Q: Can I export my analysis results?**
A: Yes! Export as JSON reports or CSV data with temporal trends from the dashboard.

**Q: Is the temporal analysis required?**
A: No, it's optional. Use it when you have time-series data (yearly, monthly snapshots).

---

## 17. References

### Academic & Industry Standards
- **AI Fairness 360**: https://github.com/Trusted-AI/AIF360
- **Fairness Definitions Explained**: https://arxiv.org/abs/1710.03184
- **EEOC Guidance on Disparate Impact**: https://www.eeoc.gov/laws/guidance/
- **Microsoft's Responsible AI Guidelines**: https://www.microsoft.com/en-us/ai/responsible-ai
- **Google's Fairness Definitions**: https://developers.google.com/machine-learning/fairness-friendly

### Technical Documentation
- **Streamlit Documentation**: https://docs.streamlit.io
- **Plotly Documentation**: https://plotly.com/python/
- **aif360 Documentation**: https://aif360.mybluemix.net/
- **Pandas Documentation**: https://pandas.pydata.org/

---

## 18. Getting Help

### Documentation
1. **Start Here**: Read [00_START_HERE.md](00_START_HERE.md)
2. **Quick Start**: Check [QUICKSTART_EXAMPLES.py](QUICKSTART_EXAMPLES.py)
3. **Implementation**: Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
4. **Examples**: Run [examples.py](examples.py)

### Support
- Report issues on the project repository
- Check existing issues for similar problems
- Review the FAQ section above

---

## 19. Version History

### v2.0.0 (April 2, 2026)
**New Features:**
- ✨ Interactive Streamlit web dashboard
- ⏱️ Temporal bias analysis with trend tracking
- 📊 Advanced Plotly visualizations
- 🔄 Automated GitHub Actions CI/CD pipeline
- 📚 Comprehensive implementation guide
- 📋 Extended documentation and examples

**Improvements:**
- 💾 Enhanced data export options
- 🎨 Better visualization and UX
- 📈 Time-series analysis support
- ✅ Expanded test coverage

### v1.0.0 (January 2, 2026)
**Initial Release:**
- Core fairness audit framework
- 3 industry-standard fairness metrics (DPD, EOD, DIR)
- Risk classification system
- Comprehensive test suite (8 tests)
- Full documentation

---

## 20. Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- All tests pass (`python test_fairness_engine.py`)
- New features include tests
- Documentation is updated

---

## License

MIT License - See LICENSE file for details

## Contact & Attribution

**Project Lead:** FairLens Development Team  
**Version:** 2.0.0  
**Status:** ✅ Production Ready  
**Last Updated:** April 2, 2026

For questions, feedback, or collaboration opportunities, please open an issue on the project repository.

---

**Disclaimer:** This framework is designed to detect potential algorithmic bias in datasets. While it incorporates industry-standard fairness metrics, addressing bias requires domain expertise, business context understanding, and comprehensive fairness interventions beyond automated detection.
