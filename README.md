# FairLens
# FairLens: Proactive Algorithmic Bias Auditing Framework

## Executive Summary

**FairLens** is a production-ready Python framework for detecting algorithmic bias in tabular datasets before model training. By implementing industry-standard fairness metrics and risk assessment, FairLens enables responsible AI teams to catch bias at the data-sourcing stage rather than post-hoc analysis.

---

## 1. Architecture Overview

### 1.1 System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    FairnessAuditor                           │
│              (Main Public Interface)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌──────────┐
   │BaseAuditor      │TabularAuditor   │MetricsCalculator│
   │(Abstract)       │(Concrete)       │(Utility)         │
   └─────────┘      └─────────┘      └──────────┘
        △                │                │
        │                │                │
        └────────────────┼────────────────┘
                    Inheritance &
                  Composition Pattern
```

### 1.2 Component Responsibilities

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **FairnessAuditor** | Public API | Orchestrates auditing; returns JSON reports |
| **BaseAuditor** | Abstract contract | Defines interface for all auditors |
| **TabularAuditor** | Concrete implementation | Handles pandas DataFrames; wraps aif360 |
| **MetricsCalculator** | Utility class | Static methods for metric computation |
| **RiskAssessment** | Risk classification | Maps metrics to risk levels |

---

## 2. Fairness Metrics Implementation

### 2.1 Demographic Parity Difference (DPD)

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

### 2.2 Equalized Odds Difference (EOD)

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

### 2.3 Disparate Impact Ratio (DIR) - The 80% Rule

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

## 3. Risk Classification Logic

| Risk Level | Criteria |
|-----------|----------|
| **LOW** | All metrics pass thresholds |
| **MEDIUM** | 1 metric exceeds threshold |
| **HIGH** | 2+ metrics exceed thresholds OR DIR < 0.6 |

---

## 4. Software Architecture Details

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

## 8. Dependencies

```
aif360>=0.5.0        # AI Fairness 360 toolkit
numpy>=1.20.0        # Numerical computing
pandas>=1.3.0        # Data manipulation
scikit-learn>=0.24.0 # Machine learning utilities
```

### Installation

```bash
pip install aif360 numpy pandas scikit-learn
```

---

## 9. Design Decisions & Rationale

### 9.1 Why Logistic Regression as Proxy Classifier?

- **Simple & Interpretable**: No hyperparameter tuning needed
- **Fast**: Trains quickly on large datasets
- **Stable**: Low variance across different random seeds
- **Assumption**: We're auditing data bias, not model performance

### 9.2 Why aif360 StandardDataset?

- **Industry Standard**: Widely used in fairness research
- **Mature**: Well-documented and battle-tested
- **Extensible**: Can add new metrics easily
- **Compatibility**: Ecosystem of fairness tools

### 9.3 Why JSON Reports?

- **Interoperability**: Works across all platforms/languages
- **Parseable**: Programmatic report generation
- **Human-Readable**: Easy to understand and audit
- **Integration**: Supports downstream systems/dashboards

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Binary Labels Only**: Multi-class classification not supported
2. **Single First Protected Attribute**: Multiple attributes not fully tested
3. **Proxy Classifier Assumptions**: May not reflect actual model behavior
4. **Fixed Favorable Classes**: Assumes 1 is favorable for all columns

### 10.2 Future Enhancements

- [ ] Multi-class support
- [ ] Calibration metrics
- [ ] Individual fairness measures
- [ ] Intersectional fairness analysis
- [ ] Counterfactual fairness
- [ ] Custom classifier support
- [ ] HTML report generation
- [ ] Dashboard integration

---

## 11. Responsible AI Considerations

### 11.1 When to Use FairLens

✅ **Appropriate Use Cases:**
- Screening datasets before training
- Identifying bias sources in data pipelines
- Regulatory compliance checks (EEOC, GDPR)
- Building fairness-aware models

❌ **Inappropriate Use Cases:**
- Replacing human judgment
- Making final hiring/lending decisions alone
- Assuming perfect fairness metrics mean no bias

### 11.2 Interpretation Guidelines

- **Metrics are correlations, not causation**
- **Multiple metrics should be considered together**
- **Context matters**: Industry, legal, ethical standards vary
- **Transparent communication**: Share findings with stakeholders

---

## 12. File Structure

```
FairLens/
├── fairness_engine.py          # Core framework (500+ lines)
├── test_fairness_engine.py     # Comprehensive test suite
├── README.md                    # This documentation
└── requirements.txt             # Python dependencies
```

---

## 13. Example Workflows

### 13.1 Pre-Training Audit

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

### 13.2 Compliance Documentation

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

## 14. FAQ

**Q: What's the difference between DPD and DIR?**
A: DPD measures absolute difference in rates; DIR measures ratio. DIR is more commonly used in legal contexts.

**Q: Can I use this on regression tasks?**
A: Not directly. Convert continuous targets to binary first (e.g., income > median).

**Q: What if my data has more than 2 groups?**
A: Current version compares group 0 vs. others. Extend TabularAuditor for pairwise comparisons.

**Q: How do I fix detected bias?**
A: FairLens identifies bias; fixing requires fairness interventions (reweighting, stratification, synthetic data, etc.).

---

## 15. References

- **AI Fairness 360**: https://github.com/Trusted-AI/AIF360
- **EEOC Guidance on Disparate Impact**: https://www.eeoc.gov/laws/guidance/
- **Fairness Definitions Explained**: https://arxiv.org/abs/1710.03184
- **Responsible AI Guidelines**: https://www.microsoft.com/en-us/ai/responsible-ai

---

## License

MIT License - See LICENSE file for details

## Contact

For questions or contributions, please open an issue on the project repository.

---

**Last Updated**: January 2, 2026
**Version**: 1.0.0
**Status**: Production Ready
