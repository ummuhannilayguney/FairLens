"""
FairLens Kullanım Örnekleri - Basit Demo

Bu script FairLens çerçevesinin temel kullanımını gösterir.
"""

import json
import pandas as pd
import numpy as np
from fairness_engine import FairnessAuditor, MetricThresholds


print("\n" + "="*70)
print("FAIRLENS USAGE EXAMPLES")
print("="*70)

# Example 1: Basic audit
print("\n[EXAMPLE 1] Basic Fairness Audit")
print("-"*70)

np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.normal(40, 10, 100).astype(int),
    'experience': np.random.exponential(5, 100).astype(int),
    'gender': np.random.randint(0, 2, 100),
    'hired': np.random.randint(0, 2, 100)
})

print(f"Dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())

auditor = FairnessAuditor(
    df=df,
    label_name='hired',
    protected_attributes=['gender']
)

report = auditor.generate_report(as_json=False)
print(f"\nRisk Level: {report['risk_level']}")
print(f"DPD: {report['demographic_parity_difference']:.4f}")
print(f"EOD: {report['equalized_odds_difference']:.4f}")
print(f"DIR: {report['disparate_impact_ratio']:.4f}")


# Example 2: Quick summary
print("\n" + "="*70)
print("[EXAMPLE 2] Quick Audit Summary")
print("-"*70)

np.random.seed(42)
df2 = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'race': np.random.randint(0, 2, 200),
    'approved': np.random.randint(0, 2, 200)
})

auditor2 = FairnessAuditor(
    df=df2,
    label_name='approved',
    protected_attributes=['race']
)

quick = auditor2.quick_audit()
print("\nQuick Summary:")
print(json.dumps(quick, indent=2))


# Example 3: Custom thresholds
print("\n" + "="*70)
print("[EXAMPLE 3] Custom Fairness Thresholds")
print("-"*70)

np.random.seed(42)
df3 = pd.DataFrame({
    'score': np.random.normal(75, 15, 150),
    'age_group': np.random.randint(0, 2, 150),
    'decision': np.random.randint(0, 2, 150)
})

strict_thresholds = MetricThresholds(
    demographic_parity_threshold=0.05,
    equalized_odds_threshold=0.05,
    disparate_impact_threshold=0.9
)

auditor3 = FairnessAuditor(
    df=df3,
    label_name='decision',
    protected_attributes=['age_group'],
    thresholds=strict_thresholds
)

report3 = auditor3.generate_report(as_json=False)
print(f"Risk Level (with stricter thresholds): {report3['risk_level']}")


# Example 4: JSON export
print("\n" + "="*70)
print("[EXAMPLE 4] JSON Report Export")
print("-"*70)

json_report = auditor.generate_report(as_json=True)
print("\nFull JSON Report:")
print(json_report[:300] + "\n... (truncated)")


# Example 5: Error handling
print("\n" + "="*70)
print("[EXAMPLE 5] Error Handling")
print("-"*70)

df_error = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 1]})

print("\nTest: Missing protected attribute")
try:
    auditor_error = FairnessAuditor(
        df=df_error,
        label_name='y',
        protected_attributes=['missing_column']
    )
    auditor_error.generate_report()
except ValueError as e:
    print(f"Successfully caught error: {str(e)[:70]}...")


print("\n" + "="*70)
print("All examples completed successfully!")
print("="*70 + "\n")
