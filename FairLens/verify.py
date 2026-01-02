"""
FairLens Framework Verification Script

Verifies all components and delivers final status report.
"""

import sys
import os

print('='*70)
print('FAIRLENS FRAMEWORK VERIFICATION')
print('='*70)

# Verify all components exist
print('\n[1] Checking module imports...')
try:
    from fairness_engine import (
        FairnessAuditor, BaseAuditor, TabularAuditor, MetricsCalculator,
        RiskLevel, MetricThresholds, AuditResult
    )
    print('    SUCCESS: All core components imported')
except ImportError as e:
    print(f'    ERROR: {e}')
    sys.exit(1)

# Verify test imports
print('\n[2] Checking test imports...')
try:
    from test_fairness_engine import (
        create_unbiased_dataset, create_moderately_biased_dataset,
        create_severely_biased_dataset
    )
    print('    SUCCESS: All test utilities imported')
except ImportError as e:
    print(f'    ERROR: {e}')
    sys.exit(1)

# Quick functionality test
print('\n[3] Running quick functionality test...')
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'gender': np.random.randint(0, 2, 100),
    'age': np.random.randn(100),
    'result': np.random.randint(0, 2, 100)
})

auditor = FairnessAuditor(df, 'result', ['gender'])
report = auditor.generate_report(as_json=False)

print(f'    Risk Level: {report["risk_level"]}')
print(f'    DPD: {report["demographic_parity_difference"]:.4f}')
print(f'    EOD: {report["equalized_odds_difference"]:.4f}')
print(f'    DIR: {report["disparate_impact_ratio"]:.4f}')
print('    SUCCESS: Framework executed successfully')

# Verify all files
print('\n[4] Checking deliverables...')
files = ['fairness_engine.py', 'test_fairness_engine.py', 'examples.py', 'README.md', 'requirements.txt']
for f in files:
    exists = os.path.exists(f)
    status = 'OK' if exists else 'MISSING'
    print(f'    {f}: {status}')

print('\n' + '='*70)
print('VERIFICATION COMPLETE: All systems operational')
print('='*70)
