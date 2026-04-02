"""
FairLens Test Paketi: Sentetik Veri Doğrulaması

Bu test scripti, farklı bias seviyeleriyle sentetik veri setleri kullanarak
FairLens fairness engine'ini doğrular. Testler şunları içerir:

1. Önyargısız Veri Seti: Tüm metrikler DÜŞÜK riski göstermeli
2. Orta Derecede Önyargılı Veri Seti: Karışık metrik ihlalleri (ORTA risk)
3. Ağır Önyargılı Veri Seti: Çok sayıda ihlal (YÜKSEK risk)
4. Uç Durumlar: Eksik değerleri, geçersiz girdileri işleme

Yazarlar: FairLens Geliştirme Ekibi
Sürüm: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple

# Import fairness engine
from fairness_engine import (
    FairnessAuditor,
    MetricThresholds,
    RiskLevel,
    BaseAuditor,
    TabularAuditor
)


def create_unbiased_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Minimum önyargı ile sentetik veri seti oluştur.
    
    Veri seti özellikleri:
    - İkili cinsiyet (0: Kadın, 1: Erkek)
    - İkili işe alım kararı (0: Reddedildi, 1: Alındı)
    - Her iki grup da benzer seçilme oranlarına sahip (~70%)
    - Özellikler rastgele dağıtılmış
    
    Parametreler:
        n_samples: Oluşturulacak örnek sayısı
        seed: Tekrarlanabilirlik için rastgele seed
    
    Döndürüm:
        pd.DataFrame: Önyargısız sentetik veri seti
    """
    np.random.seed(seed)
    
    df = pd.DataFrame({
        'gender': np.random.randint(0, 2, n_samples),
        'age': np.random.normal(40, 10, n_samples).astype(int),
        'experience_years': np.random.exponential(5, n_samples).astype(int),
        'education_level': np.random.randint(1, 5, n_samples),
        'test_score': np.random.normal(75, 15, n_samples)
    })
    
    # Korumalı grup için minimum önyargıyla işe alım kararı oluştur
    # Her iki grup da ~70% oranında işe alındı
    def hiring_decision(row):
        score = (row['age'] / 100 + row['experience_years'] / 20 + 
                row['education_level'] / 10 + row['test_score'] / 100)
        return 1 if score > np.percentile(df['age'] / 100 + df['experience_years'] / 20 + 
                                         df['education_level'] / 10 + df['test_score'] / 100, 30) else 0
    
    df['hired'] = df.apply(hiring_decision, axis=1)
    
    return df


def create_moderately_biased_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Korumalı gruba karşı orta derecede önyargı içeren veri seti oluştur.
    
    Veri seti özellikleri:
    - Grup 0 (korumalı): ~45% seçilme oranı
    - Grup 1 (ayrıcalıklı): ~70% seçilme oranı
    - Disparate Impact Oranı: ~0.64 (80% kuralını ihlal eder)
    
    Parametreler:
        n_samples: Oluşturulacak örnek sayısı
        seed: Tekrarlanabilirlik için rastgele seed
    
    Döndürüm:
        pd.DataFrame: Orta derecede önyargılı sentetik veri seti
    """
    np.random.seed(seed)
    
    df = pd.DataFrame({
        'gender': np.random.randint(0, 2, n_samples),
        'age': np.random.normal(40, 10, n_samples).astype(int),
        'experience_years': np.random.exponential(5, n_samples).astype(int),
        'education_level': np.random.randint(1, 5, n_samples),
        'test_score': np.random.normal(75, 15, n_samples)
    })
    
    # Önyargılı işe alım kararı oluştur
    def hiring_decision_biased(row):
        # Temel puan
        score = (row['age'] / 100 + row['experience_years'] / 20 + 
                row['education_level'] / 10 + row['test_score'] / 100)
        
        # Önyargı ekle: Grup 0 daha yüksek eşik gerektirir
        threshold = np.percentile(df['age'] / 100 + df['experience_years'] / 20 + 
                                df['education_level'] / 10 + df['test_score'] / 100, 35)
        
        if row['gender'] == 0:  # Korumalı grup: 45% işe alındı
            threshold = np.percentile(df['age'] / 100 + df['experience_years'] / 20 + 
                                    df['education_level'] / 10 + df['test_score'] / 100, 55)
        
        return 1 if score > threshold else 0
    
    df['hired'] = df.apply(hiring_decision_biased, axis=1)
    
    return df


def create_severely_biased_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Korumalı gruba karşı ağır önyargı içeren veri seti oluştur.
    
    Veri seti özellikleri:
    - Grup 0 (korumalı): ~20% seçilme oranı
    - Grup 1 (ayrıcalıklı): ~75% seçilme oranı
    - Disparate Impact Oranı: ~0.27 (ağır ihlal)
    - Tüm metrikler genelinde yüksek risk
    
    Parametreler:
        n_samples: Oluşturulacak örnek sayısı
        seed: Tekrarlanabilirlik için rastgele seed
    
    Döndürüm:
        pd.DataFrame: Ağır önyargılı sentetik veri seti
    """
    np.random.seed(seed)
    
    df = pd.DataFrame({
        'gender': np.random.randint(0, 2, n_samples),
        'age': np.random.normal(40, 10, n_samples).astype(int),
        'experience_years': np.random.exponential(5, n_samples).astype(int),
        'education_level': np.random.randint(1, 5, n_samples),
        'test_score': np.random.normal(75, 15, n_samples)
    })
    
    # Ağır önyargılı işe alım kararı oluştur
    hired = []
    for idx, row in df.iterrows():
        if row['gender'] == 0:  # Korumalı grup: 20% işe alındı
            hired.append(1 if np.random.random() < 0.20 else 0)
        else:  # Ayrıcalıklı grup: 75% işe alındı
            hired.append(1 if np.random.random() < 0.75 else 0)
    
    df['hired'] = hired
    
    return df


def test_unbiased_dataset():
    """Önyargısız veri seti üzerinde framework'ü test et - DÜŞÜK riski bekle."""
    print("\n" + "="*70)
    print("TEST 1: ÖNYARGISIZ VERİ SETİ")
    print("="*70)
    
    df = create_unbiased_dataset()
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nClass distribution for gender:")
    print(f"Gender 0: {(df['gender'] == 0).sum()} ({(df['gender'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"Gender 1: {(df['gender'] == 1).sum()} ({(df['gender'] == 1).sum() / len(df) * 100:.1f}%)")
    
    print("\nHiring rates by gender:")
    for gender in [0, 1]:
        rate = df[df['gender'] == gender]['hired'].mean()
        print(f"Gender {gender}: {rate:.1%} hired")
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        
        print("\n--- Audit Report ---")
        report = auditor.generate_report(as_json=False)
        print(json.dumps(report, indent=2))
        
        # Validate expectations
        assert report['risk_level'] == RiskLevel.LOW.value, \
            f"Expected LOW risk, got {report['risk_level']}"
        print(f"\n✓ TEST PASSED: Risk level is {report['risk_level']} (as expected)")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        raise


def test_moderately_biased_dataset():
    """Orta derecede önyargılı veri seti üzerinde test et - ORTA/YÜKSEK riski bekle."""
    print("\n" + "="*70)
    print("TEST 2: ORTA DERECEDİ ÖNYARGILI VERİ SETİ")
    print("="*70)
    
    df = create_moderately_biased_dataset()
    print(f"\nDataset shape: {df.shape}")
    
    print("\nHiring rates by gender:")
    for gender in [0, 1]:
        rate = df[df['gender'] == gender]['hired'].mean()
        print(f"Gender {gender}: {rate:.1%} hired")
    
    # Calculate Disparate Impact Ratio manually
    rate_0 = (df[df['gender'] == 0]['hired'] == 1).sum() / (df['gender'] == 0).sum()
    rate_1 = (df[df['gender'] == 1]['hired'] == 1).sum() / (df['gender'] == 1).sum()
    dir_manual = rate_0 / rate_1
    print(f"\nDisparate Impact Ratio (manual): {dir_manual:.3f}")
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        
        print("\n--- Audit Report ---")
        report = auditor.generate_report(as_json=False)
        print(json.dumps(report, indent=2))
        
        # Validate expectations
        assert report['risk_level'] in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value], \
            f"Expected MEDIUM or HIGH risk, got {report['risk_level']}"
        print(f"\n✓ TEST PASSED: Risk level is {report['risk_level']} (bias detected)")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        raise


def test_severely_biased_dataset():
    """Ağır önyargılı veri seti üzerinde test et - YÜKSEK riski bekle."""
    print("\n" + "="*70)
    print("TEST 3: AĞIR ÖNYARGILI VERİ SETİ")
    print("="*70)
    
    df = create_severely_biased_dataset()
    print(f"\nDataset shape: {df.shape}")
    
    print("\nHiring rates by gender:")
    for gender in [0, 1]:
        rate = df[df['gender'] == gender]['hired'].mean()
        print(f"Gender {gender}: {rate:.1%} hired")
    
    # Calculate Disparate Impact Ratio manually
    rate_0 = (df[df['gender'] == 0]['hired'] == 1).sum() / (df['gender'] == 0).sum()
    rate_1 = (df[df['gender'] == 1]['hired'] == 1).sum() / (df['gender'] == 1).sum()
    dir_manual = rate_0 / rate_1
    print(f"\nDisparate Impact Ratio (manual): {dir_manual:.3f}")
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        
        print("\n--- Audit Report ---")
        report = auditor.generate_report(as_json=False)
        print(json.dumps(report, indent=2))
        
        # Validate expectations
        assert report['risk_level'] == RiskLevel.HIGH.value, \
            f"Expected HIGH risk, got {report['risk_level']}"
        print(f"\n✓ TEST PASSED: Risk level is {report['risk_level']} (severe bias detected)")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        raise


def test_missing_label_error():
    """Eksik etiket sütunu için hata işleme testini yap."""
    print("\n" + "="*70)
    print("TEST 4: HATA İŞLEME - EKSIK ETİKET")
    print("="*70)
    
    df = create_unbiased_dataset()
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='nonexistent_label',
            protected_attributes=['gender']
        )
        report = auditor.generate_report(as_json=False)
        print("✗ TEST FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ TEST PASSED: Correctly caught error")
        print(f"Error message: {str(e)[:100]}...")


def test_missing_protected_attribute_error():
    """Eksik korumalı öznitelik için hata işleme testini yap."""
    print("\n" + "="*70)
    print("TEST 5: HATA İŞLEME - EKSIK KORUMALI ÖZNİTELİK")
    print("="*70)
    
    df = create_unbiased_dataset()
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['nonexistent_attribute']
        )
        report = auditor.generate_report(as_json=False)
        print("✗ TEST FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ TEST PASSED: Correctly caught error")
        print(f"Error message: {str(e)[:100]}...")


def test_null_values_error():
    """Null değerler için hata işleme testini yap."""
    print("\n" + "="*70)
    print("TEST 6: HATA İŞLEME - NULL DEĞERLERİ")
    print("="*70)
    
    df = create_unbiased_dataset()
    df.loc[0:10, 'gender'] = np.nan
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        report = auditor.generate_report(as_json=False)
        print("✗ TEST FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ TEST PASSED: Correctly caught error")
        print(f"Error message: {str(e)[:100]}...")


def test_json_report_format():
    """JSON rapor formatının geçerli olduğunu test et."""
    print("\n" + "="*70)
    print("TEST 7: JSON RAPOR FORMATI")
    print("="*70)
    
    df = create_unbiased_dataset()
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        
        json_report = auditor.generate_report(as_json=True)
        print(f"\nJSON Report:\n{json_report}")
        
        # Parse to validate JSON
        parsed = json.loads(json_report)
        
        # Validate required keys
        required_keys = [
            'demographic_parity_difference',
            'equalized_odds_difference',
            'disparate_impact_ratio',
            'risk_level',
            'metric_details'
        ]
        
        for key in required_keys:
            assert key in parsed, f"Missing required key: {key}"
        
        print(f"\n✓ TEST PASSED: JSON format is valid with all required keys")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        raise


def test_quick_audit():
    """quick_audit metotunu test et."""
    print("\n" + "="*70)
    print("TEST 8: HIZLI DENETİM METODU")
    print("="*70)
    
    df = create_moderately_biased_dataset()
    
    try:
        auditor = FairnessAuditor(
            df=df,
            label_name='hired',
            protected_attributes=['gender']
        )
        
        quick_result = auditor.quick_audit()
        print(f"\nQuick Audit Result:\n{json.dumps(quick_result, indent=2)}")
        
        # Validate required keys
        required_keys = [
            'risk_level',
            'demographic_parity_difference',
            'equalized_odds_difference',
            'disparate_impact_ratio'
        ]
        
        for key in required_keys:
            assert key in quick_result, f"Missing required key: {key}"
        
        print(f"\n✓ TEST PASSED: Quick audit returns essential metrics")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        raise


def run_all_tests():
    """Tüm test durumlarını çalıştır."""
    print("\n")
    print("#"*70)
    print("# FAIRLENS ÇATISI TEST PAKETİ")
    print("#"*70)
    print("# Fairness denetim işlevselliğinin test edilmesi")
    print("# sentetik veri setleriyle")
    print("#"*70)
    
    tests = [
        ("Unbiased Dataset", test_unbiased_dataset),
        ("Moderately Biased Dataset", test_moderately_biased_dataset),
        ("Severely Biased Dataset", test_severely_biased_dataset),
        ("Missing Label Error", test_missing_label_error),
        ("Missing Protected Attribute Error", test_missing_protected_attribute_error),
        ("Null Values Error", test_null_values_error),
        ("JSON Report Format", test_json_report_format),
        ("Quick Audit Method", test_quick_audit),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ Test '{test_name}' failed: {str(e)}")
    
    # Özet
    print("\n" + "#"*70)
    print("# TEST ÖZETİ")
    print("#"*70)
    print(f"Toplam Testler: {len(tests)}")
    print(f"Geçen: {passed}")
    print(f"Başarısız: {failed}")
    print(f"Başarı Oranı: {passed/len(tests)*100:.1f}%")
    print("#"*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
