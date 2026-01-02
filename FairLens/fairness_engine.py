"""
FairLens: Proaktif Algoritmik Önyargı Denetim Çerçevesi

Bu modül, model eğitiminden önce tablo veri setlerindeki algoritmik önyargıyı
tespit etmek için üretim ortamına hazır bir çerçeve sağlar. Sorumlu AI 
ilkelerine uygun endüstri standardı fairness metriklerini ve risk değerlendirmesini
uygular.

Yazarlar: FairLens Geliştirme Ekibi
Sürüm: 1.0.0
Lisans: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from aif360.datasets import StandardDataset
except ImportError:
    raise ImportError(
        "aif360 not installed. Install via: pip install aif360"
    )


class RiskLevel(Enum):
    """Önyargı riski sınıflandırması seviyeleri için numaralandırma."""
    LOW = "Düşük"
    MEDIUM = "Orta"
    HIGH = "Yüksek"


@dataclass
class MetricThresholds:
    """Fairness metrikler için endüstri standardı eşik değerleri."""
    # Demografik Parite: 0.1'den küçük önerilir
    demographic_parity_threshold: float = 0.1
    # Eşitlenmiş Şanslar: 0.1'den küçük önerilir
    equalized_odds_threshold: float = 0.1
    # Disparate Impact Oranı: 80% kuralı => >= 0.8
    disparate_impact_threshold: float = 0.8


@dataclass
class AuditResult:
    """Fairness denetiminden alınan yapılandırılmış sonuç."""
    demographic_parity_difference: float
    equalized_odds_difference: float
    disparate_impact_ratio: float
    risk_level: str
    metric_details: Dict[str, Any]

    def to_json(self) -> str:
        """Denetim sonucunu JSON string'e dönüştür."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Denetim sonucunu sözlüğe dönüştür."""
        return asdict(self)


class BaseAuditor(ABC):
    """
    Fairness denetçileri için arayüz tanımlayan soyut temel sınıf.
    
    Tüm somut denetçi uygulamaları bu sınıftan türemelidir
    ve gerekli soyut metotları uygulamalıdır.
    """

    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Veri bütünlüğü ve gerekli öznitelikleri doğrula.
        
        Döndürüm:
            Tuple[bool, List[str]]: (geçerli_mi, hata_mesajları_listesi)
        """
        pass

    @abstractmethod
    def audit(self) -> AuditResult:
        """
        Veri seti üzerinde fairness denetimi uygula.
        
        Döndürüm:
            AuditResult: Metrik puanları ve risk seviyesini içeren yapılandırılmış sonuç
        """
        pass


class MetricsCalculator:
    """
    Fairness metriklerini hesaplamak için yardımcı sınıf.
    
    Tüm metotlar statiktir, böylece durumu korumadan
    bağımsız metrik hesaplamasına izin verilir.
    """

    @staticmethod
    def demographic_parity_difference(
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> float:
        """
        Demografik Parite Farkını (DPD) hesapla.
        
        Gruplar arasında pozitif tahmin oranlarındaki farkı ölçer.
        Daha düşük bir değer daha eşit muamele gösterir.
        
        Formül: |P(Y_pred=1|A=0) - P(Y_pred=1|A=1)|
        
        Parametreler:
            y_pred: Tahmin edilen etiketler (array-like)
            protected_attr: Korumalı öznitelik değerleri (array-like)
        
        Döndürüm:
            float: DPD puanı (0 ile 1 arasında)
        """
        y_pred = np.asarray(y_pred)
        protected_attr = np.asarray(protected_attr)
        
        # Get unique values in protected attribute
        groups = np.unique(protected_attr)
        if len(groups) != 2:
            warnings.warn(
                f"Expected binary protected attribute, got {len(groups)} groups"
            )
        
        selection_rates = []
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                selection_rates.append(0.0)
            else:
                rate = (y_pred[mask] == 1).sum() / mask.sum()
                selection_rates.append(float(rate))
        
        dpd = abs(selection_rates[0] - selection_rates[1])
        return dpd

    @staticmethod
    def equalized_odds_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> float:
        """
        Eşitlenmiş Şanslar Farkını (EOD) hesapla.
        
        Gruplar arasında Doğru Pozitif Oranı (TPR) ve 
        Yanlış Pozitif Oranındaki (FPR) maksimum farkı ölçer. 
        Daha düşük değerler daha eşit şanslar gösterir.
        
        Formül: max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)
        
        Parametreler:
            y_true: Gerçek etiketler (array-like)
            y_pred: Tahmin edilen etiketler (array-like)
            protected_attr: Korumalı öznitelik değerleri (array-like)
        
        Döndürüm:
            float: EOD puanı (0 ile 1 arasında)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        protected_attr = np.asarray(protected_attr)
        
        groups = np.unique(protected_attr)
        if len(groups) != 2:
            warnings.warn(
                f"Expected binary protected attribute, got {len(groups)} groups"
            )
        
        tpr_list = []
        fpr_list = []
        
        for group in groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # True Positive Rate
            tp_mask = (y_true_group == 1) & (y_pred_group == 1)
            p_mask = y_true_group == 1
            tpr = tp_mask.sum() / max(p_mask.sum(), 1)
            tpr_list.append(float(tpr))
            
            # False Positive Rate
            fp_mask = (y_true_group == 0) & (y_pred_group == 1)
            n_mask = y_true_group == 0
            fpr = fp_mask.sum() / max(n_mask.sum(), 1)
            fpr_list.append(float(fpr))
        
        eod = max(abs(tpr_list[0] - tpr_list[1]), abs(fpr_list[0] - fpr_list[1]))
        return eod

    @staticmethod
    def disparate_impact_ratio(
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> float:
        """
        Disparate Impact Oranını (DIR) hesapla - 80% Kuralı.
        
        Korumalı grup ile ayrıcalıklı grup arasında pozitif seçilme oranının oranı.
        Endüstri standardı: oran >= 0.8 kabul edilebilir sayılır.
        
        Formül: Selection_Rate(Korumalı) / Selection_Rate(Ayrıcalıklı)
        
        Parametreler:
            y_pred: Tahmin edilen etiketler (array-like)
            protected_attr: Korumalı öznitelik değerleri (array-like)
        
        Döndürüm:
            float: DIR puanı (0'dan sonsuza, ideal olarak 1.0'e yakın)
        """
        y_pred = np.asarray(y_pred)
        protected_attr = np.asarray(protected_attr)
        
        groups = np.unique(protected_attr)
        if len(groups) != 2:
            warnings.warn(
                f"Expected binary protected attribute, got {len(groups)} groups"
            )
        
        selection_rates = []
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                selection_rates.append(0.0)
            else:
                rate = (y_pred[mask] == 1).sum() / mask.sum()
                selection_rates.append(float(rate))
        
        # Avoid division by zero
        if selection_rates[0] == 0 or selection_rates[1] == 0:
            return 0.0
        
        # Typically: group 0 is protected, group 1 is privileged
        dir_ratio = selection_rates[0] / selection_rates[1]
        return dir_ratio


class TabularAuditor(BaseAuditor):
    """
    Tablo veri setleri için BaseAuditor'ın somut uygulaması.
    
    Pandas DataFrames'i işler ve aif360 StandardDataset ile sarmalanır.
    Gerekirse Logistic Regression'ı proxy sınıflandırıcı olarak kullanarak
    fairness metriklerini hesaplar.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_name: str,
        protected_attributes: List[str],
        thresholds: Optional[MetricThresholds] = None,
        favorable_classes: Optional[Dict[str, Any]] = None
    ):
        """
        TabularAuditor'u başlat.
        
        Parametreler:
            df: Giriş pandas DataFrame'i
            label_name: Hedef/etiket değişkeninin sütun adı
            protected_attributes: Korumalı özniteliklerin sütun adları listesi
            thresholds: Özel metrik eşikleri (varsayılan: endüstri standartları)
            favorable_classes: Sütun adlarını olumlu sınıf değerleriyle eşleyen sözlük
                              (varsayılan: etiket=1 ve korumalı_nitelik=1 olumsaldır)
        
        Yükseltir:
            ValueError: label_name veya protected_attributes df'de eksikse
            TypeError: df pandas DataFrame değilse
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        self.df = df.copy()
        self.label_name = label_name
        self.protected_attributes = protected_attributes
        self.thresholds = thresholds or MetricThresholds()
        self.favorable_classes = favorable_classes or {}
        
        # Initialize as None; will be populated after validation
        self.standard_dataset: Optional[StandardDataset] = None
        self.y_pred: Optional[np.ndarray] = None

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Veri bütünlüğü ve gerekli öznitelikleri doğrula.
        
        Kontroller:
        - Etiket sütunu DataFrame'de var mı
        - Tüm korumalı öznitelikler DataFrame'de var mı
        - Kritik sütunlarda null değer yok mu
        - Etiket ikili mi yoksa ikili olarak kodlanabilir mi
        
        Döndürüm:
            Tuple[bool, List[str]]: (geçerli_mi, hata_mesajları_listesi)
        """
        errors = []
        
        # Check label exists
        if self.label_name not in self.df.columns:
            errors.append(
                f"Label '{self.label_name}' not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Check protected attributes exist
        missing_attrs = [attr for attr in self.protected_attributes 
                        if attr not in self.df.columns]
        if missing_attrs:
            errors.append(
                f"Protected attributes {missing_attrs} not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Check for null values in critical columns
        critical_cols = [self.label_name] + self.protected_attributes
        null_cols = [col for col in critical_cols 
                    if col in self.df.columns and self.df[col].isnull().any()]
        if null_cols:
            errors.append(
                f"Null values found in critical columns: {null_cols}. "
                "Please handle missing values before auditing."
            )
        
        # Check label is binary or can be encoded
        if self.label_name in self.df.columns:
            unique_labels = self.df[self.label_name].nunique()
            if unique_labels > 2:
                errors.append(
                    f"Label '{self.label_name}' has {unique_labels} unique values. "
                    "Only binary labels are supported."
                )
        
        return len(errors) == 0, errors

    def _prepare_dataset(self) -> None:
        """
        aif360 StandardDataset için verileri hazırla.
        
        Kodlamayı işler ve StandardDataset sarmalayıcısı oluşturur.
        """
        df_encoded = self.df.copy()
        
        # Encode categorical columns
        label_encoder = {}
        for col in [self.label_name] + self.protected_attributes:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoder[col] = le
        
        # Set favorable classes if not provided
        favorable_classes_dict = self.favorable_classes.copy()
        favorable_label = favorable_classes_dict.get(self.label_name, 1)
        
        # Create StandardDataset
        self.standard_dataset = StandardDataset(
            df=df_encoded,
            label_name=self.label_name,
            favorable_classes=[favorable_label],
            protected_attribute_names=self.protected_attributes,
            privileged_classes=[
                [favorable_classes_dict.get(attr, 1)] 
                for attr in self.protected_attributes
            ]
        )

    def _train_proxy_classifier(self) -> None:
        """
        Proxy tahminler için Logistic Regression sınıflandırıcısını eğit.
        
        Bu, gerçek tahminler mevcut değilse Eşitlenmiş Şanslar hesaplaması
        için kullanılır. Etiket ve korumalı öznitelikler hariç tüm özellikleri
        kullanır.
        """
        if self.standard_dataset is None:
            return
        
        # Extract features and labels
        X = self.standard_dataset.features
        y = self.standard_dataset.labels.ravel()
        
        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        clf.fit(X, y)
        
        # Generate predictions
        self.y_pred = clf.predict(X)

    def audit(self) -> AuditResult:
        """
        Tablo veri seti üzerinde fairness denetimi gerçekleştir.
        
        İşlem:
        1. Veri bütünlüğünü doğrula
        2. Veri setini aif360 için hazırla
        3. Gerekirse proxy sınıflandırıcısını eğit
        4. Fairness metriklerini hesapla
        5. Risk seviyesini değerlendir
        
        Döndürüm:
            AuditResult: Metrikler ve risk sınıflandırmasıyla yapılandırılmış sonuç
        
        Yükseltir:
            ValueError: Doğrulama başarısız olursa
        """
        # Validate data
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(
                f"Data validation failed:\n" + "\n".join(errors)
            )
        
        # Prepare dataset
        self._prepare_dataset()
        
        # Train proxy classifier
        self._train_proxy_classifier()
        
        # Get protected attribute and labels
        protected_attr_col = self.protected_attributes[0]
        protected_attr = self.df[protected_attr_col].values
        labels = self.df[self.label_name].values
        
        # Calculate metrics
        dpd = MetricsCalculator.demographic_parity_difference(
            y_pred=self.y_pred,
            protected_attr=protected_attr
        )
        
        eod = MetricsCalculator.equalized_odds_difference(
            y_true=labels,
            y_pred=self.y_pred,
            protected_attr=protected_attr
        )
        
        dir_ratio = MetricsCalculator.disparate_impact_ratio(
            y_pred=self.y_pred,
            protected_attr=protected_attr
        )
        
        # Assess risk level
        risk_level = self._assess_risk(dpd, eod, dir_ratio)
        
        # Prepare metric details
        metric_details = {
            "demographic_parity": {
                "score": dpd,
                "threshold": self.thresholds.demographic_parity_threshold,
                "interpretation": "Lower is better (closer to 0)"
            },
            "equalized_odds": {
                "score": eod,
                "threshold": self.thresholds.equalized_odds_threshold,
                "interpretation": "Lower is better (closer to 0)"
            },
            "disparate_impact": {
                "score": dir_ratio,
                "threshold": self.thresholds.disparate_impact_threshold,
                "interpretation": f"Higher is better (≥ {self.thresholds.disparate_impact_threshold} is acceptable)",
                "rule": "80% Rule"
            },
            "sample_size": len(self.df),
            "protected_attributes": self.protected_attributes
        }
        
        return AuditResult(
            demographic_parity_difference=dpd,
            equalized_odds_difference=eod,
            disparate_impact_ratio=dir_ratio,
            risk_level=risk_level.value,
            metric_details=metric_details
        )

    def _assess_risk(
        self,
        dpd: float,
        eod: float,
        dir_ratio: float
    ) -> RiskLevel:
        """
        Metrik puanlarına dayalı genel bias riskini değerlendir.
        
        Risk Sınıflandırma Mantığı:
        - DÜŞÜK: Tüm metrikler eşik kontrollerini geçer
        - ORTA: Bir veya iki metrik eşikleri aşar
        - YÜKSEK: Çok sayıda metrik eşikleri aşar veya DIR < 0.6
        
        Parametreler:
            dpd: Demografik Parite Farkı puanı
            eod: Eşitlenmiş Şanslar Farkı puanı
            dir_ratio: Disparate Impact Oranı puanı
        
        Döndürüm:
            RiskLevel: Sınıflandırılmış risk seviyesi
        """
        violations = 0
        
        # Check Demographic Parity
        if dpd > self.thresholds.demographic_parity_threshold:
            violations += 1
        
        # Check Equalized Odds
        if eod > self.thresholds.equalized_odds_threshold:
            violations += 1
        
        # Check Disparate Impact (80% Rule)
        if dir_ratio < self.thresholds.disparate_impact_threshold:
            violations += 1
        
        # Extreme case: severe disparate impact
        if dir_ratio < 0.6:
            return RiskLevel.HIGH
        
        # Classify based on number of violations
        if violations == 0:
            return RiskLevel.LOW
        elif violations == 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH


class FairnessAuditor:
    """
    FairLens fairness denetimi için halka açık API.
    
    Bu, kullanıcılar için ana giriş noktasıdır. Denetim
    işlemini yönetir ve kapsamlı raporlar oluşturmak için uygun
    metotlar sağlar.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_name: str,
        protected_attributes: List[str],
        thresholds: Optional[MetricThresholds] = None
    ):
        """
        FairnessAuditor'u başlat.
        
        Parametreler:
            df: Giriş pandas DataFrame'i
            label_name: Hedef/etiket değişkeninin sütun adı
            protected_attributes: Korumalı özniteliklerin sütun adları listesi
            thresholds: Özel metrik eşikleri (varsayılan: endüstri standartları)
        """
        self.auditor = TabularAuditor(
            df=df,
            label_name=label_name,
            protected_attributes=protected_attributes,
            thresholds=thresholds
        )

    def generate_report(self, as_json: bool = True) -> Union[Dict[str, Any], str]:
        """
        Kapsamlı fairness denetim raporu oluştur.
        
        Tam denetim işlem hattını çalıştırır ve sonuçları 
        sözlük veya JSON string olarak döndürür.
        
        Parametreler:
            as_json: True ise JSON string döndür; False ise sözlük döndür
        
        Döndürüm:
            Dict[str, Any] | str: İstenen biçimde denetim raporu
        
        Yükseltir:
            ValueError: Veri doğrulaması başarısız olursa
        """
        result = self.auditor.audit()
        
        if as_json:
            return result.to_json()
        else:
            return result.to_dict()

    def quick_audit(self) -> Dict[str, Any]:
        """
        Yalnızca gerekli metrikler ve risk seviyesiyle hızlı denetim.
        
        Tam detaylar olmadan hızlı risk değerlendirmesi için kullanışlıdır.
        
        Döndürüm:
            Dict[str, Any]: risk_level ve metrik puanlarıyla özet
        """
        result = self.auditor.audit()
        return {
            "risk_level": result.risk_level,
            "demographic_parity_difference": result.demographic_parity_difference,
            "equalized_odds_difference": result.equalized_odds_difference,
            "disparate_impact_ratio": result.disparate_impact_ratio
        }


# Örnek kullanım ve yardımcı fonksiyonlar
if __name__ == "__main__":
    # Bu bölüm yalnızca gösterim amaçlıdır
    print("FairLens Fairness Engine Modülü")
    print("Bu modülü içe aktarın ve FairnessAuditor sınıfını kullanın")
