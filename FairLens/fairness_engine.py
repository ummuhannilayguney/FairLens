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
        
        # Preprocessing warnings and metadata
        self.preprocessing_warnings: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.original_sample_size = len(self.df)
        
        # Apply data preprocessing
        self._preprocess_data()
        
        # Initialize as None; will be populated after validation
        self.standard_dataset: Optional[StandardDataset] = None
        self.y_pred: Optional[np.ndarray] = None

    def _preprocess_data(self) -> None:
        """
        Veri ön işleme: SIRAYLA tip uyuşmazlıklarını, eksik değerleri ve kodlama sorunlarını düzelt.
        
        Bu yöntem:
        1. StringDtype ve diğer problematik pandas türlerini standart NumPy türlerine dönüştürür  
        2. Hedef ve korumalı öznitelikleri sayısala çevir (önce encoding, sonra NaN removal)
        3. Yalnızca gerekli sütunları tutarak veri tabanını izole eder
        4. Hedef ve korumalı özniteliklerde NaN değerleri içeren satırları kaldırır
        5. Tüm uyarıları preprocessing_warnings listesine kaydeder
        
        Yükseltir:
            ValueError: Veri ön işleme sonrası hedef ikili değilse
        """
        # Step 1: Type casting - Problematic pandas dtypes → Standard Python types
        self._fix_dtype_issues()
        
        # Step 2: Column isolation - Keep ONLY target + protected columns (BEFORE encoding)
        self._isolate_required_columns()
        
        # Step 3: Encode target label (convert strings to 0/1)
        self._encode_target_label()
        
        # Step 4: Encode protected attributes (convert to numeric/binary)
        self._encode_protected_attributes()
        
        # Step 5: Remove rows with NaN AFTER encoding (now they're numeric or properly handled)
        self._handle_missing_values()
        
        # Log preprocessing stats
        removed_rows = self.original_sample_size - len(self.df)
        if removed_rows > 0:
            pct = (removed_rows / self.original_sample_size) * 100
            self.preprocessing_warnings.append(
                f"Removed {removed_rows} rows ({pct:.1f}%) with missing critical values"
            )

    def _fix_dtype_issues(self) -> None:
        """
        StringDtype ve diğer pandas extension dtypes'i standart NumPy türlerine çevir.
        
        Bu, AIF360 ve scikit-learn'ün StringDtype(na_value=nan) hataları vermesini önler.
        """
        for col in self.df.columns:
            dtype = self.df[col].dtype
            dtype_str = str(dtype)
            
            # Convert StringDtype to object - CRITICAL for AIF360
            # DO NOT use infer_objects() as it may reintroduce StringDtype
            if 'string' in dtype_str.lower():
                try:
                    # Reconstruct column to completely eliminate StringDtype
                    values = self.df[col].values
                    self.df[col] = pd.Series(
                        [str(x) if pd.notna(x) else x for x in values],
                        index=self.df.index,
                        dtype='object'
                    )
                    self.preprocessing_warnings.append(
                        f"Fixed StringDtype in column '{col}' for AIF360 compatibility"
                    )
                except Exception as exc:
                    try:
                        # Fallback: direct string conversion
                        self.df[col] = self.df[col].astype(str)
                    except:
                        pass
            
            # Convert boolean dtype to int
            elif dtype == bool or 'bool' in dtype_str.lower():
                try:
                    self.df[col] = self.df[col].astype(int)
                    self.preprocessing_warnings.append(
                        f"Converted boolean column '{col}' to integer"
                    )
                except Exception:
                    pass
            
            # Convert category dtype to object
            elif 'category' in dtype_str.lower():
                try:
                    self.df[col] = self.df[col].astype(object)
                    self.preprocessing_warnings.append(
                        f"Converted categorical column '{col}' to object"
                    )
                except Exception:
                    pass

    def _handle_missing_values(self) -> None:
        """
        Hedef ve korumalı öznitelik sütunlarından NaN değerli satırları kaldır.
        
        Diğer sütunlardaki NaN değerler Logistic Regression eğitimi sırasında
        otomatik olarak işlenir.
        """
        critical_cols = [self.label_name] + self.protected_attributes
        missing_before = len(self.df)
        
        for col in critical_cols:
            if col in self.df.columns:
                self.df = self.df[self.df[col].notna()]
        
        missing_after = len(self.df)
        if missing_before > missing_after:
            removed = missing_before - missing_after
            self.preprocessing_warnings.append(
                f"Removed {removed} rows with missing values in critical columns"
            )

    def _encode_target_label(self) -> None:
        """
        İkili olmayan hedef etiketleri otomatik olarak 0/1'e kodla.
        
        Eğer hedef string ise (örn: "Yes"/"No"), LabelEncoder kullanarak
        sırasıyla 0 ve 1 numaralarına eşle.
        """
        if self.label_name not in self.df.columns:
            return
        
        # Ensure the column is not StringDtype (convert if needed)
        dtype_str = str(self.df[self.label_name].dtype)
        if 'string' in dtype_str.lower():
            # Reconstruct column to eliminate StringDtype
            values = self.df[self.label_name].values
            self.df[self.label_name] = pd.Series(
                [str(x) if pd.notna(x) else x for x in values],
                index=self.df.index,
                dtype='object'
            )
        
        unique_values = self.df[self.label_name].nunique()
        
        # If already 0/1 or 1/2 binary, leave as is
        if unique_values == 2:
            unique_vals_sorted = sorted(self.df[self.label_name].unique())
            # Check if already binary numeric
            if set(unique_vals_sorted) in [{0, 1}, {1, 2}]:
                return
        
        # If not binary, check dtype (handle both object and str)
        current_dtype = str(self.df[self.label_name].dtype)
        is_string_type = self.df[self.label_name].dtype == 'object' or current_dtype in ['str', 'object', 'string']
        
        if is_string_type or unique_values > 2:
            if unique_values > 2:
                raise ValueError(
                    f"Target column '{self.label_name}' has {unique_values} unique values. "
                    f"Only binary targets (2 values) are supported. "
                    f"Found values: {list(self.df[self.label_name].unique())}"
                )
            
            # Use LabelEncoder to map string values to 0/1
            self.label_encoder = LabelEncoder()
            self.df[self.label_name] = self.label_encoder.fit_transform(self.df[self.label_name].astype(str))
            
            mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            self.preprocessing_warnings.append(
                f"Auto-encoded target '{self.label_name}' mapping: {mapping}"
            )

    def _encode_protected_attributes(self) -> None:
        """
        Korumalı öznitelikleri SIRAYLA sayısalla çevir (NaN tanıtmadan):
        1. String/Object columns: Ordinal encoding (sorted unique values -> 0,1,2...)
        2. Numeric columns (age, income): Eşik ile binary (>median -> 1, <=median -> 0)
        3. Son: TÜM korumalı öznitelikleri numeric olarak garanti et
        """
        for attr in self.protected_attributes:
            if attr not in self.df.columns:
                continue
            
            # Step 1: Fix dtype if StringDtype/str
            dtype_str = str(self.df[attr].dtype)
            if dtype_str in ['string', 'str'] or 'string' in dtype_str.lower():
                values = self.df[attr].values
                self.df[attr] = pd.Series(
                    [str(x) if pd.notna(x) else x for x in values],
                    index=self.df.index,
                    dtype='object'
                )
            
            # Step 2: Detect type and convert accordingly (SAFELY, no NaN)
            current_dtype = self.df[attr].dtype
            unique_count = self.df[attr].nunique()
            
            # Case A: Object/String column -> Ordinal encoding (safe, maps all values)
            if current_dtype == 'object':
                # Get all unique values (including NaN handling)
                unique_vals = sorted([v for v in self.df[attr].unique() if pd.notna(v)])
                mapping = {val: i for i, val in enumerate(unique_vals)}
                # Use fillna with mode to prevent NaN from encoding
                mode_val = self.df[attr].mode()[0] if len(self.df[attr].mode()) > 0 else unique_vals[0]
                self.df[attr] = self.df[attr].fillna(mode_val).map(mapping)
                self.preprocessing_warnings.append(
                    f"Encoded protected attribute '{attr}' (ordinal): {mapping}"
                )
            
            # Case B: Numeric column with >2 unique values (e.g., age) -> Binary threshold
            elif pd.api.types.is_numeric_dtype(self.df[attr]) and unique_count > 2:
                # Apply threshold: median value
                median_val = self.df[attr].median()
                self.df[attr] = (self.df[attr] > median_val).astype(int)
                self.preprocessing_warnings.append(
                    f"Converted numeric protected attribute '{attr}' to binary (threshold={median_val:.2f})"
                )
            
            # Case C: Already numeric/binary -> Ensure int64
            else:
                self.df[attr] = pd.to_numeric(self.df[attr], errors='coerce').astype('int64')
            
            # Final: Guarantee numeric with no NaN
            if not pd.api.types.is_numeric_dtype(self.df[attr]):
                self.df[attr] = pd.to_numeric(self.df[attr], errors='coerce')
            
            # Fill any remaining NaN with median or 0
            if self.df[attr].isna().any():
                fill_val = self.df[attr].median() if not self.df[attr].isna().all() else 0
                self.df[attr] = self.df[attr].fillna(fill_val)
                self.preprocessing_warnings.append(
                    f"Filled NaN in protected attribute '{attr}' with value: {fill_val}"
                )

    def _isolate_required_columns(self) -> None:
        """
        STRICT: Yalnızca hedef + korumalı öznitelikleri sakla. Tüm diğer sütunları kaldır.
        
        Gerekçe: Diğer sütunlardaki metin/kategorik veriler AIF360'ın sayısal veri beklentisini
        ihlal eder. AIF360 Logistic Regression için kendi features'ını oluşturur.
        """
        required_cols = [self.label_name] + self.protected_attributes
        cols_to_keep = [col for col in required_cols if col in self.df.columns]
        
        # Drop all columns except target and protected attributes
        cols_to_drop = [col for col in self.df.columns if col not in cols_to_keep]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.preprocessing_warnings.append(
                f"Dropped non-critical columns for AIF360 compatibility: {cols_to_drop}"
            )
        

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
        aif360 StandardDataset için verileri SIRAYLA sayısalla hazırla.
        
        İşlem:
        1. StringDtype/str tüm kalıntılarını temizle
        2. Target ve protected attributes sayısal olduğundan emin ol
        3. df.apply(pd.to_numeric) ile TÜM sütunları sayısala zorla
        4. StandardDataset oluştur
        """
        df_encoded = self.df.copy()
        
        # STEP 1: Clean all StringDtype/str remnants
        for col in df_encoded.columns:
            dtype_str = str(df_encoded[col].dtype)
            
            if 'string' in dtype_str.lower() or dtype_str == 'str':
                values = df_encoded[col].values
                df_encoded[col] = pd.Series(
                    [str(x) if pd.notna(x) else x for x in values],
                    index=df_encoded.index,
                    dtype='object'
                )
        
        # STEP 2: Ensure target column is numeric
        if self.label_name in df_encoded.columns:
            df_encoded[self.label_name] = pd.to_numeric(
                df_encoded[self.label_name],
                errors='coerce'
            ).astype('int64')
        
        # STEP 3: Ensure all protected attributes are numeric (already done in _encode_protected_attributes)
        for attr in self.protected_attributes:
            if attr in df_encoded.columns:
                df_encoded[attr] = pd.to_numeric(
                    df_encoded[attr],
                    errors='coerce'
                ).astype('int64')
        
        # STEP 4: FINAL ATOMIC CONVERSION - Force ALL columns to numeric
        # This is the nuclear option to eliminate any non-numeric values
        df_numeric = df_encoded.apply(pd.to_numeric, errors='coerce')
        
        # Check for NaN after coercion and warn
        nan_after_coerce = df_numeric.isna().sum()
        if nan_after_coerce.sum() > 0:
            cols_with_nan = nan_after_coerce[nan_after_coerce > 0].index.tolist()
            self.preprocessing_warnings.append(
                f"Warning: Coercion to numeric introduced NaN in columns: {cols_with_nan}"
            )
            # Drop rows with any NaN after coercion
            df_numeric = df_numeric.dropna()
        
        df_encoded = df_numeric
        
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
        try:
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
            
            # Get protected attribute and labels - FORCE NUMERIC
            protected_attr_col = self.protected_attributes[0]
            protected_attr = pd.to_numeric(
                self.df[protected_attr_col],
                errors='coerce'
            ).astype('int64').values
            
            labels = pd.to_numeric(
                self.df[self.label_name],
                errors='coerce'
            ).astype('int64').values
            
            # Ensure y_pred is numeric
            if self.y_pred is not None:
                self.y_pred = self.y_pred.astype('int64')
            
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
                "protected_attributes": self.protected_attributes,
                "preprocessing_warnings": self.preprocessing_warnings
            }
            
            return AuditResult(
                demographic_parity_difference=dpd,
                equalized_odds_difference=eod,
                disparate_impact_ratio=dir_ratio,
                risk_level=risk_level.value,
                metric_details=metric_details
            )
        
        except Exception as e:
            # Wrap AIF360/Scikit-learn errors with user-friendly messages
            error_msg = str(e)
            
            if "StringDtype" in error_msg or "string" in error_msg.lower():
                raise ValueError(
                    f"Data Type Error: {error_msg}\n\n"
                    "This usually means your CSV has a column with incompatible string encoding. "
                    "The system should have auto-fixed this, but try:\n"
                    "1. Re-save your CSV in UTF-8 format\n"
                    "2. Ensure no column headers have special characters"
                ) from e
            
            elif "binary" in error_msg.lower():
                raise ValueError(
                    f"Binary Encoding Error: {error_msg}\n\n"
                    "The target column must have exactly 2 unique values (e.g., 'Yes'/'No' or 'Hired'/'Not Hired')"
                ) from e
            
            elif "shape" in error_msg.lower() or "dimension" in error_msg.lower():
                raise ValueError(
                    f"Data Shape Error: {error_msg}\n\n"
                    "This often means you have missing values or mismatched column sizes. "
                    "Check that:\n"
                    "1. All rows have values in the target and protected attribute columns\n"
                    "2. Column names are spelled correctly"
                ) from e
            
            else:
                # Generic error - re-raise withpreprocessing context
                raise ValueError(
                    f"Audit Error: {error_msg}\n\n"
                    f"Preprocessing applied:\n" + 
                    "\n".join(f"  • {w}" for w in self.preprocessing_warnings) if self.preprocessing_warnings else "  (No preprocessing needed)"
                ) from e

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
        report = result.to_dict()
        
        # Add preprocessing warnings to report
        report['preprocessing_info'] = {
            'warnings': self.auditor.preprocessing_warnings,
            'samples_removed': self.auditor.original_sample_size - self.auditor.standard_dataset.features.shape[0] 
                                if self.auditor.standard_dataset else 0
        }
        
        if as_json:
            return json.dumps(report, indent=2, default=str)
        else:
            return report

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
            "disparate_impact_ratio": result.disparate_impact_ratio,
            "preprocessing_warnings": self.auditor.preprocessing_warnings
        }

    def get_preprocessing_warnings(self) -> List[str]:
        """
        Veri ön işleme sırasında karşılaşılan uyarıları ve dönüşümleri döndür.
        
        Faydalı kullanım alanları:
        - Streamlit dashboard'ta kullanıcıya geri bildirim göstermek
        - Denetim raporunda veri kalitesi notları eklemek
        - Veri dönüşümlerini takip etmek
        
        Döndürüm:
            List[str]: Uyarı mesajlarının listesi (varsa boş liste)
        """
        return self.auditor.preprocessing_warnings

    def temporal_bias_analysis(
        self,
        df: pd.DataFrame,
        time_column: str
    ) -> pd.DataFrame:
        """
        Panel Data Analizi: Zaman içinde önyargı trendlerini analiz et.
        
        Veri setini zaman dönemine göre gruplandırarak her dönem için
        Disparate Impact Oranı ve Demografik Parite'yi hesaplar.
        Trend görselleme için CSV veya Plotly'ye aktarma için kullanışlıdır.
        
        Parametreler:
            df: Giriş pandas DataFrame'i (temporal_bias_analysis yapılmış veri)
            time_column: Zaman dönemlerini içeren sütun adı (örn: 'year', 'quarter', 'month')
        
        Döndürüm:
            pd.DataFrame: Sütunları içeren:
                - time_period: Zaman dönemi değeri
                - disparate_impact_ratio: Her dönem için DIR
                - demographic_parity_difference: Her dönem için DPD
                - sample_size: Her dönem için örnek sayısı
                - risk_level: Her dönem için risk sınıflandırması
        
        Yükseltir:
            ValueError: time_column DataFrame'de yoksa veya veri geçersizse
        """
        if time_column not in df.columns:
            raise ValueError(
                f"Time column '{time_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Get the base auditor's protected attribute and label
        protected_attr_col = self.auditor.protected_attributes[0]
        label_col = self.auditor.label_name
        
        # Ensure required columns exist
        if label_col not in df.columns or protected_attr_col not in df.columns:
            raise ValueError(
                f"Label '{label_col}' or Protected Attribute '{protected_attr_col}' "
                "not found in DataFrame"
            )
        
        temporal_results = []
        
        # Group by time period and calculate metrics for each
        grouped = df.groupby(time_column, sort=False)
        
        for time_period, group_df in grouped:
            try:
                # Create a temporary auditor for this time period
                temp_auditor = FairnessAuditor(
                    df=group_df,
                    label_name=label_col,
                    protected_attributes=[protected_attr_col],
                    thresholds=self.auditor.thresholds
                )
                
                # Run audit for this period
                result = temp_auditor.auditor.audit()
                
                temporal_results.append({
                    'time_period': str(time_period),
                    'disparate_impact_ratio': result.disparate_impact_ratio,
                    'demographic_parity_difference': result.demographic_parity_difference,
                    'equalized_odds_difference': result.equalized_odds_difference,
                    'sample_size': len(group_df),
                    'risk_level': result.risk_level
                })
            except (ValueError, Exception) as e:
                # Log warning but continue with other periods
                warnings.warn(
                    f"Could not process time period '{time_period}': {str(e)}"
                )
                continue
        
        if not temporal_results:
            raise ValueError(
                f"No valid time periods found. Could not calculate temporal metrics."
            )
        
        # Convert to DataFrame and sort by time period
        temporal_df = pd.DataFrame(temporal_results)
        
        try:
            # Try to convert time_period to numeric for sorting
            temporal_df['time_period_sort'] = pd.to_numeric(
                temporal_df['time_period'],
                errors='coerce'
            )
            temporal_df = temporal_df.sort_values('time_period_sort').drop('time_period_sort', axis=1)
        except Exception:
            # If conversion fails, keep original order
            pass
        
        return temporal_df


# Örnek kullanım ve yardımcı fonksiyonlar
if __name__ == "__main__":
    # Bu bölüm yalnızca gösterim amaçlıdır
    print("FairLens Fairness Engine Modülü")
    print("Bu modülü içe aktarın ve FairnessAuditor sınıfını kullanın")
