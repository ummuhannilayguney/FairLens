"""
FairLens Visualization Utilities

Bu modül, Streamlit dashboard'u için Plotly-tabanlı görselleme fonksiyonları sağlar.
Adaletsizlik metriklerini etkileşimli ve renkli grafiklerle sunar.

Yazarlar: FairLens Geliştirme Ekibi
Sürüm: 1.0.0
Lisans: MIT
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


def create_outcome_rates_chart(
    df: pd.DataFrame,
    label_col: str,
    protected_attr_col: str,
    title: str = "Outcome Rates by Group"
) -> go.Figure:
    """
    Korumalı ve ayrıcalıklı gruplar arasında sonuç oranlarının 
    karşılaştıran bar grafiği oluştur.
    
    Parametreler:
        df: İşlenmiş veri seti
        label_col: Sonuç/etiket sütunu adı
        protected_attr_col: Korumalı öznitelik sütunu adı
        title: Grafik başlığı
    
    Döndürüm:
        plotly.graph_objects.Figure: Etkileşimli bar grafiği
    """
    # Calculate outcome rates by group
    group_rates = df.groupby(protected_attr_col)[label_col].apply(
        lambda x: (x == 1).sum() / len(x) * 100
    ).reset_index()
    
    group_rates.columns = ['Group', 'Outcome Rate (%)']
    group_rates['Group'] = group_rates['Group'].astype(str)
    
    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=group_rates['Group'],
                y=group_rates['Outcome Rate (%)'],
                marker=dict(
                    color=group_rates['Outcome Rate (%)'],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(color='darkgray', width=2)
                ),
                text=group_rates['Outcome Rate (%)'].round(2),
                textposition='outside',
                hovertemplate='<b>Group %{x}</b><br>Outcome Rate: %{y:.2f}%<extra></extra>'
            )
        ]
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Outcome Rate (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_bias_trend_chart(
    temporal_df: pd.DataFrame,
    title: str = "Bias Metrics Over Time"
) -> go.Figure:
    """
    Zaman içinde önyargı metriklerinin trendi gösteren 
    çok yönlü çizgi grafiği oluştur.
    
    temporal_bias_analysis tarafından döndürülen DataFrame'i kullanır.
    
    Parametreler:
        temporal_df: temporal_bias_analysis'ten gelen DataFrame
        title: Grafik başlığı
    
    Döndürüm:
        plotly.graph_objects.Figure: Etkileşimli çizgi grafiği
    """
    if temporal_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No temporal data available")
        return fig
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add Disparate Impact Ratio (primary axis)
    fig.add_trace(
        go.Scatter(
            x=temporal_df['time_period'],
            y=temporal_df['disparate_impact_ratio'],
            name='Disparate Impact Ratio',
            mode='lines+markers',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>DIR: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add a reference line for 80% rule
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="red",
        annotation_text="80% Rule Threshold",
        annotation_position="right",
        secondary_y=False
    )
    
    # Add Demographic Parity Difference (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=temporal_df['time_period'],
            y=temporal_df['demographic_parity_difference'],
            name='Demographic Parity Difference',
            mode='lines+markers',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>DPD: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        hovermode='x unified',
        template='plotly_white',
        height=450,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(title_text="Time Period")
    fig.update_yaxes(title_text="Disparate Impact Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Demographic Parity Difference", secondary_y=True)
    
    return fig


def create_disparate_impact_gauge(
    dir_score: float,
    title: str = "Disparate Impact Ratio (80% Rule)"
) -> go.Figure:
    """
    Disparate Impact Oranını göstermek için basit bir gösterge grafiği oluştur.
    
    Parametreler:
        dir_score: Disparate Impact Oranı (0-1+)
        title: Grafik başlığı
    
    Döndürüm:
        plotly.graph_objects.Figure: Etkileşimli gösterge grafiği
    """
    # Clamp the score to 0-1 for visualization
    clamped_score = min(dir_score, 1.0) * 100
    
    # Determine color and status
    if dir_score >= 0.8:
        color = 'green'
        status = 'PASS: ≥ 0.8'
    elif dir_score >= 0.7:
        color = 'yellow'
        status = 'WARNING: 0.7-0.8'
    else:
        color = 'red'
        status = 'FAIL: < 0.7'
    
    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=clamped_score,
                title={'text': title},
                delta={'reference': 80, 'prefix': "vs 80% threshold"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 70], 'color': 'lightcoral'},
                        {'range': [70, 80], 'color': 'lightyellow'},
                        {'range': [80, 100], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            )
        ]
    )
    
    fig.update_layout(
        height=350,
        paper_bgcolor='white',
        font={'size': 12},
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add status annotation
    fig.add_annotation(
        text=f"<b>Status: {status}</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.15,
        showarrow=False,
        font=dict(size=14, color=color),
        xanchor='center'
    )
    
    return fig


def create_equity_heatmap(
    df: pd.DataFrame,
    label_col: str,
    protected_attr_col: str,
    title: str = "Equity Analysis Heatmap"
) -> go.Figure:
    """
    İşe alım kararı matrisini göstermek için ısı haritası oluştur.
    Rows = Korumalı grup, Columns = Sonuç kategorileri
    
    Parametreler:
        df: İşlenmiş veri seti
        label_col: Sonuç/etiket sütunu adı
        protected_attr_col: Korumalı öznitelik sütunu adı
        title: Grafik başlığı
    
    Döndürüm:
        plotly.graph_objects.Figure: Etkileşimli ısı haritası
    """
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        df[protected_attr_col],
        df[label_col],
        margins=True
    )
    
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix.iloc[:-1, :-1].values,
            x=['Rejected', 'Hired'],
            y=[f'Group {i}' for i in confusion_matrix.index[:-1]],
            colorscale='Viridis',
            text=confusion_matrix.iloc[:-1, :-1].values,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Group: %{y}<br>Decision: %{x}<br>Count: %{z}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Decision",
        yaxis_title="Protected Status",
        height=350,
        template='plotly_white'
    )
    
    return fig


def create_risk_summary_table(
    metrics_dict: Dict[str, Any],
    temporal_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Metriklerin özet tablosunu oluştur.
    Streamlit'te gösterim için uygun bir DataFrame döndürür.
    
    Parametreler:
        metrics_dict: audit() tarafından döndürülen metric_details
        temporal_df: (İsteğe bağlı) temporal_bias_analysis DataFrame'i
    
    Döndürüm:
        pd.DataFrame: Özet metrikleri içeren tablo
    """
    summary_data = []
    
    # Add primary metrics
    if 'demographic_parity' in metrics_dict:
        dpd = metrics_dict['demographic_parity']
        summary_data.append({
            'Metric': 'Demographic Parity Difference',
            'Score': f"{dpd['score']:.4f}",
            'Threshold': f"{dpd['threshold']:.4f}",
            'Status': '✓ PASS' if dpd['score'] <= dpd['threshold'] else '✗ FAIL'
        })
    
    if 'equalized_odds' in metrics_dict:
        eod = metrics_dict['equalized_odds']
        summary_data.append({
            'Metric': 'Equalized Odds Difference',
            'Score': f"{eod['score']:.4f}",
            'Threshold': f"{eod['threshold']:.4f}",
            'Status': '✓ PASS' if eod['score'] <= eod['threshold'] else '✗ FAIL'
        })
    
    if 'disparate_impact' in metrics_dict:
        dir_info = metrics_dict['disparate_impact']
        summary_data.append({
            'Metric': 'Disparate Impact Ratio (80% Rule)',
            'Score': f"{dir_info['score']:.4f}",
            'Threshold': f"{dir_info['threshold']:.4f}",
            'Status': '✓ PASS' if dir_info['score'] >= dir_info['threshold'] else '✗ FAIL'
        })
    
    return pd.DataFrame(summary_data)


def format_metric_for_display(value: float, metric_type: str = 'default') -> str:
    """
    Metrik değerini insan tarafından okunabilir formatta formatla.
    
    Parametreler:
        value: Değer
        metric_type: 'percentage', 'ratio', veya 'default'
    
    Döndürüm:
        str: Formatlanmış değer
    """
    if metric_type == 'percentage':
        return f"{value * 100:.2f}%"
    elif metric_type == 'ratio':
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"


def get_risk_color(risk_level: str) -> str:
    """
    Risk seviyesine göre renk kodu döndür.
    
    Parametreler:
        risk_level: Risk seviyesi string'i ('Düşük', 'Orta', 'Yüksek')
    
    Döndürüm:
        str: Hex renk kodu
    """
    risk_colors = {
        'Low': '#2ecc71',      # Green
        'Düşük': '#2ecc71',    # Green (Turkish)
        'Medium': '#f39c12',   # Orange
        'Orta': '#f39c12',     # Orange (Turkish)
        'High': '#e74c3c',     # Red
        'Yüksek': '#e74c3c'    # Red (Turkish)
    }
    return risk_colors.get(risk_level, '#95a5a6')  # Gray for unknown


if __name__ == "__main__":
    print("FairLens Visualization Utilities")
    print("Bu modülü Streamlit app'inde içe aktarın")
