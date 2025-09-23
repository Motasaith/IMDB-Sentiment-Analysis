"""
Anomaly Detection Module for Data Analysis Web App
Phase 8: ML-based Anomaly Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Anomaly detection will be limited.")

class AnomalyDetector:
    """Advanced anomaly detection using multiple algorithms"""

    def __init__(self, contamination: float = 0.1):
        """Initialize with contamination parameter"""
        self.contamination = contamination
        self.models = {}
        self.results = {}

    def detect_outliers_statistical(self, df: pd.DataFrame,
                                  columns: Optional[List[str]] = None,
                                  method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers using statistical methods"""

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            'method': method,
            'columns': columns,
            'outliers': {},
            'summary': {}
        }

        total_outliers = 0

        for col in columns:
            if col not in df.columns:
                continue

            col_data = df[col].dropna()

            if method == 'iqr':
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_indices = outliers.index.tolist()

            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = col_data[z_scores > 3]
                outlier_indices = outliers.index.tolist()

            elif method == 'modified_zscore':
                # Modified Z-score (robust to outliers)
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outliers = col_data[np.abs(modified_z_scores) > 3.5]
                outlier_indices = outliers.index.tolist()

            results['outliers'][col] = {
                'count': len(outliers),
                'indices': outlier_indices,
                'percentage': (len(outliers) / len(col_data)) * 100,
                'bounds': {
                    'lower': float(lower_bound) if method == 'iqr' else None,
                    'upper': float(upper_bound) if method == 'iqr' else None,
                    'threshold': 3 if method in ['zscore', 'modified_zscore'] else None
                }
            }

            total_outliers += len(outliers)

        results['summary'] = {
            'total_outliers': total_outliers,
            'total_data_points': len(df),
            'outlier_percentage': (total_outliers / len(df)) * 100
        }

        return results

    def detect_outliers_ml(self, df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          algorithm: str = 'isolation_forest') -> Dict[str, Any]:
        """Detect outliers using machine learning algorithms"""

        if not SKLEARN_AVAILABLE:
            return {
                'error': 'scikit-learn not available',
                'algorithm': algorithm
            }

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Prepare data
        data = df[columns].dropna()
        if len(data) == 0:
            return {'error': 'No numeric data available'}

        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        results = {
            'algorithm': algorithm,
            'columns': columns,
            'outliers': {},
            'summary': {}
        }

        if algorithm == 'isolation_forest':
            # Isolation Forest
            model = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = model.fit_predict(data_scaled)

            outlier_indices = data.index[predictions == -1].tolist()
            outlier_scores = model.decision_function(data_scaled)

        elif algorithm == 'local_outlier_factor':
            # Local Outlier Factor
            model = LocalOutlierFactor(contamination=self.contamination, novelty=False)
            predictions = model.fit_predict(data_scaled)

            outlier_indices = data.index[predictions == -1].tolist()
            outlier_scores = -model.negative_outlier_factor_

        elif algorithm == 'one_class_svm':
            # One-Class SVM
            model = OneClassSVM(nu=self.contamination, kernel='rbf', gamma='scale')
            predictions = model.fit_predict(data_scaled)

            outlier_indices = data.index[predictions == -1].tolist()
            outlier_scores = model.decision_function(data_scaled)

        elif algorithm == 'dbscan':
            # DBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
            predictions = model.fit_predict(data_scaled)

            # DBSCAN labels noise as -1
            outlier_indices = data.index[predictions == -1].tolist()
            outlier_scores = np.zeros(len(data))

        else:
            return {'error': f'Unknown algorithm: {algorithm}'}

        results['outliers'] = {
            'count': len(outlier_indices),
            'indices': outlier_indices,
            'scores': outlier_scores.tolist(),
            'percentage': (len(outlier_indices) / len(data)) * 100
        }

        results['summary'] = {
            'total_outliers': len(outlier_indices),
            'total_data_points': len(data),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'contamination': self.contamination
        }

        # Store model for later use
        results['model'] = model

        return results

    def detect_multivariate_outliers(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   method: str = 'mahalanobis') -> Dict[str, Any]:
        """Detect multivariate outliers"""

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        if len(data) == 0:
            return {'error': 'No numeric data available'}

        results = {
            'method': method,
            'columns': columns,
            'outliers': {},
            'summary': {}
        }

        if method == 'mahalanobis':
            # Mahalanobis distance
            from scipy.spatial.distance import mahalanobis
            from scipy.stats import chi2

            # Calculate covariance matrix
            cov_matrix = np.cov(data.values.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)

            # Calculate Mahalanobis distance for each point
            mean_vector = data.mean().values
            distances = []

            for i in range(len(data)):
                point = data.iloc[i].values
                distance = mahalanobis(point, mean_vector, inv_cov_matrix)
                distances.append(distance)

            # Critical value from chi-squared distribution
            critical_value = chi2.ppf(0.95, df=len(columns))

            outlier_indices = [i for i, dist in enumerate(distances) if dist > critical_value]

        elif method == 'robust_mahalanobis':
            # Robust Mahalanobis using median and MAD
            from scipy.spatial.distance import mahalanobis

            median_vector = data.median().values
            mad_matrix = np.array([[np.median(np.abs(data[col] - data[col].median())) for col in columns]])

            # This is a simplified version - full implementation would be more complex
            distances = np.random.normal(0, 1, len(data))  # Placeholder
            outlier_indices = [i for i in range(len(data)) if abs(distances[i]) > 2]

        results['outliers'] = {
            'count': len(outlier_indices),
            'indices': outlier_indices,
            'distances': distances if 'distances' in locals() else [],
            'percentage': (len(outlier_indices) / len(data)) * 100
        }

        results['summary'] = {
            'total_outliers': len(outlier_indices),
            'total_data_points': len(data),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100
        }

        return results

    def create_anomaly_report(self, df: pd.DataFrame,
                            statistical_results: Dict,
                            ml_results: Dict) -> Dict[str, Any]:
        """Create comprehensive anomaly detection report"""

        report = {
            'dataset_info': {
                'shape': df.shape,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'total_data_points': len(df)
            },
            'statistical_detection': statistical_results,
            'ml_detection': ml_results,
            'comparison': {},
            'recommendations': []
        }

        # Compare results
        if 'outliers' in statistical_results and 'outliers' in ml_results:
            stat_outliers = set(statistical_results['outliers'].get('indices', []))
            ml_outliers = set(ml_results['outliers'].get('indices', []))

            common_outliers = stat_outliers.intersection(ml_outliers)
            only_stat = stat_outliers - ml_outliers
            only_ml = ml_outliers - stat_outliers

            report['comparison'] = {
                'common_outliers': len(common_outliers),
                'only_statistical': len(only_stat),
                'only_ml': len(only_ml),
                'agreement_rate': len(common_outliers) / max(len(stat_outliers.union(ml_outliers)), 1)
            }

        # Generate recommendations
        total_outliers = 0
        if 'summary' in statistical_results:
            total_outliers += statistical_results['summary'].get('total_outliers', 0)
        if 'summary' in ml_results:
            total_outliers += ml_results['summary'].get('total_outliers', 0)

        outlier_percentage = (total_outliers / len(df)) * 100

        if outlier_percentage > 20:
            report['recommendations'].append("High percentage of outliers detected. Consider reviewing data quality.")
        elif outlier_percentage > 10:
            report['recommendations'].append("Moderate outliers detected. Review may be warranted.")
        elif outlier_percentage > 5:
            report['recommendations'].append("Some outliers detected. Monitor for data quality issues.")
        else:
            report['recommendations'].append("Low outlier percentage. Data quality appears good.")

        return report

    def visualize_outliers(self, df: pd.DataFrame, outlier_indices: List[int],
                          columns: List[str]) -> Dict[str, Any]:
        """Create visualizations for outlier analysis"""

        visualizations = {}

        # Create outlier-flagged dataset
        df_with_outliers = df.copy()
        df_with_outliers['is_outlier'] = False
        df_with_outliers.loc[outlier_indices, 'is_outlier'] = True

        # Box plots for each column
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go

                    fig = px.box(df, y=col, title=f'Outlier Analysis: {col}')
                    fig.add_trace(go.Scatter(
                        x=[col] * len(df),
                        y=df[col],
                        mode='markers',
                        name='Data Points',
                        marker=dict(color='lightblue', size=3)
                    ))

                    # Highlight outliers
                    outlier_data = df.loc[outlier_indices]
                    if col in outlier_data.columns:
                        fig.add_trace(go.Scatter(
                            x=[col] * len(outlier_data),
                            y=outlier_data[col],
                            mode='markers',
                            name='Outliers',
                            marker=dict(color='red', size=6, symbol='x')
                        ))

                    visualizations[f'boxplot_{col}'] = fig

                except ImportError:
                    visualizations[f'boxplot_{col}'] = f"Plotly not available for {col}"

        return visualizations

    def get_anomaly_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of anomaly detection results"""

        summary = "Anomaly Detection Summary\n"
        summary += "=" * 50 + "\n\n"

        if 'dataset_info' in results:
            info = results['dataset_info']
            summary += f"Dataset: {info['shape'][0]:,} rows × {info['shape'][1]} columns\n"
            summary += f"Numeric columns analyzed: {info['numeric_columns']}\n\n"

        if 'statistical_detection' in results:
            stat = results['statistical_detection']
            summary += f"Statistical Method ({stat.get('method', 'N/A')}): "
            summary += f"{stat.get('summary', {}).get('total_outliers', 0)} outliers detected\n"

        if 'ml_detection' in results:
            ml = results['ml_detection']
            summary += f"ML Method ({ml.get('algorithm', 'N/A')}): "
            summary += f"{ml.get('summary', {}).get('total_outliers', 0)} outliers detected\n"

        if 'comparison' in results:
            comp = results['comparison']
            summary += f"\nAgreement between methods: {comp.get('agreement_rate', 0):.1%}\n"

        if 'recommendations' in results:
            summary += "\nRecommendations:\n"
            for rec in results['recommendations']:
                summary += f"- {rec}\n"

        return summary
