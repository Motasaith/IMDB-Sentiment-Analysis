"""
Enhanced Data Cleaning Utilities for Data Analysis Web App
Phase 2: Data Cleaning and Preprocessing Tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataCleaningTools:
    """Comprehensive data cleaning utilities for the web app"""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame"""
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []

    def get_missing_info(self) -> Dict[str, Any]:
        """Get comprehensive missing value information"""
        missing_info = {}

        # Missing values per column
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100

        missing_info['total_missing'] = missing_count.sum()
        missing_info['missing_by_column'] = {
            'column': missing_count.index.tolist(),
            'count': missing_count.values.tolist(),
            'percentage': missing_percent.values.tolist()
        }

        # Columns with missing values
        missing_info['columns_with_missing'] = missing_count[missing_count > 0].index.tolist()
        missing_info['columns_without_missing'] = missing_count[missing_count == 0].index.tolist()

        return missing_info

    def handle_missing_values(self, strategy: str = 'drop',
                            columns: Optional[List[str]] = None,
                            fill_value: Any = None) -> pd.DataFrame:
        """Handle missing values with various strategies"""

        df_clean = self.df.copy()

        if columns is None:
            columns = self.df.columns.tolist()

        if strategy == 'drop':
            # Drop rows with missing values
            df_clean = df_clean.dropna(subset=columns)
            self.cleaning_log.append(f"Dropped rows with missing values in columns: {columns}")

        elif strategy == 'drop_columns':
            # Drop columns with missing values
            df_clean = df_clean.drop(columns=columns)
            self.cleaning_log.append(f"Dropped columns with missing values: {columns}")

        elif strategy == 'fill_mean':
            # Fill with mean (numeric columns only)
            numeric_columns = df_clean[columns].select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            self.cleaning_log.append(f"Filled missing values with mean in numeric columns: {numeric_columns}")

        elif strategy == 'fill_median':
            # Fill with median (numeric columns only)
            numeric_columns = df_clean[columns].select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            self.cleaning_log.append(f"Filled missing values with median in numeric columns: {numeric_columns}")

        elif strategy == 'fill_mode':
            # Fill with mode (categorical columns)
            categorical_columns = df_clean[columns].select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown')
            self.cleaning_log.append(f"Filled missing values with mode in categorical columns: {categorical_columns}")

        elif strategy == 'fill_constant':
            # Fill with constant value
            if fill_value is not None:
                for col in columns:
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(fill_value)
                self.cleaning_log.append(f"Filled missing values with constant '{fill_value}' in columns: {columns}")

        elif strategy == 'forward_fill':
            # Forward fill
            df_clean[columns] = df_clean[columns].fillna(method='ffill')
            self.cleaning_log.append(f"Applied forward fill to columns: {columns}")

        elif strategy == 'backward_fill':
            # Backward fill
            df_clean[columns] = df_clean[columns].fillna(method='bfill')
            self.cleaning_log.append(f"Applied backward fill to columns: {columns}")

        self.df = df_clean
        return df_clean

    def remove_duplicates(self, subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""

        df_clean = self.df.copy()

        if subset is None:
            # Remove all duplicates
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(keep=keep)
            removed_count = initial_count - len(df_clean)
        else:
            # Remove duplicates based on specific columns
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
            removed_count = initial_count - len(df_clean)

        self.cleaning_log.append(f"Removed {removed_count} duplicate rows (keep='{keep}', subset={subset})")
        self.df = df_clean
        return df_clean

    def detect_outliers(self, method: str = 'iqr',
                       columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers using various methods"""

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {
            'method': method,
            'columns': columns,
            'outliers': {}
        }

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == 'iqr':
                # IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_info['outliers'][col] = {
                    'count': len(outliers),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'indices': outliers.index.tolist()
                }

            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > 3]  # 3 standard deviations
                outlier_info['outliers'][col] = {
                    'count': len(outliers),
                    'threshold': 3,
                    'indices': outliers.index.tolist()
                }

        return outlier_info

    def handle_outliers(self, method: str = 'remove',
                       columns: Optional[List[str]] = None,
                       outlier_method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers by removing or capping"""

        df_clean = self.df.copy()

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = self.detect_outliers(outlier_method, columns)

        for col in columns:
            if col not in df_clean.columns:
                continue

            if method == 'remove':
                # Remove outliers
                outlier_indices = outlier_info['outliers'][col]['indices']
                df_clean = df_clean.drop(index=outlier_indices)
                self.cleaning_log.append(f"Removed {len(outlier_indices)} outliers from column '{col}'")

            elif method == 'cap':
                # Cap outliers
                if outlier_method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound,
                                           np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col]))
                    self.cleaning_log.append(f"Capped outliers in column '{col}' using IQR method")

        self.df = df_clean
        return df_clean

    def convert_data_types(self, type_conversions: Dict[str, str]) -> pd.DataFrame:
        """Convert data types for specified columns"""

        df_clean = self.df.copy()

        for column, target_type in type_conversions.items():
            if column not in df_clean.columns:
                continue

            try:
                if target_type == 'int':
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce').astype('Int64')
                elif target_type == 'float':
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                elif target_type == 'string':
                    df_clean[column] = df_clean[column].astype('string')
                elif target_type == 'category':
                    df_clean[column] = df_clean[column].astype('category')
                elif target_type == 'datetime':
                    df_clean[column] = pd.to_datetime(df_clean[column], errors='coerce')
                elif target_type == 'boolean':
                    df_clean[column] = df_clean[column].astype('boolean')

                self.cleaning_log.append(f"Converted column '{column}' to {target_type}")

            except Exception as e:
                self.cleaning_log.append(f"Failed to convert column '{column}' to {target_type}: {str(e)}")

        self.df = df_clean
        return df_clean

    def get_column_insights(self, top_n: int = 10) -> Dict[str, Any]:
        """Get comprehensive insights for each column"""

        insights = {}

        for col in self.df.columns:
            col_data = self.df[col]
            col_insights = {
                'column_name': col,
                'data_type': str(col_data.dtype),
                'missing_count': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }

            # Type-specific insights
            if col_data.dtype in ['int64', 'float64']:
                col_insights.update({
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'type': 'numeric'
                })

            elif col_data.dtype == 'object' or col_data.dtype.name == 'string':
                # Get top N most frequent values
                top_values = col_data.value_counts().head(top_n)
                col_insights.update({
                    'top_values': top_values.to_dict(),
                    'type': 'categorical'
                })

            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_insights.update({
                    'min_date': str(col_data.min()),
                    'max_date': str(col_data.max()),
                    'type': 'datetime'
                })

            insights[col] = col_insights

        return insights

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations performed"""

        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'operations_performed': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log
        }

        return summary

    def reset_data(self) -> pd.DataFrame:
        """Reset to original data"""
        self.df = self.df.copy()  # This should be the original data
        self.cleaning_log = []
        return self.df
