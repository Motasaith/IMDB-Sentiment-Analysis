"""
Interactive Data Explorer for Data Analysis Web App
Phase 9: Advanced Interactive Data Tables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class InteractiveDataExplorer:
    """Interactive data exploration with advanced filtering and search"""

    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame"""
        self.df = df.copy()
        self.filtered_df = df.copy()
        self.filters = {}
        self.search_term = ""
        self.sort_config = {}

    def search_data(self, search_term: str,
                   columns: Optional[List[str]] = None,
                   case_sensitive: bool = False) -> pd.DataFrame:
        """Search data across specified columns"""

        if not search_term.strip():
            self.filtered_df = self.df.copy()
            self.search_term = ""
            return self.filtered_df

        self.search_term = search_term

        if columns is None:
            columns = self.df.columns.tolist()

        # Create search pattern
        if not case_sensitive:
            search_term = search_term.lower()

        # Filter rows that contain the search term in any of the specified columns
        mask = pd.Series(False, index=self.df.index)

        for col in columns:
            if col in self.df.columns:
                col_data = self.df[col].astype(str)
                if not case_sensitive:
                    col_data = col_data.str.lower()

                col_mask = col_data.str.contains(search_term, na=False, regex=False)
                mask = mask | col_mask

        self.filtered_df = self.df[mask].copy()
        return self.filtered_df

    def apply_filters(self, filters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Apply multiple filters to the data"""

        self.filters = filters
        filtered_df = self.df.copy()

        for column, filter_config in filters.items():
            if column not in filtered_df.columns:
                continue

            filter_type = filter_config.get('type', 'none')
            filter_value = filter_config.get('value')

            if filter_type == 'none' or filter_value is None:
                continue

            col_data = filtered_df[column]

            if filter_type == 'range':
                # Numeric range filter
                if pd.api.types.is_numeric_dtype(col_data):
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[
                        (col_data >= min_val) & (col_data <= max_val)
                    ]

            elif filter_type == 'select':
                # Categorical selection filter
                if isinstance(filter_value, list):
                    filtered_df = filtered_df[col_data.isin(filter_value)]

            elif filter_type == 'text':
                # Text-based filter
                if isinstance(filter_value, str):
                    filtered_df = filtered_df[
                        col_data.astype(str).str.contains(filter_value, case=False, na=False)
                    ]

            elif filter_type == 'date_range':
                # Date range filter
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    start_date, end_date = filter_value
                    filtered_df = filtered_df[
                        (col_data >= start_date) & (col_data <= end_date)
                    ]

            elif filter_type == 'boolean':
                # Boolean filter
                if filter_value in [True, False]:
                    filtered_df = filtered_df[col_data == filter_value]

            elif filter_type == 'null':
                # Null/Non-null filter
                if filter_value == 'null':
                    filtered_df = filtered_df[col_data.isnull()]
                elif filter_value == 'not_null':
                    filtered_df = filtered_df[col_data.notnull()]

        self.filtered_df = filtered_df
        return filtered_df

    def sort_data(self, sort_config: Dict[str, str]) -> pd.DataFrame:
        """Sort data by specified columns"""

        self.sort_config = sort_config

        if not sort_config:
            return self.filtered_df

        sort_columns = []
        ascending = []

        for col, direction in sort_config.items():
            if col in self.filtered_df.columns:
                sort_columns.append(col)
                ascending.append(direction.lower() == 'asc')

        if sort_columns:
            self.filtered_df = self.filtered_df.sort_values(
                by=sort_columns,
                ascending=ascending
            ).reset_index(drop=True)

        return self.filtered_df

    def paginate_data(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Paginate the filtered data"""

        total_rows = len(self.filtered_df)
        total_pages = (total_rows + page_size - 1) // page_size

        # Adjust page number if out of bounds
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        paginated_data = self.filtered_df.iloc[start_idx:end_idx]

        pagination_info = {
            'current_page': page,
            'page_size': page_size,
            'total_rows': total_rows,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'start_index': start_idx + 1,
            'end_index': end_idx
        }

        return {
            'data': paginated_data,
            'pagination': pagination_info
        }

    def get_column_statistics(self, column: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific column"""

        if column not in self.df.columns:
            return {'error': f'Column {column} not found'}

        col_data = self.df[column].dropna()

        stats = {
            'column_name': column,
            'data_type': str(self.df[column].dtype),
            'total_count': len(self.df),
            'non_null_count': len(col_data),
            'null_count': len(self.df) - len(col_data),
            'null_percentage': ((len(self.df) - len(col_data)) / len(self.df)) * 100
        }

        # Type-specific statistics
        if pd.api.types.is_numeric_dtype(self.df[column]):
            stats.update({
                'type': 'numeric',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75))
            })

        elif pd.api.types.is_categorical_dtype(self.df[column]) or self.df[column].dtype == 'object':
            value_counts = col_data.value_counts()
            stats.update({
                'type': 'categorical',
                'unique_count': len(value_counts),
                'unique_percentage': (len(value_counts) / len(col_data)) * 100,
                'top_values': value_counts.head(10).to_dict(),
                'least_common': value_counts.tail(5).to_dict()
            })

        elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
            stats.update({
                'type': 'datetime',
                'min_date': str(col_data.min()),
                'max_date': str(col_data.max()),
                'date_range_days': (col_data.max() - col_data.min()).days
            })

        elif pd.api.types.is_bool_dtype(self.df[column]):
            bool_counts = col_data.value_counts()
            stats.update({
                'type': 'boolean',
                'true_count': int(bool_counts.get(True, 0)),
                'false_count': int(bool_counts.get(False, 0)),
                'true_percentage': (bool_counts.get(True, 0) / len(col_data)) * 100
            })

        return stats

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""

        report = {
            'overall_quality': {},
            'column_quality': {},
            'recommendations': []
        }

        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()

        report['overall_quality'] = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': (duplicate_rows / self.df.shape[0]) * 100,
            'completeness_score': ((total_cells - missing_cells) / total_cells) * 100
        }

        # Column-level quality
        for col in self.df.columns:
            col_data = self.df[col]
            col_report = self.get_column_statistics(col)

            # Quality score for column
            quality_score = 100
            issues = []

            # Missing values penalty
            if col_report['null_percentage'] > 0:
                quality_score -= col_report['null_percentage'] * 0.5
                issues.append(f"Missing values: {col_report['null_percentage']:.1f}%")

            # Duplicate values penalty (for non-key columns)
            if col_report.get('type') == 'categorical':
                unique_pct = col_report.get('unique_percentage', 0)
                if unique_pct < 10:
                    quality_score -= (10 - unique_pct) * 2
                    issues.append(f"Low uniqueness: {unique_pct:.1f}%")

            col_report['quality_score'] = max(0, quality_score)
            col_report['issues'] = issues
            report['column_quality'][col] = col_report

        # Generate recommendations
        overall_quality = report['overall_quality']

        if overall_quality['missing_percentage'] > 20:
            report['recommendations'].append("High missing data percentage. Consider data imputation or collection.")
        elif overall_quality['missing_percentage'] > 10:
            report['recommendations'].append("Moderate missing data. Review and handle missing values.")

        if overall_quality['duplicate_percentage'] > 5:
            report['recommendations'].append("High duplicate rate. Consider deduplication.")

        # Column-specific recommendations
        for col, col_info in report['column_quality'].items():
            if col_info['quality_score'] < 50:
                report['recommendations'].append(f"Column '{col}' has low quality score. Review data quality.")

        return report

    def export_filtered_data(self, format: str = 'csv',
                           filename: str = 'filtered_data') -> Dict[str, Any]:
        """Export filtered data to various formats"""

        try:
            if format.lower() == 'csv':
                filepath = f"{filename}.csv"
                self.filtered_df.to_csv(filepath, index=False)

            elif format.lower() == 'excel':
                filepath = f"{filename}.xlsx"
                self.filtered_df.to_excel(filepath, index=False)

            elif format.lower() == 'json':
                filepath = f"{filename}.json"
                self.filtered_df.to_json(filepath, orient='records', indent=2)

            else:
                return {
                    'success': False,
                    'error': f'Unsupported format: {format}'
                }

            return {
                'success': True,
                'filepath': filepath,
                'format': format,
                'rows': len(self.filtered_df),
                'columns': len(self.filtered_df.columns)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_filter_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for filtering options based on data"""

        suggestions = {
            'numeric_filters': [],
            'categorical_filters': [],
            'date_filters': [],
            'boolean_filters': []
        }

        for col in self.df.columns:
            col_data = self.df[col].dropna()

            if pd.api.types.is_numeric_dtype(self.df[col]):
                suggestions['numeric_filters'].append({
                    'column': col,
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median())
                })

            elif pd.api.types.is_categorical_dtype(self.df[col]) or self.df[col].dtype == 'object':
                unique_values = col_data.unique()
                if len(unique_values) <= 20:  # Only suggest if not too many unique values
                    suggestions['categorical_filters'].append({
                        'column': col,
                        'unique_values': unique_values.tolist(),
                        'value_counts': col_data.value_counts().to_dict()
                    })

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                suggestions['date_filters'].append({
                    'column': col,
                    'min_date': str(col_data.min()),
                    'max_date': str(col_data.max())
                })

            elif pd.api.types.is_bool_dtype(self.df[col]):
                suggestions['boolean_filters'].append({
                    'column': col,
                    'true_count': int((col_data == True).sum()),
                    'false_count': int((col_data == False).sum())
                })

        return suggestions

    def reset_filters(self) -> pd.DataFrame:
        """Reset all filters and return to original data"""

        self.filtered_df = self.df.copy()
        self.filters = {}
        self.search_term = ""
        self.sort_config = {}

        return self.filtered_df
