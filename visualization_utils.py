"""
Enhanced Visualization Utilities for Data Analysis Web App
Phase 5: Advanced Data Visualization Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    """Advanced visualization tools for data analysis"""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame"""
        self.df = df.copy()
        self.color_palette = px.colors.qualitative.Set3

    def get_column_types(self) -> Dict[str, List[str]]:
        """Categorize columns by data type"""

        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'boolean': []
        }

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                column_types['numeric'].append(col)
            elif pd.api.types.is_categorical_dtype(self.df[col]) or self.df[col].dtype == 'object':
                column_types['categorical'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_types['datetime'].append(col)
            elif pd.api.types.is_bool_dtype(self.df[col]):
                column_types['boolean'].append(col)

        return column_types

    def create_auto_dashboard(self, max_charts: int = 12) -> Dict[str, Any]:
        """Create automatic dashboard with relevant charts"""

        column_types = self.get_column_types()
        charts = {}

        # Numeric columns - histograms and correlations
        numeric_cols = column_types['numeric'][:4]  # Limit to first 4

        if len(numeric_cols) >= 2:
            # Correlation heatmap
            charts['correlation'] = self.create_correlation_heatmap(numeric_cols)

        for i, col in enumerate(numeric_cols):
            if i >= max_charts // 3:  # Limit charts
                break
            charts[f'histogram_{col}'] = self.create_histogram(col)

        # Categorical columns - bar charts
        categorical_cols = column_types['categorical'][:4]  # Limit to first 4

        for i, col in enumerate(categorical_cols):
            if i >= max_charts // 3:  # Limit charts
                break
            charts[f'barchart_{col}'] = self.create_bar_chart(col)

        # Boolean columns - pie charts
        boolean_cols = column_types['boolean'][:2]  # Limit to first 2

        for col in boolean_cols:
            charts[f'piechart_{col}'] = self.create_pie_chart(col)

        return charts

    def create_histogram(self, column: str,
                        bins: int = 30,
                        title: Optional[str] = None) -> go.Figure:
        """Create interactive histogram"""

        if title is None:
            title = f'Distribution of {column}'

        fig = px.histogram(
            self.df, x=column,
            nbins=bins,
            title=title,
            marginal='box'
        )

        fig.update_layout(
            showlegend=False,
            height=400
        )

        return fig

    def create_bar_chart(self, column: str,
                        top_n: int = 10,
                        title: Optional[str] = None) -> go.Figure:
        """Create interactive bar chart for categorical data"""

        if title is None:
            title = f'Top {top_n} values in {column}'

        # Get top N most frequent values
        top_values = self.df[column].value_counts().head(top_n)

        fig = px.bar(
            x=top_values.values,
            y=top_values.index,
            orientation='h',
            title=title,
            labels={'x': 'Count', 'y': column}
        )

        fig.update_layout(
            height=max(300, len(top_values) * 30),
            showlegend=False
        )

        return fig

    def create_pie_chart(self, column: str,
                        title: Optional[str] = None) -> go.Figure:
        """Create pie chart for boolean/categorical data"""

        if title is None:
            title = f'Distribution of {column}'

        value_counts = self.df[column].value_counts()

        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            hole=0.4
        )

        fig.update_layout(
            height=400,
            showlegend=True
        )

        return fig

    def create_correlation_heatmap(self, columns: List[str],
                                  title: str = "Correlation Heatmap") -> go.Figure:
        """Create interactive correlation heatmap"""

        corr_matrix = self.df[columns].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            height=500,
            width=500
        )

        return fig

    def create_scatter_plot(self, x_col: str, y_col: str,
                          color_col: Optional[str] = None,
                          size_col: Optional[str] = None,
                          title: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot"""

        if title is None:
            title = f'{x_col} vs {y_col}'

        fig = px.scatter(
            self.df, x=x_col, y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            trendline="ols" if color_col is None else None
        )

        fig.update_layout(
            height=500
        )

        return fig

    def create_interactive_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                                  color_col: Optional[str] = None,
                                  title: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot (alias for create_scatter_plot with df parameter)"""

        if title is None:
            title = f'{x_col} vs {y_col}'

        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            raise ValueError(f"Column '{x_col}' must be numeric for scatter plot. Current type: {df[x_col].dtype}")

        if not pd.api.types.is_numeric_dtype(df[y_col]):
            raise ValueError(f"Column '{y_col}' must be numeric for scatter plot. Current type: {df[y_col].dtype}")

        # Check if color column is provided and is categorical
        if color_col and not pd.api.types.is_numeric_dtype(df[color_col]):
            # For categorical color columns, use them as-is
            pass

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col,
            title=title,
            trendline="ols" if color_col is None else None
        )

        fig.update_layout(
            height=500
        )

        return fig

    def create_box_plot(self, x_col: str, y_col: str,
                       title: Optional[str] = None) -> go.Figure:
        """Create box plot"""

        if title is None:
            title = f'{y_col} by {x_col}'

        fig = px.box(
            self.df, x=x_col, y=y_col,
            title=title
        )

        fig.update_layout(
            height=400
        )

        return fig

    def create_violin_plot(self, x_col: str, y_col: str,
                          title: Optional[str] = None) -> go.Figure:
        """Create violin plot"""

        if title is None:
            title = f'{y_col} by {x_col}'

        fig = px.violin(
            self.df, x=x_col, y=y_col,
            title=title,
            box=True
        )

        fig.update_layout(
            height=400
        )

        return fig

    def create_line_chart(self, x_col: str, y_col: str,
                         title: Optional[str] = None) -> go.Figure:
        """Create line chart for time series data"""

        if title is None:
            title = f'{y_col} over {x_col}'

        fig = px.line(
            self.df, x=x_col, y=y_col,
            title=title
        )

        fig.update_layout(
            height=400
        )

        return fig

    def create_area_chart(self, x_col: str, y_cols: List[str],
                         title: Optional[str] = None) -> go.Figure:
        """Create stacked area chart"""

        if title is None:
            title = f'Trends for {", ".join(y_cols)}'

        fig = px.area(
            self.df, x=x_col, y=y_cols,
            title=title
        )

        fig.update_layout(
            height=400
        )

        return fig

    def create_treemap(self, path_cols: List[str], value_col: str,
                      title: Optional[str] = None) -> go.Figure:
        """Create treemap visualization"""

        if title is None:
            title = f'Treemap by {", ".join(path_cols)}'

        fig = px.treemap(
            self.df, path=path_cols, values=value_col,
            title=title
        )

        return fig

    def create_sunburst(self, path_cols: List[str], value_col: str,
                        title: Optional[str] = None) -> go.Figure:
        """Create sunburst chart"""

        if title is None:
            title = f'Sunburst by {", ".join(path_cols)}'

        fig = px.sunburst(
            self.df, path=path_cols, values=value_col,
            title=title
        )

        return fig

    def create_parallel_coordinates(self, columns: List[str],
                                   color_col: Optional[str] = None,
                                   title: Optional[str] = None) -> go.Figure:
        """Create parallel coordinates plot"""

        if title is None:
            title = f'Parallel Coordinates for {", ".join(columns)}'

        fig = px.parallel_coordinates(
            self.df, dimensions=columns,
            color=color_col,
            title=title
        )

        return fig

    def create_radar_chart(self, categories_col: str, values_col: str,
                          title: Optional[str] = None) -> go.Figure:
        """Create radar chart"""

        if title is None:
            title = f'Radar Chart: {values_col} by {categories_col}'

        # Prepare data for radar chart
        radar_data = self.df.groupby(categories_col)[values_col].mean().reset_index()

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=radar_data[values_col],
            theta=radar_data[categories_col],
            fill='toself',
            name=values_col
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=title,
            height=400
        )

        return fig

    def create_custom_dashboard(self, chart_configs: List[Dict[str, Any]]) -> Dict[str, go.Figure]:
        """Create custom dashboard with specified chart configurations"""

        charts = {}

        for config in chart_configs:
            chart_type = config.get('type')
            chart_id = config.get('id', f'chart_{len(charts)}')

            if chart_type == 'histogram':
                charts[chart_id] = self.create_histogram(
                    config['column'],
                    config.get('bins', 30),
                    config.get('title')
                )
            elif chart_type == 'bar':
                charts[chart_id] = self.create_bar_chart(
                    config['column'],
                    config.get('top_n', 10),
                    config.get('title')
                )
            elif chart_type == 'scatter':
                charts[chart_id] = self.create_scatter_plot(
                    config['x'], config['y'],
                    config.get('color'),
                    config.get('size'),
                    config.get('title')
                )
            elif chart_type == 'correlation':
                numeric_cols = config.get('columns', self.get_column_types()['numeric'])
                charts[chart_id] = self.create_correlation_heatmap(
                    numeric_cols,
                    config.get('title', 'Correlation Heatmap')
                )

        return charts

    def get_chart_summary(self) -> Dict[str, Any]:
        """Get summary of available chart types and recommendations"""

        column_types = self.get_column_types()

        summary = {
            'available_charts': [
                'histogram', 'bar_chart', 'pie_chart', 'correlation_heatmap',
                'scatter_plot', 'interactive_scatter', 'box_plot', 'violin_plot', 'line_chart',
                'area_chart', 'treemap', 'sunburst', 'parallel_coordinates', 'radar_chart'
            ],
            'recommendations': {},
            'column_suggestions': {}
        }

        # Recommendations based on data types
        if column_types['numeric']:
            summary['recommendations']['numeric'] = [
                'histogram', 'correlation_heatmap', 'scatter_plot', 'interactive_scatter', 'box_plot'
            ]

        if column_types['categorical']:
            summary['recommendations']['categorical'] = [
                'bar_chart', 'pie_chart', 'treemap'
            ]

        if column_types['datetime'] and column_types['numeric']:
            summary['recommendations']['time_series'] = [
                'line_chart', 'area_chart'
            ]

        # Column suggestions
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary['column_suggestions'][col] = ['histogram', 'scatter_plot', 'interactive_scatter', 'box_plot']
            elif self.df[col].dtype == 'object':
                summary['column_suggestions'][col] = ['bar_chart', 'pie_chart']

        return summary
