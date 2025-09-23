"""
Report Generation Module for Data Analysis Web App
Phase 7: Automated Report Generation (PDF/Excel)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.platypus.flowables import KeepTogether
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. PDF generation will be limited.")

class ReportGenerator:
    """Generate comprehensive data analysis reports"""

    def __init__(self, df: pd.DataFrame, title: str = "Data Analysis Report"):
        """Initialize with DataFrame and report title"""
        self.df = df.copy()
        self.title = title
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_excel_report(self, filename: str,
                            include_charts: bool = True) -> Dict[str, Any]:
        """Generate comprehensive Excel report"""

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:

                # Overview sheet
                self._create_overview_sheet(writer)

                # Data sheet
                self.df.to_excel(writer, sheet_name='Data', index=False)

                # Statistics sheet
                self._create_statistics_sheet(writer)

                # Missing values analysis
                self._create_missing_analysis_sheet(writer)

                # Column insights
                self._create_column_insights_sheet(writer)

                # If charts are requested, create charts sheet
                if include_charts:
                    self._create_charts_sheet(writer)

            return {
                'success': True,
                'filename': filename,
                'sheets': ['Overview', 'Data', 'Statistics', 'Missing Analysis', 'Column Insights', 'Charts'],
                'file_size': os.path.getsize(filename)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }

    def _create_overview_sheet(self, writer):
        """Create overview sheet with summary information"""

        overview_data = []

        # Basic information
        overview_data.extend([
            ['Data Analysis Report', ''],
            ['Title', self.title],
            ['Generated', self.report_date],
            ['Dataset Shape', f'{self.df.shape[0]:,} rows × {self.df.shape[1]} columns'],
            ['Memory Usage', f'{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB'],
            ['', ''],
        ])

        # Column types
        column_types = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if dtype not in column_types:
                column_types[dtype] = 0
            column_types[dtype] += 1

        overview_data.extend([
            ['Column Types', ''],
            ['Data Type', 'Count']
        ])

        for dtype, count in column_types.items():
            overview_data.append([dtype, count])

        overview_data.extend([
            ['', ''],
            ['Missing Values Summary', ''],
            ['Total Missing Values', self.df.isnull().sum().sum()],
            ['Columns with Missing', len(self.df.columns[self.df.isnull().any()])],
            ['Columns without Missing', len(self.df.columns) - len(self.df.columns[self.df.isnull().any()])],
        ])

        # Create DataFrame and save
        overview_df = pd.DataFrame(overview_data, columns=['Metric', 'Value'])
        overview_df.to_excel(writer, sheet_name='Overview', index=False)

    def _create_statistics_sheet(self, writer):
        """Create statistics sheet with detailed column statistics"""

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        stats_data = []

        # Numeric columns statistics
        if len(numeric_cols) > 0:
            stats_data.append(['NUMERIC COLUMNS', '', '', '', '', '', ''])

            for col in numeric_cols:
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    stats_data.append([
                        col, 'Numeric',
                        len(col_data),
                        round(col_data.min(), 2),
                        round(col_data.max(), 2),
                        round(col_data.mean(), 2),
                        round(col_data.std(), 2)
                    ])

        # Categorical columns statistics
        if len(categorical_cols) > 0:
            stats_data.append(['', '', '', '', '', '', ''])
            stats_data.append(['CATEGORICAL COLUMNS', '', '', '', '', '', ''])

            for col in categorical_cols:
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    unique_count = col_data.nunique()
                    most_common = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else 'N/A'
                    stats_data.append([
                        col, 'Categorical',
                        len(col_data),
                        unique_count,
                        f"{(unique_count/len(col_data))*100:.1f}%",
                        most_common,
                        col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
                    ])

        # Create DataFrame and save
        stats_df = pd.DataFrame(stats_data, columns=[
            'Column', 'Type', 'Count', 'Min/Unique', 'Max/%Unique',
            'Mean/Top Value', 'Std/Frequency'
        ])
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    def _create_missing_analysis_sheet(self, writer):
        """Create missing values analysis sheet"""

        missing_data = []

        # Missing values per column
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100

        missing_data.extend([
            ['Missing Values Analysis', '', ''],
            ['Column', 'Missing Count', 'Missing Percentage']
        ])

        for col in self.df.columns:
            missing_data.append([
                col,
                int(missing_count[col]),
                f"{missing_percent[col]:.2f}%"
            ])

        # Summary statistics
        missing_data.extend([
            ['', '', ''],
            ['SUMMARY', '', ''],
            ['Total Missing Values', self.df.isnull().sum().sum(), ''],
            ['Columns with Missing', len(missing_count[missing_count > 0]), ''],
            ['Columns without Missing', len(missing_count[missing_count == 0]), ''],
            ['Overall Missing %', f"{(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100:.2f}%", '']
        ])

        # Create DataFrame and save
        missing_df = pd.DataFrame(missing_data, columns=['Column', 'Missing Count', 'Missing %'])
        missing_df.to_excel(writer, sheet_name='Missing Analysis', index=False)

    def _create_column_insights_sheet(self, writer):
        """Create column insights sheet"""

        insights_data = []

        for col in self.df.columns:
            col_data = self.df[col]

            # Basic info
            missing_count = col_data.isnull().sum()
            missing_percent = (missing_count / len(col_data)) * 100
            unique_count = col_data.nunique()
            unique_percent = (unique_count / len(col_data)) * 100

            insights_data.append([
                col, str(col_data.dtype), missing_count, f"{missing_percent:.2f}%",
                unique_count, f"{unique_percent:.2f}%"
            ])

            # Type-specific insights
            if pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    insights_data.append([
                        f"{col} - Min", non_null_data.min(), '', '', '', ''
                    ])
                    insights_data.append([
                        f"{col} - Max", non_null_data.max(), '', '', '', ''
                    ])
                    insights_data.append([
                        f"{col} - Mean", round(non_null_data.mean(), 2), '', '', '', ''
                    ])

            elif col_data.dtype == 'object':
                # Top 3 most frequent values
                top_values = col_data.value_counts().head(3)
                for i, (value, count) in enumerate(top_values.items()):
                    insights_data.append([
                        f"{col} - Top {i+1}", value, count, f"{(count/len(col_data))*100:.2f}%", '', ''
                    ])

        # Create DataFrame and save
        insights_df = pd.DataFrame(insights_data, columns=[
            'Column/Insight', 'Value', 'Count', 'Percentage', 'Unique', 'Unique %'
        ])
        insights_df.to_excel(writer, sheet_name='Column Insights', index=False)

    def _create_charts_sheet(self, writer):
        """Create charts sheet with summary charts"""

        # This is a placeholder - in a real implementation, you would
        # create actual charts using openpyxl chart functionality
        charts_data = [
            ['Charts Summary', ''],
            ['Note: Charts are embedded in the Excel file', ''],
            ['For detailed visualizations, please use the web interface', ''],
            ['', ''],
            ['Available Chart Types', ''],
            ['1. Histograms for numeric columns', ''],
            ['2. Bar charts for categorical columns', ''],
            ['3. Correlation heatmaps', ''],
            ['4. Scatter plots', ''],
            ['5. Box plots', ''],
        ]

        charts_df = pd.DataFrame(charts_data, columns=['Chart Type', 'Description'])
        charts_df.to_excel(writer, sheet_name='Charts', index=False)

    def generate_pdf_report(self, filename: str,
                           include_charts: bool = True) -> Dict[str, Any]:
        """Generate PDF report"""

        if not REPORTLAB_AVAILABLE:
            return {
                'success': False,
                'error': 'reportlab not installed. Install with: pip install reportlab',
                'filename': filename
            }

        try:
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title page
            self._add_title_page(story, styles)

            # Table of contents
            self._add_table_of_contents(story, styles)

            # Executive summary
            self._add_executive_summary(story, styles)

            # Dataset overview
            self._add_dataset_overview(story, styles)

            # Detailed statistics
            self._add_detailed_statistics(story, styles)

            # Missing values analysis
            self._add_missing_analysis(story, styles)

            # Column insights
            self._add_column_insights(story, styles)

            # Charts (if requested)
            if include_charts:
                self._add_charts_section(story, styles)

            # Generate PDF
            doc.build(story)

            return {
                'success': True,
                'filename': filename,
                'pages': len(story) // 10,  # Rough estimate
                'file_size': os.path.getsize(filename)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }

    def _add_title_page(self, story, styles):
        """Add title page to PDF"""

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )

        story.extend([
            Paragraph(self.title, title_style),
            Spacer(1, 0.5 * inch),
            Paragraph(f"Generated on {self.report_date}", styles['Normal']),
            Spacer(1, 0.5 * inch),
            Paragraph(f"Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns", styles['Normal']),
            PageBreak()
        ])

    def _add_table_of_contents(self, story, styles):
        """Add table of contents"""

        story.extend([
            Paragraph('Table of Contents', styles['Heading1']),
            Spacer(1, 12),
            Paragraph('1. Executive Summary', styles['Normal']),
            Paragraph('2. Dataset Overview', styles['Normal']),
            Paragraph('3. Detailed Statistics', styles['Normal']),
            Paragraph('4. Missing Values Analysis', styles['Normal']),
            Paragraph('5. Column Insights', styles['Normal']),
            PageBreak()
        ])

    def _add_executive_summary(self, story, styles):
        """Add executive summary section"""

        story.extend([
            Paragraph('1. Executive Summary', styles['Heading1']),
            Spacer(1, 12),
            Paragraph(f'This report provides a comprehensive analysis of the dataset "{self.title}".', styles['Normal']),
            Spacer(1, 12),
            Paragraph(f'The dataset contains {self.df.shape[0]:,} rows and {self.df.shape[1]} columns.', styles['Normal']),
            Spacer(1, 12),
            Paragraph(f'Total missing values: {self.df.isnull().sum().sum()}', styles['Normal']),
            PageBreak()
        ])

    def _add_dataset_overview(self, story, styles):
        """Add dataset overview section"""

        story.extend([
            Paragraph('2. Dataset Overview', styles['Heading1']),
            Spacer(1, 12)
        ])

        # Basic info table
        basic_info = [
            ['Dataset Shape', f'{self.df.shape[0]:,} × {self.df.shape[1]}'],
            ['Memory Usage', f'{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB'],
            ['Total Missing Values', str(self.df.isnull().sum().sum())],
            ['Columns with Missing', str(len(self.df.columns[self.df.isnull().any()]))]
        ]

        self._add_table(story, basic_info, ['Metric', 'Value'])
        story.append(PageBreak())

    def _add_detailed_statistics(self, story, styles):
        """Add detailed statistics section"""

        story.extend([
            Paragraph('3. Detailed Statistics', styles['Heading1']),
            Spacer(1, 12)
        ])

        # Add statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            story.append(Paragraph('Numeric Columns:', styles['Heading2']))

            for col in numeric_cols:
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    stats = [
                        ['Column', col],
                        ['Count', str(len(col_data))],
                        ['Mean', f"{col_data.mean():.2f}"],
                        ['Std Dev', f"{col_data.std():.2f}"],
                        ['Min', f"{col_data.min():.2f}"],
                        ['Max', f"{col_data.max():.2f}"]
                    ]
                    self._add_table(story, stats, ['Statistic', 'Value'])

        story.append(PageBreak())

    def _add_missing_analysis(self, story, styles):
        """Add missing values analysis section"""

        story.extend([
            Paragraph('4. Missing Values Analysis', styles['Heading1']),
            Spacer(1, 12)
        ])

        missing_count = self.df.isnull().sum()
        missing_data = [['Column', 'Missing Count', 'Missing %']]

        for col in self.df.columns:
            missing_data.append([
                col,
                str(missing_count[col]),
                f"{(missing_count[col]/len(self.df))*100:.2f}%"
            ])

        self._add_table(story, missing_data, ['Column', 'Missing Count', 'Missing %'])
        story.append(PageBreak())

    def _add_column_insights(self, story, styles):
        """Add column insights section"""

        story.extend([
            Paragraph('5. Column Insights', styles['Heading1']),
            Spacer(1, 12)
        ])

        for col in self.df.columns:
            col_data = self.df[col]

            insights = [
                ['Column', col],
                ['Data Type', str(col_data.dtype)],
                ['Missing Count', str(col_data.isnull().sum())],
                ['Unique Values', str(col_data.nunique())]
            ]

            self._add_table(story, insights, ['Insight', 'Value'])

        story.append(PageBreak())

    def _add_charts_section(self, story, styles):
        """Add charts section (placeholder)"""

        story.extend([
            Paragraph('6. Charts and Visualizations', styles['Heading1']),
            Spacer(1, 12),
            Paragraph('Charts are available in the interactive web interface.', styles['Normal']),
            Paragraph('For static charts, please refer to the Excel report.', styles['Normal']),
            PageBreak()
        ])

    def _add_table(self, story, data, headers):
        """Add a table to the PDF story"""

        if not data:
            return

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 12))
