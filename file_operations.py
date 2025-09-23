"""
File Operations Module for Data Analysis Web App
Phase 6: File Upload, Conversion, and Management
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class FileOperations:
    """Handle file operations including upload, conversion, and management"""

    def __init__(self, upload_dir: str = "uploads"):
        """Initialize with upload directory"""
        self.upload_dir = upload_dir
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure upload directory exists"""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    def load_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a file and return DataFrame with metadata"""

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Get file metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_extension': file_extension,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }

            return df, metadata

        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {str(e)}")

    def save_file(self, df: pd.DataFrame, file_path: str,
                  index: bool = False) -> Dict[str, Any]:
        """Save DataFrame to file with specified format"""

        file_extension = os.path.splitext(file_path)[1].lower()
        directory = os.path.dirname(file_path)

        # Ensure directory exists
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        try:
            if file_extension == '.csv':
                df.to_csv(file_path, index=index)
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=index)
            elif file_extension == '.json':
                df.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Get saved file metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'shape': df.shape,
                'saved_successfully': True
            }

            return metadata

        except Exception as e:
            raise Exception(f"Error saving file {file_path}: {str(e)}")

    def convert_format(self, df: pd.DataFrame, from_format: str,
                      to_format: str, output_path: str) -> Dict[str, Any]:
        """Convert DataFrame from one format to another"""

        supported_conversions = {
            'csv': ['excel', 'json'],
            'excel': ['csv', 'json'],
            'json': ['csv', 'excel']
        }

        if to_format not in supported_conversions.get(from_format, []):
            raise ValueError(f"Conversion from {from_format} to {to_format} not supported")

        # Save in new format
        return self.save_file(df, output_path)

    def merge_files(self, file_paths: List[str],
                   merge_type: str = 'concat',
                   merge_column: Optional[str] = None,
                   how: str = 'outer') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Merge multiple files into a single DataFrame"""

        if not file_paths:
            raise ValueError("No files provided for merging")

        # Load all files
        dfs = []
        metadata_list = []

        for file_path in file_paths:
            df, metadata = self.load_file(file_path)
            dfs.append(df)
            metadata_list.append(metadata)

        if merge_type == 'concat':
            # Concatenate along rows
            merged_df = pd.concat(dfs, ignore_index=True)

        elif merge_type == 'merge':
            # Merge on specified column
            if merge_column is None:
                raise ValueError("merge_column must be specified for merge operation")

            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=merge_column, how=how)

        else:
            raise ValueError(f"Unsupported merge type: {merge_type}")

        # Create merge metadata
        merge_metadata = {
            'merge_type': merge_type,
            'input_files': [meta['file_name'] for meta in metadata_list],
            'input_shapes': [meta['shape'] for meta in metadata_list],
            'output_shape': merged_df.shape,
            'merge_column': merge_column,
            'how': how
        }

        return merged_df, merge_metadata

    def get_file_preview(self, file_path: str, rows: int = 5) -> Dict[str, Any]:
        """Get preview of file content"""

        df, metadata = self.load_file(file_path)

        preview = {
            'metadata': metadata,
            'head': df.head(rows).to_dict('records'),
            'tail': df.tail(rows).to_dict('records'),
            'sample': df.sample(min(rows, len(df))).to_dict('records') if len(df) > 0 else []
        }

        return preview

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file structure and content"""

        try:
            df, metadata = self.load_file(file_path)

            validation = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'metadata': metadata
            }

            # Check for missing values
            missing_info = df.isnull().sum()
            if missing_info.sum() > 0:
                validation['warnings'].append(f"File contains {missing_info.sum()} missing values")

            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                validation['warnings'].append(f"File contains {duplicates} duplicate rows")

            # Check for mixed data types in columns
            for col in df.columns:
                unique_types = df[col].dropna().apply(type).unique()
                if len(unique_types) > 1:
                    validation['warnings'].append(f"Column '{col}' contains mixed data types")

            # Check file size
            if metadata['file_size'] > 100 * 1024 * 1024:  # 100MB
                validation['warnings'].append("File is very large (>100MB) - processing may be slow")

            return validation

        except Exception as e:
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'metadata': None
            }

    def batch_process_files(self, file_paths: List[str],
                          operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple files with specified operations"""

        results = {
            'processed_files': [],
            'failed_files': [],
            'total_processed': 0,
            'total_failed': 0
        }

        for file_path in file_paths:
            try:
                # Load file
                df, metadata = self.load_file(file_path)

                # Apply operations
                for operation in operations:
                    op_type = operation.get('type')
                    params = operation.get('params', {})

                    if op_type == 'clean_missing':
                        df = self._apply_cleaning_operation(df, 'missing', params)
                    elif op_type == 'remove_duplicates':
                        df = self._apply_cleaning_operation(df, 'duplicates', params)
                    elif op_type == 'convert_types':
                        df = self._apply_cleaning_operation(df, 'types', params)

                # Save processed file
                output_path = self._get_output_path(file_path, 'processed')
                save_metadata = self.save_file(df, output_path)

                results['processed_files'].append({
                    'input_file': file_path,
                    'output_file': output_path,
                    'metadata': save_metadata
                })
                results['total_processed'] += 1

            except Exception as e:
                results['failed_files'].append({
                    'file': file_path,
                    'error': str(e)
                })
                results['total_failed'] += 1

        return results

    def _apply_cleaning_operation(self, df: pd.DataFrame,
                                operation: str, params: Dict) -> pd.DataFrame:
        """Apply specific cleaning operation"""

        if operation == 'missing':
            strategy = params.get('strategy', 'drop')
            columns = params.get('columns', None)

            if strategy == 'drop':
                return df.dropna(subset=columns) if columns else df.dropna()
            elif strategy == 'fill_mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                return df.fillna(df[numeric_cols].mean())
            elif strategy == 'fill_median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                return df.fillna(df[numeric_cols].median())

        elif operation == 'duplicates':
            subset = params.get('subset', None)
            keep = params.get('keep', 'first')
            return df.drop_duplicates(subset=subset, keep=keep)

        elif operation == 'types':
            type_conversions = params.get('conversions', {})
            for col, target_type in type_conversions.items():
                if col in df.columns:
                    try:
                        if target_type == 'numeric':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif target_type == 'string':
                            df[col] = df[col].astype('string')
                        elif target_type == 'category':
                            df[col] = df[col].astype('category')
                    except:
                        pass  # Skip if conversion fails

        return df

    def _get_output_path(self, input_path: str, suffix: str) -> str:
        """Generate output path with suffix"""

        directory, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)

        output_filename = f"{name}_{suffix}{ext}"
        output_path = os.path.join(directory, output_filename)

        return output_path

    def cleanup_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Clean up specified files"""

        results = {
            'deleted_files': [],
            'failed_deletions': [],
            'total_deleted': 0,
            'total_failed': 0
        }

        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    results['deleted_files'].append(file_path)
                    results['total_deleted'] += 1
            except Exception as e:
                results['failed_deletions'].append({
                    'file': file_path,
                    'error': str(e)
                })
                results['total_failed'] += 1

        return results
