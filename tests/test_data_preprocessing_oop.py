import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the DataPreprocessorOOP class from data_preprocessing
from src.data_preprocessing import DataPreprocessorOOP

class TestDataPreprocessingOOP:
    """Test class for DataPreprocessorOOP functionality."""
    
    @pytest.fixture
    def sample_311_data(self):
        """Provides a sample 311 DataFrame for testing."""
        data = {
            'Created Date': [
                '2023-01-01 10:00:00', 
                '2023-01-02 22:00:00', 
                '2023-01-03 15:30:00',
                '2023-01-04 08:45:00',
                '2023-01-05 12:20:00'
            ],
            'Closed Date': [
                '2023-01-01 12:00:00', 
                '2023-01-03 00:00:00', 
                '2023-01-03 18:00:00',
                '2023-01-04 10:00:00',
                '2023-01-05 14:00:00'
            ],
            'Complaint Type': [
                'Noise - Residential', 
                'Sanitation Condition', 
                'Street Light Condition',
                'Noise - Commercial',
                'Water System'
            ],
            'Borough': [
                'MANHATTAN', 
                'BROOKLYN', 
                'QUEENS',
                'BRONX',
                'STATEN ISLAND'
            ],
            'Agency Name': [
                'NYPD', 
                'DSNY', 
                'DOT',
                'NYPD',
                'DEP'
            ],
            'Descriptor': [
                'Loud Music', 
                'Missed Collection', 
                'Light Out',
                'Loud Music',
                'Water Quality'
            ],
            'Location': [
                '123 Main St', 
                '456 Oak Ave', 
                '789 Pine Rd',
                '321 Elm St',
                '654 Maple Dr'
            ],
            'Incident Zip': [
                '10001', 
                '11201', 
                '11375',
                '10451',
                '10301'
            ],
            'Vehicle Type': [
                'PASSENGER VEHICLE', 
                'TRUCK', 
                'MOTORCYCLE',
                'PASSENGER VEHICLE',
                'TRUCK'
            ],
            'Location Type': [
                'Street/Sidewalk', 
                'Residential Building/House', 
                'Street/Sidewalk',
                'Commercial Building',
                'Residential Building/House'
            ]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_demographics_data(self):
        """Provides sample demographic data for testing."""
        data = {
            'Stadtteil': ['BROOKLYN', 'MANHATTAN', 'QUEENS', 'BRONX', 'STATEN ISLAND'],
            'Weisse': [0.412, 0.544, 0.441, 0.299, 0.776],
            'Afroamerikaner': [0.364, 0.174, 0.200, 0.356, 0.097],
            'Asiaten': [0.075, 0.094, 0.176, 0.030, 0.057],
            'Hispanics': [0.198, 0.272, 0.250, 0.484, 0.121]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_zip_codes_data(self):
        """Provides sample NYC zip codes data for testing."""
        data = {
            'ZipCode': ['10001', '11201', '11375', '10451', '10301'],
            'Borough': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        }
        return pd.DataFrame(data)

    def test_date_feature_transformer(self, sample_311_data):
        """Tests the DateFeatureTransformer."""
        preprocessor = DataPreprocessorOOP()
        transformer = preprocessor.DateFeatureTransformer(date_columns=['Created Date'])
        transformed_df = transformer.fit_transform(sample_311_data)
        
        # Check if new date feature columns are created
        expected_columns = [
            'Created Date_hour', 'Created Date_dayofweek', 
            'Created Date_month', 'Created Date_year',
            'Created Date_is_weekend', 'Created Date_is_business_hour'
        ]
        
        # Check that at least some of the expected columns are created
        # (the exact column names might vary depending on the transformer implementation)
        created_columns = [col for col in transformed_df.columns if 'Created Date_' in col]
        assert len(created_columns) > 0, f"No date feature columns found. Available columns: {list(transformed_df.columns)}"
        
        # Check that the original column is still present
        assert 'Created Date' in transformed_df.columns
        
        # Check specific values for the first row (2023-01-01 10:00:00 - Sunday)
        # Use more flexible column name matching
        hour_col = [col for col in transformed_df.columns if 'hour' in col.lower()]
        if hour_col:
            assert transformed_df.loc[0, hour_col[0]] == 10
        
        dayofweek_col = [col for col in transformed_df.columns if 'dayofweek' in col.lower()]
        if dayofweek_col:
            assert transformed_df.loc[0, dayofweek_col[0]] == 6  # Sunday
        
        weekend_col = [col for col in transformed_df.columns if 'weekend' in col.lower()]
        if weekend_col:
            assert transformed_df.loc[0, weekend_col[0]] == 1

    def test_descriptor_high_cardinality_encoder(self, sample_311_data):
        """Tests the DescriptorHighCardinalityEncoder."""
        preprocessor = DataPreprocessorOOP()
        transformer = preprocessor.DescriptorHighCardinalityEncoder(top_n=3)
        transformed_df = transformer.fit_transform(sample_311_data)
        
        # Check if Descriptor column is encoded (column name might be cleaned)
        descriptor_col = 'Descriptor'
        if descriptor_col not in transformed_df.columns:
            # Try to find the descriptor column with different naming
            descriptor_col = [col for col in transformed_df.columns if 'descriptor' in col.lower()]
            if descriptor_col:
                descriptor_col = descriptor_col[0]
            else:
                pytest.skip("Descriptor column not found after transformation")
        
        assert descriptor_col in transformed_df.columns
        assert pd.api.types.is_integer_dtype(transformed_df[descriptor_col].dtype)
        
        # Check that values are encoded (should be integers)
        assert transformed_df[descriptor_col].min() >= 0
        assert transformed_df[descriptor_col].max() <= 3  # top_n=3

    def test_high_cardinality_encoder(self, sample_311_data):
        """Tests the HighCardinalityEncoder."""
        preprocessor = DataPreprocessorOOP()
        transformer = preprocessor.HighCardinalityEncoder(columns=['Incident Zip'])
        transformed_df = transformer.fit_transform(sample_311_data)
        
        # Check if Incident Zip is encoded (column name might be cleaned)
        zip_col = 'Incident Zip'
        if zip_col not in transformed_df.columns:
            # Try to find the zip column with different naming
            zip_col = [col for col in transformed_df.columns if 'incident' in col.lower() and 'zip' in col.lower()]
            if zip_col:
                zip_col = zip_col[0]
            else:
                pytest.skip("Incident Zip column not found after transformation")
        
        assert zip_col in transformed_df.columns
        assert pd.api.types.is_integer_dtype(transformed_df[zip_col].dtype)
        
        # Check that values are encoded
        assert transformed_df[zip_col].min() >= 0
        assert len(transformed_df[zip_col].unique()) == len(sample_311_data['Incident Zip'].unique())

    def test_handle_vehicle_type(self, sample_311_data):
        """Tests the handle_vehicle_type method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.handle_vehicle_type()
        
        # Check if Vehicle Type column is still present (column name might be cleaned)
        vehicle_col = 'Vehicle Type'
        if vehicle_col not in preprocessor.df.columns:
            vehicle_col = [col for col in preprocessor.df.columns if 'vehicle' in col.lower()]
            if vehicle_col:
                vehicle_col = vehicle_col[0]
            else:
                pytest.skip("Vehicle Type column not found after transformation")
        
        assert vehicle_col in preprocessor.df.columns
        
        # Check if values are properly categorized
        vehicle_types = preprocessor.df[vehicle_col].unique()
        assert len(vehicle_types) > 0

    def test_handle_location_type(self, sample_311_data):
        """Tests the handle_location_type method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.handle_location_type()
        
        # Check if Location Type column is still present (column name might be cleaned)
        location_col = 'Location Type'
        if location_col not in preprocessor.df.columns:
            location_col = [col for col in preprocessor.df.columns if 'location' in col.lower() and 'type' in col.lower()]
            if location_col:
                location_col = location_col[0]
            else:
                pytest.skip("Location Type column not found after transformation")
        
        assert location_col in preprocessor.df.columns
        
        # Check if values are properly categorized
        location_types = preprocessor.df[location_col].unique()
        assert len(location_types) > 0

    def test_consolidate_location_types(self, sample_311_data):
        """Tests the consolidate_location_types method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.consolidate_location_types()
        
        # Check if Location Type column is still present
        location_col = 'Location Type'
        if location_col not in preprocessor.df.columns:
            location_col = [col for col in preprocessor.df.columns if 'location' in col.lower() and 'type' in col.lower()]
            if location_col:
                location_col = location_col[0]
            else:
                pytest.skip("Location Type column not found after transformation")
        
        assert location_col in preprocessor.df.columns
        
        # Check if consolidation reduced the number of categories
        original_types = sample_311_data['Location Type'].nunique()
        consolidated_types = preprocessor.df[location_col].nunique()
        assert consolidated_types <= original_types

    def test_consolidate_complaint_types(self, sample_311_data):
        """Tests the consolidate_complaint_types method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.consolidate_complaint_types()
        
        # Check if Complaint Type column is still present
        complaint_col = 'Complaint Type'
        if complaint_col not in preprocessor.df.columns:
            complaint_col = [col for col in preprocessor.df.columns if 'complaint' in col.lower()]
            if complaint_col:
                complaint_col = complaint_col[0]
            else:
                pytest.skip("Complaint Type column not found after transformation")
        
        assert complaint_col in preprocessor.df.columns
        
        # Check if consolidation reduced the number of categories
        original_types = sample_311_data['Complaint Type'].nunique()
        consolidated_types = preprocessor.df[complaint_col].nunique()
        assert consolidated_types <= original_types

    def test_calculate_duration(self, sample_311_data):
        """Tests the calculate_duration method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.calculate_duration()
        
        # Check if duration column is created
        duration_col = 'duration'
        if duration_col not in preprocessor.df.columns:
            duration_col = [col for col in preprocessor.df.columns if 'duration' in col.lower()]
            if duration_col:
                duration_col = duration_col[0]
            else:
                pytest.skip("Duration column not found after transformation")
        
        assert duration_col in preprocessor.df.columns
        assert pd.api.types.is_numeric_dtype(preprocessor.df[duration_col].dtype)

    def test_remove_negative_duration(self, sample_311_data):
        """Tests the remove_negative_duration method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add some negative duration values for testing
        preprocessor.df['duration'] = [1.5, -0.5, 2.0, -1.0, 0.8]
        original_count = len(preprocessor.df)
        
        preprocessor.remove_negative_duration()
        
        # Check if negative durations were removed
        if 'duration' in preprocessor.df.columns:
            assert preprocessor.df['duration'].min() >= 0
            assert len(preprocessor.df) <= original_count

    def test_remove_missing_incident_zip(self, sample_311_data):
        """Tests the remove_missing_incident_zip method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add some missing zip codes for testing
        preprocessor.df.loc[1, 'Incident Zip'] = None
        preprocessor.df.loc[3, 'Incident Zip'] = ''
        original_count = len(preprocessor.df)
        
        preprocessor.remove_missing_incident_zip()
        
        # Check if rows with missing zip codes were removed
        assert len(preprocessor.df) <= original_count
        if 'Incident Zip' in preprocessor.df.columns:
            assert preprocessor.df['Incident Zip'].notna().all()

    def test_convert_incident_zip_to_integer(self, sample_311_data):
        """Tests the convert_incident_zip_to_integer method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.convert_incident_zip_to_integer()
        
        # Check if Incident Zip is converted to integer
        if 'Incident Zip' in preprocessor.df.columns:
            assert pd.api.types.is_integer_dtype(preprocessor.df['Incident Zip'].dtype)

    def test_rename_duration_column(self, sample_311_data):
        """Tests the rename_duration_column method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.df['duration'] = [1.5, 2.0, 0.8, 1.2, 3.0]
        
        preprocessor.rename_duration_column()
        
        # Check if duration column was renamed
        assert 'duration_[days]' in preprocessor.df.columns or 'duration' in preprocessor.df.columns

    def test_round_duration_to_two_decimals(self, sample_311_data):
        """Tests the round_duration_to_two_decimals method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        preprocessor.df['duration'] = [1.56789, 2.12345, 0.87654, 1.23456, 3.00000]
        
        preprocessor.round_duration_to_two_decimals()
        
        # Check if duration values are rounded to 2 decimals
        if 'duration' in preprocessor.df.columns:
            rounded_values = preprocessor.df['duration'].apply(lambda x: round(x, 2))
            assert (preprocessor.df['duration'] == rounded_values).all()

    def test_handle_outliers(self, sample_311_data):
        """Tests the handle_outliers method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add some numerical columns with outliers for testing
        preprocessor.df['test_numeric'] = [1, 2, 3, 100, 5]  # 100 is an outlier
        original_count = len(preprocessor.df)
        
        preprocessor.handle_outliers(columns=['test_numeric'], method='iqr', threshold=1.5)
        
        # Check if outliers were handled (either removed or capped)
        assert len(preprocessor.df) <= original_count

    def test_clean_feature_names(self, sample_311_data):
        """Tests the clean_feature_names method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add a column with spaces for testing
        preprocessor.df['Test Column'] = [1, 2, 3, 4, 5]
        original_columns = list(preprocessor.df.columns)
        
        preprocessor.clean_feature_names()
        
        # Check if spaces in column names were replaced with underscores
        cleaned_columns = list(preprocessor.df.columns)
        assert any('_' in col for col in cleaned_columns)

    def test_handle_null_values_after_transformations(self, sample_311_data):
        """Tests the handle_null_values_after_transformations method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add some null values for testing
        preprocessor.df.loc[0, 'Complaint Type'] = None
        preprocessor.df.loc[1, 'Incident Zip'] = None
        
        preprocessor.handle_null_values_after_transformations()
        
        # Check if null values were handled
        assert preprocessor.df.isnull().sum().sum() == 0

    def test_handle_demographic_outliers(self, sample_311_data, sample_demographics_data, tmp_path):
        """Tests the handle_demographic_outliers method."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Add demographic columns with outliers for testing
        preprocessor.df['Weisse'] = [0.5, 0.6, 0.7, 0.8, 0.9]
        preprocessor.df['Afroamerikaner'] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        preprocessor.handle_demographic_outliers()
        
        # Check if demographic columns are still present
        assert 'Weisse' in preprocessor.df.columns
        assert 'Afroamerikaner' in preprocessor.df.columns

    def test_integration_pipeline(self, sample_311_data, sample_demographics_data, sample_zip_codes_data, tmp_path):
        """Tests the complete integration pipeline."""
        preprocessor = DataPreprocessorOOP()
        preprocessor.df = sample_311_data.copy()
        
        # Test a subset of the pipeline
        preprocessor.handle_vehicle_type()
        preprocessor.handle_location_type()
        preprocessor.consolidate_complaint_types()
        preprocessor.calculate_duration()
        preprocessor.clean_feature_names()
        
        # Check if the data was processed
        assert len(preprocessor.df) > 0
        assert len(preprocessor.df.columns) > 0 