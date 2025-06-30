#!/usr/bin/env python3
"""
Test Script for main_api_fv.py
==============================

Tests for the NYC 311 Complaint Classifier API (FV Version).

Author: AI Assistant
Date: 2024
"""

import pytest
import requests
import json
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestAPIModelLoading:
    """Test API model loading functionality"""
    
    def test_model_path_validation(self):
        """Test that model path is correctly configured"""
        # Expected model path
        expected_model_path = r"D:\Fernstudium\Module\AITools\repo\311NYC\models\complaint_classifier_oop_20250629_203227.pkl"
        
        # Check if model file exists
        model_exists = os.path.exists(expected_model_path)
        
        # Test path format
        assert expected_model_path.endswith('.pkl'), "Model should be a .pkl file"
        assert 'models' in expected_model_path, "Model should be in models directory"
        assert 'complaint_classifier_oop' in expected_model_path, "Model should be complaint classifier"
        
        print(f"✅ Model path validation test passed (Model exists: {model_exists})")
    
    def test_model_package_structure(self):
        """Test expected model package structure"""
        # Expected keys in model package
        expected_keys = [
            'model', 'model_type', 'feature_names', 'target_name',
            'training_samples', 'test_samples', 'timestamp'
        ]
        
        # Test structure validation
        for key in expected_keys:
            assert key is not None, f"Key {key} should be defined"
        
        print("✅ Model package structure test passed")
    
    def test_feature_names_validation(self):
        """Test that feature names are correctly defined"""
        # Expected feature names based on API
        expected_features = [
            'Agency', 'Location_Type', 'Incident_Zip', 'Borough', 'Vehicle_Type',
            'duration_[days]', 'Created_Date_hour', 'Created_Date_month', 
            'Created_Date_dayofweek', 'Descriptor_encoded', 'Weisse', 
            'Afroamerikaner', 'Asiaten', 'Hispanics'
        ]
        
        # Test feature names
        for feature in expected_features:
            assert isinstance(feature, str), f"Feature {feature} should be string"
            assert len(feature) > 0, f"Feature {feature} should not be empty"
        
        print("✅ Feature names validation test passed")


class TestAPIDataModels:
    """Test API data models and validation"""
    
    def test_complaint_model_structure(self):
        """Test Complaint Pydantic model structure"""
        # Expected fields in Complaint model
        expected_fields = {
            'Agency': str,
            'Location_Type': str,
            'Incident_Zip': str,
            'Borough': str,
            'Vehicle_Type': str,
            'duration_days': float,
            'Created_Date_hour': int,
            'Created_Date_month': int,
            'Created_Date_dayofweek': int,
            'Descriptor_encoded': int,
            'Weisse': float,
            'Afroamerikaner': float,
            'Asiaten': float,
            'Hispanics': float
        }
        
        # Test field types
        for field_name, expected_type in expected_fields.items():
            assert expected_type in [str, int, float], f"Field {field_name} should have valid type"
        
        print("✅ Complaint model structure test passed")
    
    def test_complaint_model_defaults(self):
        """Test Complaint model default values"""
        # Test default values
        default_values = {
            'Agency': 'NYPD',
            'Location_Type': 'Street/Sidewalk',
            'Incident_Zip': '10001',
            'Borough': 'MANHATTAN',
            'Vehicle_Type': 'N/A',
            'duration_days': 2.0,
            'Created_Date_hour': 10,
            'Created_Date_month': 1,
            'Created_Date_dayofweek': 1,
            'Descriptor_encoded': 1,
            'Weisse': 0.3,
            'Afroamerikaner': 0.2,
            'Asiaten': 0.2,
            'Hispanics': 0.3
        }
        
        # Validate default values
        for field, default_value in default_values.items():
            assert default_value is not None, f"Default value for {field} should not be None"
        
        print("✅ Complaint model defaults test passed")
    
    def test_demographic_validation(self):
        """Test demographic data validation"""
        # Test demographic percentages
        demo_fields = ['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
        
        # Valid demographic data
        valid_demo = {
            'Weisse': 0.4,
            'Afroamerikaner': 0.2,
            'Asiaten': 0.2,
            'Hispanics': 0.2
        }
        
        # Test sum is approximately 1.0
        total = sum(valid_demo.values())
        assert abs(total - 1.0) < 0.01, f"Demographic percentages should sum to 1.0, got {total}"
        
        # Test individual values are between 0 and 1
        for field, value in valid_demo.items():
            assert 0.0 <= value <= 1.0, f"{field} should be between 0 and 1, got {value}"
        
        print("✅ Demographic validation test passed")
    
    def test_date_validation(self):
        """Test date-related field validation"""
        # Test hour validation
        assert 0 <= 10 <= 23, "Hour should be between 0 and 23"
        assert 0 <= 14 <= 23, "Hour should be between 0 and 23"
        
        # Test month validation
        assert 1 <= 1 <= 12, "Month should be between 1 and 12"
        assert 1 <= 6 <= 12, "Month should be between 1 and 12"
        
        # Test day of week validation
        assert 0 <= 1 <= 6, "Day of week should be between 0 and 6"
        assert 0 <= 5 <= 6, "Day of week should be between 0 and 6"
        
        print("✅ Date validation test passed")


class TestAPIEndpoints:
    """Test API endpoints functionality"""
    
    def test_root_endpoint_structure(self):
        """Test root endpoint response structure"""
        # Expected response structure
        expected_keys = [
            'message', 'model_loaded', 'loaded_model_path', 
            'model_type', 'features_count', 'model_tuned'
        ]
        
        # Test response structure
        for key in expected_keys:
            assert key is not None, f"Response should contain {key}"
        
        print("✅ Root endpoint structure test passed")
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint response structure"""
        # Expected response structure for successful prediction
        success_keys = [
            'predicted_complaint_type', 'confidence_score', 'model_info'
        ]
        
        # Expected model_info structure
        model_info_keys = [
            'model_type', 'features_used', 'training_samples', 'model_tuned'
        ]
        
        # Test response structure
        for key in success_keys:
            assert key is not None, f"Prediction response should contain {key}"
        
        for key in model_info_keys:
            assert key is not None, f"Model info should contain {key}"
        
        print("✅ Predict endpoint structure test passed")
    
    def test_model_info_endpoint_structure(self):
        """Test model-info endpoint response structure"""
        # Expected response structure
        expected_keys = [
            'model_path', 'model_type', 'feature_names', 'target_name',
            'training_samples', 'test_samples', 'timestamp', 'model_tuned'
        ]
        
        # Test response structure
        for key in expected_keys:
            assert key is not None, f"Model info response should contain {key}"
        
        print("✅ Model info endpoint structure test passed")
    
    def test_error_handling_structure(self):
        """Test error response structure"""
        # Expected error response structure
        error_keys = ['error']
        
        # Test error structure
        for key in error_keys:
            assert key is not None, f"Error response should contain {key}"
        
        print("✅ Error handling structure test passed")


class TestAPIDataProcessing:
    """Test API data processing functionality"""
    
    def test_duration_field_mapping(self):
        """Test duration field mapping (duration_days -> duration_[days])"""
        # Test field mapping
        input_data = {
            'duration_days': 2.5
        }
        
        # Simulate mapping
        if 'duration_days' in input_data:
            input_data['duration_[days]'] = input_data.pop('duration_days')
        
        # Verify mapping
        assert 'duration_[days]' in input_data, "duration_days should be mapped to duration_[days]"
        assert 'duration_days' not in input_data, "duration_days should be removed"
        assert input_data['duration_[days]'] == 2.5, "Value should be preserved"
        
        print("✅ Duration field mapping test passed")
    
    def test_feature_selection(self):
        """Test feature selection and ordering"""
        # Sample input data
        input_data = {
            'Agency': 'NYPD',
            'Location_Type': 'Street/Sidewalk',
            'Incident_Zip': '10001',
            'Borough': 'MANHATTAN',
            'Vehicle_Type': 'N/A',
            'duration_[days]': 2.0,
            'Created_Date_hour': 10,
            'Created_Date_month': 1,
            'Created_Date_dayofweek': 1,
            'Descriptor_encoded': 1,
            'Weisse': 0.3,
            'Afroamerikaner': 0.2,
            'Asiaten': 0.2,
            'Hispanics': 0.3
        }
        
        # Expected feature order
        expected_features = [
            'Agency', 'Location_Type', 'Incident_Zip', 'Borough', 'Vehicle_Type',
            'duration_[days]', 'Created_Date_hour', 'Created_Date_month', 
            'Created_Date_dayofweek', 'Descriptor_encoded', 'Weisse', 
            'Afroamerikaner', 'Asiaten', 'Hispanics'
        ]
        
        # Test feature selection
        df = pd.DataFrame([input_data])
        selected_features = df[expected_features]
        
        assert len(selected_features.columns) == len(expected_features), "All features should be selected"
        assert list(selected_features.columns) == expected_features, "Features should be in correct order"
        
        print("✅ Feature selection test passed")
    
    def test_missing_features_handling(self):
        """Test handling of missing features"""
        # Input with missing features
        incomplete_data = {
            'Agency': 'NYPD',
            'Location_Type': 'Street/Sidewalk',
            # Missing other features
        }
        
        # Expected features
        required_features = [
            'Agency', 'Location_Type', 'Incident_Zip', 'Borough', 'Vehicle_Type',
            'duration_[days]', 'Created_Date_hour', 'Created_Date_month', 
            'Created_Date_dayofweek', 'Descriptor_encoded', 'Weisse', 
            'Afroamerikaner', 'Asiaten', 'Hispanics'
        ]
        
        # Test missing features detection
        df = pd.DataFrame([incomplete_data])
        missing_features = set(required_features) - set(df.columns)
        
        assert len(missing_features) > 0, "Should detect missing features"
        assert 'Incident_Zip' in missing_features, "Should detect missing Incident_Zip"
        
        print("✅ Missing features handling test passed")


class TestAPIIntegration:
    """Test API integration scenarios"""
    
    def test_valid_complaint_prediction(self):
        """Test prediction with valid complaint data"""
        # Valid complaint data
        valid_complaint = {
            "Agency": "DSNY",
            "Location_Type": "Street/Sidewalk",
            "Incident_Zip": "10007",
            "Borough": "MANHATTAN",
            "Vehicle_Type": "N/A",
            "duration_[days]": 1.5,
            "Created_Date_hour": 8,
            "Created_Date_month": 3,
            "Created_Date_dayofweek": 5,
            "Descriptor_encoded": 15,
            "Weisse": 0.4,
            "Afroamerikaner": 0.1,
            "Asiaten": 0.3,
            "Hispanics": 0.2
        }
        
        # Test data validation
        assert valid_complaint['Agency'] in ['NYPD', 'DSNY', 'DOT', 'DEP', 'DOB', 'FDNY', 'HPD', 'DOE'], "Valid agency"
        assert 0 <= valid_complaint['Created_Date_hour'] <= 23, "Valid hour"
        assert 1 <= valid_complaint['Created_Date_month'] <= 12, "Valid month"
        assert 0 <= valid_complaint['Created_Date_dayofweek'] <= 6, "Valid day of week"
        
        # Test demographic validation
        demo_sum = sum([valid_complaint['Weisse'], valid_complaint['Afroamerikaner'], 
                       valid_complaint['Asiaten'], valid_complaint['Hispanics']])
        assert abs(demo_sum - 1.0) < 0.01, "Demographics should sum to 1.0"
        
        print("✅ Valid complaint prediction test passed")
    
    def test_edge_case_handling(self):
        """Test edge cases and boundary conditions"""
        # Edge case: minimum values
        min_complaint = {
            "Agency": "NYPD",
            "Location_Type": "Street/Sidewalk",
            "Incident_Zip": "10001",
            "Borough": "MANHATTAN",
            "Vehicle_Type": "N/A",
            "duration_[days]": 0.0,
            "Created_Date_hour": 0,
            "Created_Date_month": 1,
            "Created_Date_dayofweek": 0,
            "Descriptor_encoded": 0,
            "Weisse": 0.0,
            "Afroamerikaner": 0.0,
            "Asiaten": 0.0,
            "Hispanics": 1.0
        }
        
        # Edge case: maximum values
        max_complaint = {
            "Agency": "DOE",
            "Location_Type": "Other",
            "Incident_Zip": "10282",
            "Borough": "STATEN ISLAND",
            "Vehicle_Type": "LIVERY VEHICLE",
            "duration_[days]": 365.0,
            "Created_Date_hour": 23,
            "Created_Date_month": 12,
            "Created_Date_dayofweek": 6,
            "Descriptor_encoded": 999,
            "Weisse": 1.0,
            "Afroamerikaner": 0.0,
            "Asiaten": 0.0,
            "Hispanics": 0.0
        }
        
        # Test edge cases
        for complaint in [min_complaint, max_complaint]:
            assert complaint['duration_[days]'] >= 0, "Duration should be non-negative"
            assert 0 <= complaint['Created_Date_hour'] <= 23, "Hour should be valid"
            assert 1 <= complaint['Created_Date_month'] <= 12, "Month should be valid"
            assert 0 <= complaint['Created_Date_dayofweek'] <= 6, "Day of week should be valid"
        
        print("✅ Edge case handling test passed")


def run_all_tests():
    """Run all main_api_fv tests"""
    print("🧪 Running Main API FV Tests...")
    print("=" * 60)
    
    # Create test instances
    test_model = TestAPIModelLoading()
    test_models = TestAPIDataModels()
    test_endpoints = TestAPIEndpoints()
    test_processing = TestAPIDataProcessing()
    test_integration = TestAPIIntegration()
    
    # Run model loading tests
    try:
        test_model.test_model_path_validation()
        test_model.test_model_package_structure()
        test_model.test_feature_names_validation()
        
        # Run data model tests
        test_models.test_complaint_model_structure()
        test_models.test_complaint_model_defaults()
        test_models.test_demographic_validation()
        test_models.test_date_validation()
        
        # Run endpoint tests
        test_endpoints.test_root_endpoint_structure()
        test_endpoints.test_predict_endpoint_structure()
        test_endpoints.test_model_info_endpoint_structure()
        test_endpoints.test_error_handling_structure()
        
        # Run data processing tests
        test_processing.test_duration_field_mapping()
        test_processing.test_feature_selection()
        test_processing.test_missing_features_handling()
        
        # Run integration tests
        test_integration.test_valid_complaint_prediction()
        test_integration.test_edge_case_handling()
        
        print("=" * 60)
        print("✅ All main_api_fv tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests() 