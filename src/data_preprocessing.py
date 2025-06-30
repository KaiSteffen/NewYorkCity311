import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

import scipy
scipy.interp = np.interp

from pathlib import Path
from sklearn.model_selection import train_test_split
import re
from collections import Counter

class DataPreprocessorOOP:
    def __init__(self, base_path=None):
        # Adjust base path to work from src directory
        if base_path is None:
            # Go up one level from src to get to the project root
            self.BASE_PATH = Path(__file__).parent.parent
        else:
            self.BASE_PATH = Path(base_path)
        self.df = None

    class DateFeatureTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, date_columns):
            self.date_columns = date_columns
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = X.copy()
            for col in self.date_columns:
                if col in X.columns:
                    X[f'{col}_hour'] = pd.to_datetime(X[col]).dt.hour
                    X[f'{col}_month'] = pd.to_datetime(X[col]).dt.month
                    X[f'{col}_dayofweek'] = pd.to_datetime(X[col]).dt.dayofweek
            return X

    class DescriptorHighCardinalityEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, top_n=50):
            self.top_n = top_n
            self.top_descriptors = []
            self.label_encoder = None
        def fit(self, X, y=None):
            if 'Descriptor' in X.columns:
                descriptor_counts = X['Descriptor'].value_counts()
                self.top_descriptors = descriptor_counts.head(self.top_n).index.tolist()
                X_temp = X.copy()
                X_temp['Descriptor_processed'] = X_temp['Descriptor'].apply(
                    lambda x: x if x in self.top_descriptors else 'sonstiges'
                )
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(X_temp['Descriptor_processed'])
                print(f"Fitted descriptor encoder with {len(self.top_descriptors)} top descriptors + 'sonstiges'")
                print(f"Top descriptors: {self.top_descriptors}")
            return self
        def transform(self, X):
            X = X.copy()
            if 'Descriptor' in X.columns and self.label_encoder is not None:
                X['Descriptor_processed'] = X['Descriptor'].apply(
                    lambda x: x if x in self.top_descriptors else 'sonstiges'
                )
                X['Descriptor_encoded'] = self.label_encoder.transform(X['Descriptor_processed'])
                print(f"\n📊 Descriptor Encoding Results:")
                print(f"Total unique descriptors: {X['Descriptor'].nunique()}")
                print(f"Top {self.top_n} descriptors kept as individual categories:")
                descriptor_counts = X['Descriptor'].value_counts()
                for i, descriptor in enumerate(self.top_descriptors, 1):
                    count = descriptor_counts[descriptor]
                    print(f"{i:2d}. {descriptor}: {count:,} entries")
                sonstiges_count = (X['Descriptor_processed'] == 'sonstiges').sum()
                print(f"   'sonstiges': {sonstiges_count:,} entries")
                print(f"\nEncoded into {len(self.label_encoder.classes_)} categories")
                X = X.drop(columns=['Descriptor', 'Descriptor_processed'])
                print(f"Transformed 'Descriptor' into encoded feature with {len(self.label_encoder.classes_)} categories")
            return X

    class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns
            self.label_encoders = {}
        def fit(self, X, y=None):
            for col in self.columns:
                if col in X.columns:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(X[col])
            return self
        def transform(self, X):
            X = X.copy()
            for col in self.columns:
                if col in X.columns and col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col])
            return X

    def load_and_check_data(self):
        """Load the filtered data and check for null values"""
        print("="*60)
        print("LOADING AND CHECKING DATA")
        print("="*60)
        
        # Load the filtered dataset
        print("Loading filtered_311_data_top30.csv...")
        try:
            self.df = pd.read_csv(f'{self.BASE_PATH}/data/filtered_311_data_top30.csv')
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Original dataset shape: {self.df.shape}")
        except FileNotFoundError:
            print("❌ File not found: data/filtered_311_data_top30.csv")
            return None
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            return None
        
        # Show all column names
        print("\n📋 All column names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. '{col}'")
        
        # Check for null values but don't remove them yet
        print("\n🔍 Checking for null values...")
        null_counts = self.df.isnull().sum()
        total_null = null_counts.sum()
        
        if total_null == 0:
            print("✅ No null values found in the dataset!")
        else:
            print(f"⚠️  Found {total_null} null values:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"   - {col}: {count} null values")
            print("📝 Note: Null values will be handled after transformations")
        
        return self.df

    def handle_vehicle_type(self):
        """Handle Vehicle Type column: replace missing with 'none', keep existing types for encoding"""
        print("\n" + "="*60)
        print("HANDLING VEHICLE TYPE")
        print("="*60)
        
        if 'Vehicle Type' not in self.df.columns:
            print("⚠️  'Vehicle Type' column not found - skipping")
            return
        
        # Show original distribution with detailed counts
        print("📊 Original Vehicle Type distribution:")
        vehicle_counts = self.df['Vehicle Type'].value_counts(dropna=False)
        total_records = len(self.df)
        
        print(f"Total records: {total_records:,}")
        print("Original values:")
        for value, count in vehicle_counts.items():
            percentage = (count / total_records) * 100
            print(f"  - '{value}': {count:,} records ({percentage:.2f}%)")
        
        # Count NaN values separately
        nan_count = self.df['Vehicle Type'].isna().sum()
        if nan_count > 0:
            nan_percentage = (nan_count / total_records) * 100
            print(f"  - NaN values: {nan_count:,} records ({nan_percentage:.2f}%)")
        
        # Replace only missing values with 'none', keep existing vehicle types
        print("\n🔄 Processing Vehicle Type...")
        print("   - Replacing missing values with 'none'")
        print("   - Keeping existing vehicle types for high-cardinality encoding")
        
        # Handle NaN and empty values
        self.df['Vehicle Type'] = self.df['Vehicle Type'].fillna('none')
        self.df['Vehicle Type'] = self.df['Vehicle Type'].apply(
            lambda x: 'none' if pd.isna(x) or str(x).strip() == '' or str(x).lower() in ['nan', ''] else x
        )
        
        # Show processed distribution with detailed counts
        print("📊 Processed Vehicle Type distribution:")
        processed_counts = self.df['Vehicle Type'].value_counts()
        
        print("After processing:")
        for value, count in processed_counts.items():
            percentage = (count / total_records) * 100
            print(f"  - '{value}': {count:,} records ({percentage:.2f}%)")
        
        print(f"✅ Vehicle Type processing completed - {len(processed_counts)} categories for encoding")

    def handle_location_type(self):
        """Handle Location Type column: replace missing with 'Other' and group rare categories"""
        print("\n" + "="*60)
        print("HANDLING LOCATION TYPE")
        print("="*60)
        
        if 'Location Type' not in self.df.columns:
            print("⚠️  'Location Type' column not found - skipping")
            return
        
        # Show original distribution
        print("📊 Original Location Type distribution:")
        location_counts = self.df['Location Type'].value_counts(dropna=False)
        print(location_counts)
        
        # Replace missing values with 'Other'
        print("\n🔄 Processing Location Type...")
        self.df['Location Type'] = self.df['Location Type'].fillna('Other')
        self.df['Location Type'] = self.df['Location Type'].apply(
            lambda x: 'Other' if pd.isna(x) or str(x).strip() == '' or str(x).lower() == 'nan' else x
        )
        
        # Show processed distribution
        print("📊 Processed Location Type distribution:")
        processed_counts = self.df['Location Type'].value_counts()
        print(processed_counts)
        
        print(f"✅ Location Type processing completed - {len(processed_counts)} categories")

    def consolidate_location_types(self):
        """Consolidate Location Type categories with less than 0.01% contribution into 'Other'"""
        print("\n" + "="*60)
        print("CONSOLIDATING LOCATION TYPE CATEGORIES")
        print("="*60)
        
        if 'Location Type' not in self.df.columns:
            print("⚠️  'Location Type' column not found - skipping")
            return
        
        # Show original location type distribution
        location_counts = self.df['Location Type'].value_counts()
        total_records = len(self.df)
        
        print(f"📊 Original Location Type distribution:")
        print(f"   - Total records: {total_records:,}")
        print(f"   - Unique location types: {len(location_counts)}")
        
        # Calculate percentages
        location_percentages = (location_counts / total_records * 100).round(4)
        
        print(f"\n📋 Original Location Type categories and percentages:")
        for location_type, count in location_counts.items():
            percentage = location_percentages[location_type]
            print(f"   - {location_type}: {count:,} ({percentage:.4f}%)")
        
        # Identify categories with less than 0.01% contribution
        threshold = 0.01
        small_categories = location_percentages[location_percentages < threshold]
        
        if len(small_categories) == 0:
            print(f"\n✅ No Location Type categories with less than {threshold}% found!")
            return
        
        print(f"\n🔍 Found {len(small_categories)} categories with less than {threshold}% contribution:")
        total_small_count = 0
        for location_type, percentage in small_categories.items():
            count = location_counts[location_type]
            total_small_count += count
            print(f"   - {location_type}: {count:,} ({percentage:.4f}%)")
        
        print(f"\n📊 Total records to be consolidated: {total_small_count:,} ({total_small_count/total_records*100:.4f}%)")
        
        # Consolidate small categories into 'Other'
        small_categories_list = small_categories.index.tolist()
        consolidation_mask = self.df['Location Type'].isin(small_categories_list)
        
        self.df.loc[consolidation_mask, 'Location Type'] = 'Other'
        
        # Show final distribution
        final_location_counts = self.df['Location Type'].value_counts()
        final_location_percentages = (final_location_counts / len(self.df) * 100).round(4)
        
        print(f"\n✅ Consolidation completed!")
        print(f"📊 Final Location Type distribution:")
        for location_type, count in final_location_counts.items():
            percentage = final_location_percentages[location_type]
            print(f"   - {location_type}: {count:,} ({percentage:.4f}%)")
        
        # Show summary
        print(f"\n📈 Consolidation summary:")
        print(f"   - Categories consolidated: {len(small_categories)}")
        print(f"   - Records moved to 'Other': {total_small_count:,}")
        print(f"   - Final unique categories: {len(final_location_counts)}")

    def consolidate_complaint_types(self):
        """Consolidate Complaint Type categories to reduce cardinality"""
        print("\n" + "="*60)
        print("CONSOLIDATING COMPLAINT TYPES")
        print("="*60)
        
        if 'Complaint Type' not in self.df.columns:
            print("⚠️  'Complaint Type' column not found - skipping")
            return
        
        # Show original distribution
        print("📊 Original Complaint Type distribution:")
        complaint_counts = self.df['Complaint Type'].value_counts()
        print(f"Total unique complaint types: {len(complaint_counts)}")
        print(complaint_counts.head(10))
        
        # Keep top complaint types and consolidate others
        top_complaints = complaint_counts.head(20).index.tolist()
        
        # Apply consolidation
        self.df['Complaint Type'] = self.df['Complaint Type'].apply(
            lambda x: x if x in top_complaints else 'Other'
        )
        
        # Show consolidated distribution
        print("📊 Consolidated Complaint Type distribution:")
        consolidated_counts = self.df['Complaint Type'].value_counts()
        print(f"Total unique complaint types after consolidation: {len(consolidated_counts)}")
        print(consolidated_counts.head(10))
        
        print(f"✅ Complaint Type consolidation completed - {len(consolidated_counts)} categories")

    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """Handle outliers in numerical columns"""
        print("\n" + "="*60)
        print("HANDLING OUTLIERS")
        print("="*60)
        
        if columns is None:
            # Get numerical columns
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and demographic columns
            exclude_cols = ['Complaint_Type', 'Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
            columns = [col for col in numerical_cols if col not in exclude_cols]
        
        print(f"Processing {len(columns)} numerical columns for outliers...")
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"   - {col}: {outliers} outliers detected and capped")
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                else:
                    print(f"   - {col}: No outliers detected")
        
        print("✅ Outlier handling completed")

    def calculate_duration(self):
        """Calculate duration between Created Date and Closed Date"""
        print("\n" + "="*60)
        print("CALCULATING DURATION")
        print("="*60)
        
        if 'Created Date' not in self.df.columns or 'Closed Date' not in self.df.columns:
            print("⚠️  Date columns not found - skipping duration calculation")
            return
        
        # Convert to datetime
        self.df['Created Date'] = pd.to_datetime(self.df['Created Date'])
        self.df['Closed Date'] = pd.to_datetime(self.df['Closed Date'])
        
        # Calculate duration in days
        self.df['duration'] = (self.df['Closed Date'] - self.df['Created Date']).dt.total_seconds() / (24 * 3600)
        
        # Show duration statistics
        print("📊 Duration statistics:")
        print(f"   - Mean duration: {self.df['duration'].mean():.2f} days")
        print(f"   - Median duration: {self.df['duration'].median():.2f} days")
        print(f"   - Min duration: {self.df['duration'].min():.2f} days")
        print(f"   - Max duration: {self.df['duration'].max():.2f} days")
        
        print("✅ Duration calculation completed")

    def display_borough_values(self):
        """Display unique borough values"""
        print("\n" + "="*60)
        print("BOROUGH VALUES")
        print("="*60)
        
        if 'Borough' in self.df.columns:
            borough_counts = self.df['Borough'].value_counts()
            print("📊 Borough distribution:")
            for borough, count in borough_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"   - {borough}: {count:,} records ({percentage:.1f}%)")
        else:
            print("⚠️  'Borough' column not found")

    def apply_transformations_with_demographics(self):
        """Apply transformations and merge demographics before encoding"""
        print("\n" + "="*60)
        print("APPLYING DATA TRANSFORMATIONS WITH DEMOGRAPHICS")
        print("="*60)
        
        # Step 1: Remove specified columns before transformations
        columns_to_remove = ['location', 'Creation Date', 'Closed Date']
        df_clean = self.df.copy()
        
        for col in columns_to_remove:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
                print(f"   🗑️  Removed column: {col}")
        
        # Step 2: Apply date feature transformation only for Created Date
        print("1. Applying date feature transformation (Created Date only)...")
        date_transformer = self.DateFeatureTransformer(['Created Date'])
        df_transformed = date_transformer.transform(df_clean)
        print(f"   ✅ Created date features. Shape: {df_transformed.shape}")
        
        # Step 3: Apply descriptor encoding
        print("2. Applying descriptor encoding...")
        descriptor_encoder = self.DescriptorHighCardinalityEncoder(top_n=50)
        descriptor_encoder.fit(df_transformed)
        df_transformed = descriptor_encoder.transform(df_transformed)
        
        # Show descriptor encoding information
        if 'Descriptor_encoded' in df_transformed.columns:
            print(f"   ✅ Descriptor encoded. Unique values: {df_transformed['Descriptor_encoded'].nunique()}")
        
        # Step 4: Apply high-cardinality encoding to Descriptor_encoded column
        print("3. Applying high-cardinality encoding to Descriptor_encoded...")
        if 'Descriptor_encoded' in df_transformed.columns:
            descriptor_high_card_encoder = self.HighCardinalityEncoder(['Descriptor_encoded'])
            descriptor_high_card_encoder.fit(df_transformed)
            df_transformed = descriptor_high_card_encoder.transform(df_transformed)
            print(f"   ✅ Descriptor_encoded high-cardinality encoded")
        
        # Step 5: FILL MISSING BOROUGHS WITH ZIP CODES BEFORE DEMOGRAPHIC MERGE
        print("4. Filling missing borough entries with zip codes (before demographic merge)...")
        print(f"🔍 DEBUG: About to call fill_missing_boroughs_with_zip_codes")
        print(f"🔍 DEBUG: Current dataframe shape: {df_transformed.shape}")
        print(f"🔍 DEBUG: Current columns: {list(df_transformed.columns)}")
        df_transformed = self.fill_missing_boroughs_with_zip_codes(df_transformed)
        print(f"🔍 DEBUG: After fill_missing_boroughs_with_zip_codes - shape: {df_transformed.shape}")
        
        # Step 6: MERGE DEMOGRAPHICS AFTER BOROUGH FILLING
        print("5. Merging demographic data (after borough filling)...")
        print(f"🔍 DEBUG: About to call load_and_merge_demographics")
        df_transformed = self.load_and_merge_demographics(df_transformed)
        print(f"🔍 DEBUG: After load_and_merge_demographics - shape: {df_transformed.shape}")
        
        # Step 7: Apply high-cardinality encoding to remaining categorical columns
        print("6. Applying high-cardinality encoding to remaining categorical columns...")
        categorical_columns = [col for col in df_transformed.columns if df_transformed[col].dtype == 'object']
        
        # Exclude demographic columns from encoding
        demographic_columns = ['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
        categorical_columns = [col for col in categorical_columns if col not in demographic_columns]
        
        # Ensure Location Type, Vehicle Type, and Incident Zip are included if they exist
        if 'Location Type' in df_transformed.columns and 'Location Type' not in categorical_columns:
            categorical_columns.append('Location Type')
        if 'Vehicle Type' in df_transformed.columns and 'Vehicle Type' not in categorical_columns:
            categorical_columns.append('Vehicle Type')
        if 'Incident Zip' in df_transformed.columns and 'Incident Zip' not in categorical_columns:
            categorical_columns.append('Incident Zip')
        
        if categorical_columns:
            print(f"   📋 Categorical columns to encode: {categorical_columns}")
            
            # Special handling for specific columns
            if 'Location Type' in categorical_columns:
                print(f"   🏢 Location Type will be encoded with high-cardinality encoding")
            if 'Vehicle Type' in categorical_columns:
                print(f"   🚗 Vehicle Type will be encoded with high-cardinality encoding (keeping all vehicle types)")
            if 'Incident Zip' in categorical_columns:
                print(f"   📮 Incident Zip will be encoded with high-cardinality encoding (zip codes are high-cardinality)")
            
            high_card_encoder = self.HighCardinalityEncoder(categorical_columns)
            high_card_encoder.fit(df_transformed)
            df_transformed[categorical_columns] = high_card_encoder.transform(df_transformed[categorical_columns])
            print(f"   ✅ Encoded {len(categorical_columns)} categorical columns")
        else:
            print("   ✅ No additional categorical columns to encode")
        
        # Update the main dataframe
        self.df = df_transformed
        return df_transformed

    def clean_feature_names(self):
        """Clean feature names by replacing spaces with underscores"""
        self.df.columns = self.df.columns.str.replace(' ', '_')

    def handle_null_values_after_transformations(self):
        """Handle null values after all transformations"""
        print("\n" + "="*60)
        print("HANDLING NULL VALUES AFTER TRANSFORMATIONS")
        print("="*60)
        
        # Check for null values
        null_counts = self.df.isnull().sum()
        total_null = null_counts.sum()
        
        if total_null == 0:
            print("✅ No null values found after transformations")
            return
        
        print(f"⚠️  Found {total_null} null values after transformations:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   - {col}: {count} null values")
        
        # Handle null values based on column type - ONLY for columns that actually have null values
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:  # Only process columns that actually have null values
                if self.df[col].dtype in ['object', 'category']:
                    # For categorical columns, fill with mode
                    mode_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    self.df[col] = self.df[col].fillna(mode_value)
                    print(f"   - {col}: Filled {null_count} null values with mode '{mode_value}'")
                else:
                    # For numerical columns, fill with median
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"   - {col}: Filled {null_count} null values with median {median_value:.2f}")
        
        # Verify no null values remain
        remaining_nulls = self.df.isnull().sum().sum()
        if remaining_nulls == 0:
            print("✅ All null values successfully handled")
        else:
            print(f"⚠️  {remaining_nulls} null values still remain")

    def remove_negative_duration(self):
        """Remove rows with negative duration"""
        print("\n" + "="*60)
        print("REMOVING NEGATIVE DURATION")
        print("="*60)
        
        if 'duration' not in self.df.columns:
            print("⚠️  'duration' column not found - skipping")
            return
        
        initial_count = len(self.df)
        negative_duration = (self.df['duration'] < 0).sum()
        
        if negative_duration > 0:
            print(f"📊 Found {negative_duration} rows with negative duration")
            self.df = self.df[self.df['duration'] >= 0]
            removed_count = initial_count - len(self.df)
            print(f"🗑️  Removed {removed_count} rows with negative duration")
        else:
            print("✅ No negative duration values found")
        
        print(f"📊 Remaining rows: {len(self.df):,}")

    def fill_missing_boroughs_with_zip_codes(self, df):
        """Fill missing borough entries using NYC zip codes"""
        print("\n" + "="*60)
        print("FILLING MISSING BOROUGHS WITH ZIP CODES")
        print("="*60)
        
        print(f"🔍 DEBUG: Method called with dataframe shape: {df.shape}")
        print(f"🔍 DEBUG: Available columns: {list(df.columns)}")
        
        if 'Borough' not in df.columns or 'Incident Zip' not in df.columns:
            print("⚠️  'Borough' or 'Incident Zip' column not found - skipping")
            print(f"🔍 DEBUG: Borough column exists: {'Borough' in df.columns}")
            print(f"🔍 DEBUG: Incident Zip column exists: {'Incident Zip' in df.columns}")
            return df
        
        # Load NYC zip codes
        zip_codes_file = Path(f'{self.BASE_PATH}/data/nyc-zip-codes.csv')
        print(f"🔍 DEBUG: Looking for zip codes file at: {zip_codes_file}")
        print(f"🔍 DEBUG: File exists: {zip_codes_file.exists()}")
        
        if not zip_codes_file.exists():
            print(f"❌ NYC zip codes file not found: {zip_codes_file}")
            print("   Skipping borough filling")
            return df
        
        try:
            # Load NYC zip codes
            nyc_zip_codes_df = pd.read_csv(zip_codes_file)
            print(f"✅ Loaded NYC zip codes file: {nyc_zip_codes_df.shape}")
            print(f"🔍 DEBUG: ZIP codes file columns: {list(nyc_zip_codes_df.columns)}")
            print(f"🔍 DEBUG: First few rows of ZIP codes:")
            print(nyc_zip_codes_df.head())
            
            # Create zip code to borough mapping
            zip_to_borough = dict(zip(nyc_zip_codes_df['ZipCode'], nyc_zip_codes_df['Borough'].str.upper()))
            print(f"📊 Created zip code to borough mapping: {len(zip_to_borough)} entries")
            
            # Show sample of mapping
            print(f"📋 Sample zip code to borough mapping:")
            sample_mapping = list(zip_to_borough.items())[:10]
            for zip_code, borough in sample_mapping:
                print(f"   - {zip_code} → {borough}")
            
            df_processed = df.copy()
            
            # Identify missing borough entries
            missing_borough_mask = df_processed['Borough'].isna() | (df_processed['Borough'].astype(str).str.strip() == '')
            missing_borough_count = missing_borough_mask.sum()
            
            print(f"🔍 DEBUG: Missing borough analysis:")
            print(f"   - Total rows: {len(df_processed)}")
            print(f"   - Missing borough rows: {missing_borough_count}")
            print(f"   - Borough value counts:")
            print(df_processed['Borough'].value_counts(dropna=False).head(10))
            
            if missing_borough_count == 0:
                print("✅ No missing borough entries found!")
                return df_processed
            
            print(f"\n📊 Found {missing_borough_count:,} records with missing borough entries")
            
            # Check which of these have valid incident zip codes
            missing_borough_df = df_processed[missing_borough_mask]
            print(f"🔍 DEBUG: Incident Zip data types in missing borough rows:")
            print(f"   - Incident Zip dtype: {missing_borough_df['Incident Zip'].dtype}")
            print(f"   - Sample Incident Zip values: {missing_borough_df['Incident Zip'].head(10).tolist()}")
            
            # Convert Incident Zip to string for comparison
            missing_borough_df['Incident Zip'] = missing_borough_df['Incident Zip'].astype(str)
            zip_to_borough_keys = [str(k) for k in zip_to_borough.keys()]
            
            valid_zip_mask = missing_borough_df['Incident Zip'].isin(zip_to_borough_keys)
            valid_zip_count = valid_zip_mask.sum()
            invalid_zip_count = (~valid_zip_mask).sum()
            
            print(f"📊 Missing borough records analysis:")
            print(f"   - Records with valid zip codes: {valid_zip_count:,}")
            print(f"   - Records without valid zip codes: {invalid_zip_count:,}")
            
            # Show examples of records without valid zip codes
            if invalid_zip_count > 0:
                invalid_zip_examples = missing_borough_df[~valid_zip_mask][['Incident Zip', 'Complaint Type']].head(5)
                print(f"\n📋 Examples of records without valid zip codes:")
                for idx, row in invalid_zip_examples.iterrows():
                    print(f"   - Zip: {row['Incident Zip']}, Complaint: {row['Complaint Type']}")
            
            # Fill borough for records with valid zip codes
            filled_count = 0
            for idx in missing_borough_df[valid_zip_mask].index:
                zip_code = str(df_processed.loc[idx, 'Incident Zip'])
                borough = zip_to_borough[zip_code]
                df_processed.loc[idx, 'Borough'] = borough
                filled_count += 1
            
            print(f"\n✅ Filled borough for {filled_count:,} records using zip codes")
            
            # Remove records that still have missing borough (no valid zip code)
            final_missing_mask = df_processed['Borough'].isna() | (df_processed['Borough'].astype(str).str.strip() == '')
            final_missing_count = final_missing_mask.sum()
            
            if final_missing_count > 0:
                print(f"⚠️  Removing {final_missing_count:,} records that still have missing borough (no valid zip code)")
                df_processed = df_processed[~final_missing_mask]
            
            print(f"📊 Final dataset shape: {df_processed.shape}")
            print(f"📊 Borough distribution after filling:")
            borough_counts = df_processed['Borough'].value_counts()
            for borough, count in borough_counts.items():
                percentage = (count / len(df_processed)) * 100
                print(f"   - {borough}: {count:,} ({percentage:.2f}%)")
            
            return df_processed
            
        except Exception as e:
            print(f"❌ Error filling missing boroughs: {str(e)}")
            import traceback
            traceback.print_exc()
            return df

    def load_and_merge_demographics(self, df):
        """Load and merge demographic data"""
        print("\n" + "="*60)
        print("LOADING AND MERGING DEMOGRAPHIC DATA")
        print("="*60)
        
        try:
            # Load demographic data from text file
            demo_file_path = f'{self.BASE_PATH}/data/bevoelkerungsgruppen.txt'
            print(f"Loading demographic data from: {demo_file_path}")
            
            # Parse the formatted text file
            demo_data = []
            current_borough = None
            
            with open(demo_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('===') or line.startswith('Bevölkerungsverteilung'):
                    continue
                
                # Check if this is a borough name
                if line in ['BROOKLYN:', 'MANHATTAN:', 'QUEENS:', 'BRONX:', 'STATEN ISLAND:']:
                    current_borough = line.replace(':', '')
                    continue
                
                # Parse demographic values
                if current_borough and ':' in line:
                    if 'Weiße:' in line:
                        weisse = float(line.split('(')[0].split(':')[1].strip())
                    elif 'Afroamerikaner:' in line:
                        afroamerikaner = float(line.split('(')[0].split(':')[1].strip())
                    elif 'Asiaten:' in line:
                        asiaten = float(line.split('(')[0].split(':')[1].strip())
                    elif 'Hispanics/Latinos:' in line:
                        hispanics = float(line.split('(')[0].split(':')[1].strip())
                        
                        # Add the complete borough data
                        demo_data.append({
                            'Stadtteil': current_borough,
                            'Weisse': weisse,
                            'Afroamerikaner': afroamerikaner,
                            'Asiaten': asiaten,
                            'Hispanics': hispanics
                        })
                        current_borough = None
            
            demo_df = pd.DataFrame(demo_data)
            print(f"✅ Demographic data loaded: {demo_df.shape}")
            print(f"📊 Boroughs in demographic data: {list(demo_df['Stadtteil'])}")
            print(f"📊 Demographic data preview:")
            print(demo_df)
            
            # Normalize borough names for better matching
            print(f"\n🔄 Normalizing borough names for matching...")
            
            # Create a mapping dictionary for borough name variations
            borough_mapping = {
                'BROOKLYN': 'BROOKLYN',
                'MANHATTAN': 'MANHATTAN', 
                'QUEENS': 'QUEENS',
                'BRONX': 'BRONX',
                'STATEN ISLAND': 'STATEN ISLAND',
                'Staten Island': 'STATEN ISLAND',
                'Staten island': 'STATEN ISLAND',
                'staten island': 'STATEN ISLAND',
                'brooklyn': 'BROOKLYN',
                'manhattan': 'MANHATTAN',
                'queens': 'QUEENS',
                'bronx': 'BRONX'
            }
            
            # Normalize borough names in main dataframe
            df['Borough_Normalized'] = df['Borough'].str.upper().str.strip()
            df['Borough_Normalized'] = df['Borough_Normalized'].map(borough_mapping).fillna(df['Borough_Normalized'])
            
            # Show borough distribution before merge
            print(f"📊 Borough distribution in main data:")
            borough_counts = df['Borough_Normalized'].value_counts()
            for borough, count in borough_counts.items():
                print(f"   - '{borough}': {count:,} rows")
            
            # Merge with main dataframe using normalized names
            df = df.merge(demo_df, left_on='Borough_Normalized', right_on='Stadtteil', how='left')
            
            # Remove the temporary normalized column
            df = df.drop(columns=['Borough_Normalized'])
            
            # Check merge results
            merged_count = df['Weisse'].notna().sum()
            total_count = len(df)
            print(f"📊 Merge results: {merged_count:,} out of {total_count:,} rows have demographic data ({merged_count/total_count*100:.1f}%)")
            
            # Show sample of merged data
            print(f"📊 Sample merged data:")
            sample_cols = ['Borough', 'Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
            print(df[sample_cols].head())
            
            # Debug: Show boroughs that couldn't be merged
            if merged_count < total_count:
                print(f"\n🔍 Borough mapping analysis:")
                unmapped_mask = df['Weisse'].isna()
                unmapped_boroughs = df.loc[unmapped_mask, 'Borough'].value_counts()
                print(f"📊 Unmapped borough values ({unmapped_boroughs.sum()} total rows):")
                for borough, count in unmapped_boroughs.items():
                    print(f"   - '{borough}': {count:,} rows")
                
                # Show expected borough names from demographic data
                print(f"\n📋 Available borough names in demographic data:")
                for borough in demo_df['Stadtteil']:
                    print(f"   - '{borough}'")
            
            # Remove rows with missing demographic values instead of filling with 0
            demo_cols = ['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
            missing_demo_mask = df[demo_cols].isna().any(axis=1)
            missing_demo_count = missing_demo_mask.sum()
            
            if missing_demo_count > 0:
                print(f"   🗑️  Removing {missing_demo_count:,} rows with missing demographic data")
                df = df[~missing_demo_mask]
                print(f"   📊 Remaining rows after removal: {len(df):,}")
            else:
                print(f"   ✅ All rows have complete demographic data")
            
            # Remove the Stadtteil column from merge
            if 'Stadtteil' in df.columns:
                df = df.drop(columns=['Stadtteil'])
            
            print("✅ Demographic data successfully merged")
            return df
            
        except FileNotFoundError:
            print("❌ Demographic data file not found")
            print("   Creating dummy demographic columns with zeros")
            
            # Create dummy demographic columns
            df['Weisse'] = 0.0
            df['Afroamerikaner'] = 0.0
            df['Asiaten'] = 0.0
            df['Hispanics'] = 0.0
            
            return df
        except Exception as e:
            print(f"❌ Error loading demographic data: {str(e)}")
            print("   Creating dummy demographic columns with zeros")
            
            # Create dummy demographic columns
            df['Weisse'] = 0.0
            df['Afroamerikaner'] = 0.0
            df['Asiaten'] = 0.0
            df['Hispanics'] = 0.0
            
            return df

    def remove_missing_incident_zip(self):
        """Remove rows with missing Incident Zip"""
        print("\n" + "="*60)
        print("REMOVING MISSING INCIDENT ZIP")
        print("="*60)
        
        if 'Incident Zip' not in self.df.columns:
            print("⚠️  'Incident Zip' column not found - skipping")
            return
        
        initial_count = len(self.df)
        
        # Remove rows with missing zip codes
        self.df = self.df.dropna(subset=['Incident Zip'])
        
        # Also remove empty strings
        self.df = self.df[self.df['Incident Zip'] != '']
        
        removed_count = initial_count - len(self.df)
        print(f"🗑️  Removed {removed_count:,} rows with missing Incident Zip")
        print(f"📊 Remaining rows: {len(self.df):,}")

    def convert_incident_zip_to_integer(self):
        """Convert Incident Zip to integer"""
        print("\n" + "="*60)
        print("CONVERTING INCIDENT ZIP TO INTEGER")
        print("="*60)
        
        if 'Incident Zip' not in self.df.columns:
            print("⚠️  'Incident Zip' column not found - skipping")
            return
        
        # Convert to integer
        self.df['Incident Zip'] = pd.to_numeric(self.df['Incident Zip'], errors='coerce').astype('Int64')
        
        print("✅ Incident Zip converted to integer")

    def rename_duration_column(self):
        """Rename duration column to include units"""
        print("\n" + "="*60)
        print("RENAMING DURATION COLUMN")
        print("="*60)
        
        if 'duration' in self.df.columns:
            self.df = self.df.rename(columns={'duration': 'duration_[days]'})
            print("✅ Duration column renamed to 'duration_[days]'")
        else:
            print("⚠️  'duration' column not found - skipping")

    def round_duration_to_two_decimals(self):
        """Round duration values to two decimal places"""
        print("\n" + "="*60)
        print("ROUNDING DURATION TO TWO DECIMALS")
        print("="*60)
        
        duration_col = 'duration_[days]' if 'duration_[days]' in self.df.columns else 'duration'
        
        if duration_col in self.df.columns:
            self.df[duration_col] = self.df[duration_col].round(2)
            print("✅ Duration values rounded to two decimal places")
        else:
            print("⚠️  Duration column not found - skipping")

    def handle_demographic_outliers(self):
        """Handle outliers in demographic columns"""
        print("\n" + "="*60)
        print("HANDLING DEMOGRAPHIC OUTLIERS")
        print("="*60)
        
        demo_cols = ['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
        
        for col in demo_cols:
            if col in self.df.columns:
                # Cap values at 1.0 (100%)
                outliers = (self.df[col] > 1.0).sum()
                if outliers > 0:
                    print(f"   - {col}: {outliers} values > 1.0 capped at 1.0")
                    self.df[col] = self.df[col].clip(upper=1.0)
                
                # Cap values at 0.0 (0%)
                outliers = (self.df[col] < 0.0).sum()
                if outliers > 0:
                    print(f"   - {col}: {outliers} values < 0.0 capped at 0.0")
                    self.df[col] = self.df[col].clip(lower=0.0)
        
        print("✅ Demographic outlier handling completed")

    def run_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("="*80)
        print("NYC 311 DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Load data
        self.load_and_check_data()
        
        if self.df is None:
            print("❌ Failed to load data - stopping pipeline")
            return None
        
        # Handle vehicle type
        self.handle_vehicle_type()
        
        # Handle location type
        self.handle_location_type()
        
        # Consolidate location types
        self.consolidate_location_types()
        
        # Consolidate complaint types
        self.consolidate_complaint_types()
        
        # Calculate duration
        self.calculate_duration()
        
        # Remove negative duration
        self.remove_negative_duration()
        
        # Remove missing incident zip
        self.remove_missing_incident_zip()
        
        # Convert incident zip to integer
        self.convert_incident_zip_to_integer()
        
        # Rename duration column
        self.rename_duration_column()
        
        # Round duration to two decimals
        self.round_duration_to_two_decimals()
        
        # Handle outliers
        self.handle_outliers()
        
        # Apply transformations with demographics (includes encoding and merging)
        self.apply_transformations_with_demographics()
        
        # Handle null values after transformations
        self.handle_null_values_after_transformations()
        
        # Handle demographic outliers
        self.handle_demographic_outliers()
        
        # Clean feature names
        self.clean_feature_names()
        
        # Remove specific columns at the end
        print("\n" + "="*60)
        print("REMOVING FINAL COLUMNS")
        print("="*60)
        
        columns_to_remove = ['Location', 'Created_Date']
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                print(f"   🗑️  Removed column: {col}")
            else:
                print(f"   ⚠️  Column '{col}' not found - skipping")
        
        # Save processed data
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Create output directory if it doesn't exist
        Path(f'{self.BASE_PATH}/data').mkdir(parents=True, exist_ok=True)
        
        # Save the processed data
        output_path = f'{self.BASE_PATH}/data/processed_311_data_consolidated.csv'
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to: {output_path}")
        print(f"📊 Final dataset shape: {self.df.shape}")
        
        # Show final columns
        print(f"\n📋 Final columns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. '{col}'")
        
        # Show first 10 rows
        print(f"\n📄 First 10 rows:")
        print(self.df.head(10))
        
        # Verify demographic columns are present
        demo_cols = ['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']
        present_demo_cols = [col for col in demo_cols if col in self.df.columns]
        print(f"\n👥 Demographic columns present: {present_demo_cols}")
        
        if len(present_demo_cols) == len(demo_cols):
            print("✅ All demographic columns successfully included in final dataset")
        else:
            missing_cols = [col for col in demo_cols if col not in self.df.columns]
            print(f"⚠️  Missing demographic columns: {missing_cols}")
        
        print("="*80)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return self.df

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessorOOP()
    
    # Run the complete pipeline
    processed_df = preprocessor.run_pipeline()
    
    if processed_df is not None:
        print("\n🎉 Preprocessing completed successfully!")
        print(f"📊 Final dataset: {processed_df.shape[0]:,} rows, {processed_df.shape[1]} columns")
        print("📁 Output file: data/processed_311_data_consolidated.csv")
    else:
        print("\n❌ Preprocessing failed!") 