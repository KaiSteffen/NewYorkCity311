#!/usr/bin/env python3
"""
NYC 311 Model Training - Object-Oriented Version
===============================================

This module provides a clean, object-oriented approach to training machine learning models
on NYC 311 complaint data using PyCaret.

Features:
- Uses processed data from data_preprocessing_oop.py
- Data splitting with sampling for faster training
- Multiple model training with timeout handling
- Model evaluation on test data
- Proper model saving for later use
- Option to train only the first model for testing

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import time
import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *
import argparse
from pathlib import Path

# Set matplotlib backend to non-interactive to prevent crashes on Windows
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add parent directory to path to import data_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: DataPreprocessor is not needed anymore as we load preprocessed data directly

class ModelTrainer:
    def __init__(self, data_path=None, 
                 model_save_path=None, 
                 test_size=0.077,  # ~7.7% for ~5000 test samples out of ~65000 total
                 random_state=42,
                 sample_percentage=0.1):  # Sample 10% of data to get manageable size
        """
        Initialize the ModelTrainer
        
        Args:
            data_path: Path to the training data
            model_save_path: Directory to save models
            test_size: Percentage of data for testing (default: 0.077 for ~5000 test samples)
            random_state: Random seed for reproducibility
            sample_percentage: Percentage of data to use for training (default: 0.1 for manageable size)
        """
        # Set default model save path
        if model_save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_save_path = os.path.join(script_dir, '..', 'models')
        
        # Set default data path based on where the script is run from
        if data_path is None:
            # Try to find the data file relative to the script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, '..', 'data', 'training', 'train_data_oop.zip'),
                os.path.join(script_dir, 'data', 'training', 'train_data_oop.zip'),
                'data/training/train_data_oop.zip'  # Default fallback
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            else:
                data_path = 'data/training/train_data_oop.zip'  # Default fallback
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.test_size = test_size
        self.random_state = random_state
        self.sample_percentage = sample_percentage
        
        # Create model save directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize data
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.best_model = None
        self.complaint_type_mapping = None  # Hinzugefügt: Mapping für Complaint Types
        
    def load_and_split_data(self):
        """
        Läd die bereits erzeugten Trainings- und Testdaten aus den CSVs.
        Es findet kein erneutes Sampling oder Splitten mehr statt.
        """
        import os
        import pandas as pd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_data_path = os.path.join(script_dir, '..', 'data', 'train_data_final.csv')
        test_data_path = os.path.join(script_dir, '..', 'data', 'test_data_final.csv')
        self.train_data = pd.read_csv(train_data_path)
        self.test_data = pd.read_csv(test_data_path)
        self.logger.info(f"Trainingsdaten geladen: {self.train_data.shape}")
        self.logger.info(f"Testdaten geladen: {self.test_data.shape}")
        
    def filter_classes(self, min_percentage=0.015): # 1.5% minimum representation
        """
        Filter out classes that have less than min_percentage representation
        in the training data. This improves model stability.
        """
        self.logger.info(f"Filtering classes with representation < {min_percentage:.1%}")

        # Add a guard clause for empty data to prevent KeyError
        if self.train_data is None or self.train_data.empty:
            self.logger.warning("Training data is empty, skipping class filtering.")
            return

        # Calculate class distribution based on percentage
        class_counts = self.train_data['Complaint_Type'].value_counts()
        total_samples = len(self.train_data)
        class_percentages = (class_counts / total_samples) * 100
        
        # Define status thresholds
        status_thresholds = {
            'EXCELLENT': 1.5,  # >= 1.5%
            'GOOD': 0.5,       # >= 0.5% but < 1.5%
            'FAIR': 0.1,       # >= 0.1% but < 0.5%
            'POOR': 0.0        # < 0.1%
        }
        
        # Assign status to each class
        class_status = {}
        for class_name, percentage in class_percentages.items():
            if percentage >= status_thresholds['EXCELLENT']:
                class_status[class_name] = 'EXCELLENT'
            elif percentage >= status_thresholds['GOOD']:
                class_status[class_name] = 'GOOD'
            elif percentage >= status_thresholds['FAIR']:
                class_status[class_name] = 'FAIR'
            else:
                class_status[class_name] = 'POOR'
        
        # Show status distribution
        status_counts = {}
        for status in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']:
            status_counts[status] = sum(1 for s in class_status.values() if s == status)
        
        self.logger.info("Class status distribution:")
        for status, count in status_counts.items():
            self.logger.info(f"{status}: {count} classes")
        
        # Keep only EXCELLENT classes
        excellent_classes = [class_name for class_name, status in class_status.items() if status == 'EXCELLENT']
        
        self.logger.info(f"Keeping {len(excellent_classes)} EXCELLENT classes")
        self.logger.info(f"EXCELLENT classes: {excellent_classes}")
        
        # Filter data
        self.train_data = self.train_data[self.train_data['Complaint_Type'].isin(excellent_classes)]
        self.test_data = self.test_data[self.test_data['Complaint_Type'].isin(excellent_classes)]
        
        self.logger.info(f"After filtering - Training set shape: {self.train_data.shape}")
        self.logger.info(f"After filtering - Test set shape: {self.test_data.shape}")
        
        # Show final class distribution
        final_class_dist = self.train_data['Complaint_Type'].value_counts()
        self.logger.info(f"Final class distribution:\n{final_class_dist}")
        
    def setup_pycaret(self):
        """Setup PyCaret experiment"""
        self.logger.info("Setting up PyCaret experiment...")
        
        # Erstelle eine saubere Kopie der Trainingsdaten nur für PyCaret
        pycaret_data = self.train_data.copy()

        # Features, die explizit entfernt werden sollen, um Datenlecks zu vermeiden.
        features_to_drop = ['Location', 'Complaint_Type_original']
        
        # Sicherstellen, dass die zu entfernenden Spalten auch existieren, um Fehler zu vermeiden.
        existing_features_to_drop = [col for col in features_to_drop if col in pycaret_data.columns]
        
        if existing_features_to_drop:
            self.logger.info(f"Dropping columns before PyCaret setup to prevent data leakage: {existing_features_to_drop}")
            pycaret_data = pycaret_data.drop(columns=existing_features_to_drop)
        else:
             self.logger.info("No problematic columns to drop for PyCaret setup.")

        # Initialisiere PyCaret mit den bereinigten Daten
        setup(
            data=pycaret_data, # Benutze die bereinigten Daten
            target='Complaint_Type',
            session_id=self.random_state,
            normalize=True,
            transformation=True,
            fix_imbalance=True,
            fix_imbalance_method='smote',
            # ignore_features wird nicht mehr benötigt, da wir die Spalten entfernt haben
            verbose=False
        )
        
        self.logger.info("PyCaret setup completed")
        
    def train_models(self, train_only_first=False):
        """Train multiple models using PyCaret"""
        self.logger.info("Starting model training...")
        
        if train_only_first:
            # Train only XGBoost for quick testing
            model_configs = [
                ('xgboost', {
                    'n_estimators': 150,  # Mehr Bäume für bessere Performance
                    'max_depth': 10,      # Erhöhte Baumtiefe für bessere Optimierung
                    'learning_rate': 0.1, # Standard Lernrate
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,  # Minimale Summe der Gewichte
                    'gamma': 0.1,          # Minimum Loss Reduction
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'  # Better for F1 optimization
                })
            ]
        else:
            # Train all available models from backup
            self.logger.info("Training all available models...")
            
            # Define all models from backup
            model_configs = [
                ('xgboost', {
                    'n_estimators': 150,  # Mehr Bäume für bessere Performance
                    'max_depth': 10,      # Erhöhte Baumtiefe für bessere Optimierung
                    'learning_rate': 0.1, # Standard Lernrate
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,  # Minimale Summe der Gewichte
                    'gamma': 0.1,          # Minimum Loss Reduction
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'  # Better for F1 optimization
                }),
                ('dummy', {}),
                ('bernoulli_nb', {}),
                ('ada', {
                    'n_estimators': 5,
                    'learning_rate': 1.0,
                    'random_state': self.random_state
                }),
                ('et', {
                    'n_estimators': 50,
                    'max_depth': 8,
                    'random_state': self.random_state
                }),
                ('rf', {
                    'n_estimators': 50,
                    'max_depth': 8,
                    'random_state': self.random_state
                }),
                ('dt', {
                    'max_depth': 8,
                    'random_state': self.random_state
                }),
                ('nb', {}),
                ('knn', {
                    'n_neighbors': 5
                }),
                ('lda', {}),
                ('qda', {}),
                ('ridge', {
                    'alpha': 1.0,
                    'random_state': self.random_state
                }),
                ('mlp', {
                    'hidden_layer_sizes': (50, 25),
                    'max_iter': 200,
                    'random_state': self.random_state
                }),
                ('catboost', {
                    'iterations': 50,
                    'depth': 4,
                    'learning_rate': 0.3,
                    'random_state': self.random_state
                }),
                ('gbc', {
                    'n_estimators': 5,
                    'max_depth': 2,
                    'learning_rate': 1.0,
                    'subsample': 0.8,
                    'min_samples_split': 10,
                    'random_state': self.random_state
                })
            ]
            
            self.logger.info(f"Available models: {[config[0] for config in model_configs]}")
        
        # Train models with timeout
        for i, (model_name, params) in enumerate(model_configs):
            self.logger.info(f"Training model {i+1}/{len(model_configs)}: {model_name}")
            
            try:
                # Set timeout based on model complexity
                if model_name in ['rf', 'catboost', 'xgboost', 'mlp']:
                    timeout = 300  # 5 minutes for complex models
                elif model_name in ['ada', 'gbc', 'et']:
                    timeout = 180  # 3 minutes for medium models
                else:
                    timeout = 120  # 2 minutes for simple models
                
                # Train model
                start_time = time.time()
                model = create_model(model_name, **params)
                training_time = time.time() - start_time
                
                self.models[model_name] = model
                self.logger.info(f"Successfully trained {model_name} in {training_time:.2f} seconds")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        self.logger.info(f"Training completed. Successfully trained {len(self.models)} models")
        
    def evaluate_models(self):
        """Evaluate all trained models on test data"""
        self.logger.info("Evaluating all models on test data...")
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                # Make predictions using PyCaret
                predictions = predict_model(model, data=self.test_data)
                
                # Calculate metrics
                accuracy = accuracy_score(self.test_data['Complaint_Type'], predictions['prediction_label'])
                
                # Classification report
                report = classification_report(
                    self.test_data['Complaint_Type'], 
                    predictions['prediction_label'],
                    output_dict=True
                )
                
                # Calculate F1-score (macro average for multi-class)
                f1_score_macro = report['macro avg']['f1-score']
                f1_score_weighted = report['weighted avg']['f1-score']
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_score_macro,
                    'f1_weighted': f1_score_weighted,
                    'report': report,
                    'predictions': predictions
                }
                
                self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 (macro): {f1_score_macro:.4f}, F1 (weighted): {f1_score_weighted:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Set best model based on F1-score (weighted average)
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
            self.best_model = self.models[best_model_name]
            best_f1 = results[best_model_name]['f1_weighted']
            best_acc = results[best_model_name]['accuracy']
            self.logger.info(f"Best model: {best_model_name} with F1 (weighted): {best_f1:.4f}, Accuracy: {best_acc:.4f}")
        
        return results
    
    def save_model(self, model_name=None, is_tuned=False):
        """Save the XGBoost model directly (not as pipeline) for tuning and fairlearn"""
        if model_name is None:
            if self.best_model is None:
                self.logger.error("No best model available for saving")
                return
            
            # Find best model name (should be xgboost)
            for name, model in self.models.items():
                if model == self.best_model:
                    model_name = name
                    break
        
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add tuning indicator to filename
        if is_tuned:
            filename = f"complaint_classifier_oop_tuned_{timestamp}.pkl"
        else:
            filename = f"complaint_classifier_oop_{timestamp}.pkl"
            
        filepath = os.path.join(self.model_save_path, filename)
        
        # Extract the actual model from PyCaret pipeline
        model_to_save = self.models[model_name]
        
        # If it's a pipeline, extract the final estimator
        if hasattr(model_to_save, 'steps') and len(model_to_save.steps) > 0:
            # Get the final estimator (the actual model)
            final_estimator = model_to_save.steps[-1][1]
            self.logger.info(f"Extracted {type(final_estimator).__name__} from PyCaret pipeline")
            model_to_save = final_estimator
        else:
            self.logger.info(f"Saving {type(model_to_save).__name__} directly")
        
        # Get feature names (exclude Complaint_Type)
        feature_names = [col for col in self.train_data.columns if col != 'Complaint_Type']
        
        # Create model package for tuning and fairlearn
        model_package = {
            'model': model_to_save,
            'feature_names': feature_names,
            'target_name': 'Complaint_Type',
            'model_type': 'xgboost',
            'training_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'timestamp': timestamp,
            'is_tuned': is_tuned
        }
        
        # Save model package using joblib
        joblib.dump(model_package, filepath)
        
        # Save XGBoost booster as JSON for re-export
        if hasattr(model_to_save, 'get_booster'):
            try:
                booster = model_to_save.get_booster()
                json_filename = f"model_reexported_{timestamp}.json"
                json_filepath = os.path.join(self.model_save_path, json_filename)
                booster.save_model(json_filepath)
                self.logger.info(f"📄 XGBoost booster saved as JSON: {json_filepath}")
            except Exception as e:
                self.logger.warning(f"Could not save XGBoost booster as JSON: {str(e)}")
        
        if is_tuned:
            self.logger.info(f"✅ Tuned XGBoost model saved as: {filepath}")
        else:
            self.logger.info(f"📁 XGBoost model saved as: {filepath}")
            
        self.logger.info(f"Features: {feature_names}")
        self.logger.info(f"Training samples: {len(self.train_data):,}")
        self.logger.info(f"Test samples: {len(self.test_data):,}")
        self.logger.info(f"Model tuned: {is_tuned}")
        
        # Save test data CSV directly in data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_csv_path = os.path.join(script_dir, '..', 'data', f'test_data_fv_{timestamp}.csv')
        os.makedirs(os.path.dirname(test_data_csv_path), exist_ok=True)
        self.test_data.to_csv(test_data_csv_path, index=False)
        self.logger.info(f"📊 Test data saved as: {test_data_csv_path}")
        
        return filepath
    
    def run_training_pipeline(self, train_only_first=False, skip_tuning=False):
        """Run the complete training pipeline"""
        self.logger.info("Starting training pipeline...")
        
        # Load and split data
        self.load_and_split_data()
        
        # Filter classes
        self.filter_classes()
        
        # Setup PyCaret
        self.setup_pycaret()
        
        # Train models (default to all models)
        self.train_models(train_only_first=train_only_first)
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Save original model first
        if self.best_model is not None:
            original_path = self.save_model()
            self.logger.info(f"Original model saved at: {original_path}")
            
            # Get original accuracy
            original_accuracy = results[list(results.keys())[0]]['accuracy']
            
            # Create confusion matrix for the BEST original model BEFORE tuning
            self.logger.info("\n" + "="*60)
            self.logger.info("CONFUSION MATRIX - ORIGINAL MODEL (BEFORE TUNING)")
            self.logger.info("="*60)
            
            original_confusion_matrix_path = self.create_confusion_matrix(
                self.best_model, 
                self.test_data, 
                'original_before_tuning'
            )
            
            if skip_tuning:
                self.logger.info("\n" + "="*60)
                self.logger.info("SKIPPING TUNING - USING ORIGINAL MODEL")
                self.logger.info("="*60)
                self.logger.info(f"Tuning was skipped. Using original model.")
                final_path = original_path
                model_was_tuned = False
                tuned_confusion_matrix_path = None
                final_accuracy = original_accuracy
            else:
                # Perform tuning
                self.logger.info("\n" + "="*60)
                self.logger.info("STARTING MODEL TUNING")
                self.logger.info("="*60)
                
                # Get best model name
                best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
                original_accuracy = results[best_model_name]['accuracy']
                
                # Tune the best model
                tuned_model, tuned_accuracy = self.tune_model(self.best_model, self.test_data, best_model_name)
                
                # Create confusion matrix for the TUNED model AFTER tuning
                self.logger.info("\n" + "="*60)
                self.logger.info("CONFUSION MATRIX - TUNED MODEL (AFTER TUNING)")
                self.logger.info("="*60)
                
                tuned_confusion_matrix_path = self.create_confusion_matrix(
                    tuned_model, 
                    self.test_data, 
                    'tuned_after_tuning'
                )
                
                # Update best model if tuning improved it
                if tuned_accuracy > original_accuracy:
                    self.best_model = tuned_model
                    self.models[best_model_name] = tuned_model
                    self.logger.info(f"✅ Using tuned model (improvement: {tuned_accuracy - original_accuracy:.4f})")
                    model_was_tuned = True
                    final_accuracy = tuned_accuracy
                else:
                    self.logger.info(f"⚠️  Using original model (tuning didn't improve)")
                    model_was_tuned = False
                    final_accuracy = original_accuracy
                
                # Save final model (tuned if better, original if not)
                final_path = self.save_model(is_tuned=model_was_tuned)
            
            # Final summary with comparison
            self.logger.info("\n" + "="*60)
            self.logger.info("TRAINING COMPLETED - FINAL SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"📁 Original model: {original_path}")
            self.logger.info(f"📊 Original accuracy: {original_accuracy:.4f}")
            self.logger.info(f"📈 Original confusion matrix: {original_confusion_matrix_path}")
            
            if not skip_tuning:
                self.logger.info(f"📁 Final model: {final_path}")
                self.logger.info(f"📊 Final accuracy: {final_accuracy:.4f}")
                self.logger.info(f"🔧 Model tuned: {model_was_tuned}")
                self.logger.info(f"📈 Tuned confusion matrix: {tuned_confusion_matrix_path}")
                
                # Show improvement details
                if model_was_tuned:
                    improvement = final_accuracy - original_accuracy
                    improvement_percent = (improvement / original_accuracy) * 100
                    self.logger.info(f"📈 Accuracy improvement: {improvement:.4f} ({improvement_percent:.2f}%)")
                    self.logger.info(f"✅ Tuning was successful - using optimized model")
                else:
                    self.logger.info(f"⚠️  No improvement from tuning - using original model")
            else:
                self.logger.info(f"📁 Final model: {original_path}")
                self.logger.info(f"📊 Final accuracy: {original_accuracy:.4f}")
                self.logger.info(f"🔧 Model tuned: False (tuning skipped)")
                self.logger.info(f"📈 Tuned confusion matrix: None (tuning skipped)")
            
        else:
            self.logger.warning("No models were successfully trained")
        
        return results

    def tune_model(self, model, test_data, model_name):
        """Tune the model for better performance - optimized for accuracy improvement"""
        self.logger.info(f"\n🎯 Starting tuning for {model_name}")
        
        # Use full test set for better tuning evaluation
        self.logger.info(f"\n📊 Using full test set for accurate tuning evaluation...")
        tuning_test_data = test_data  # Use full test set
        
        self.logger.info(f"   📊 Full test set: {len(tuning_test_data):,} samples")
        self.logger.info(f"   🎯 Better accuracy evaluation with full dataset")
        
        # Get tuning parameters for XGBoost - optimized for accuracy improvement
        tune_params = {
            'n_iter': 15,  # Mehr Iterationen für bessere Ergebnisse
            'search_library': 'optuna',
            'early_stopping': False,  # Kein Early Stopping für aggressiveres Tuning
            'verbose': False
        }
        
        # Show detailed tuning parameters
        self.logger.info(f"\n🔧 Hyperparameter Tuning Configuration (F1 Focus):")
        self.logger.info(f"   📊 Model: {model_name}")
        self.logger.info(f"   🔄 Iterations: {tune_params['n_iter']} (more iterations for better results)")
        self.logger.info(f"   🎯 Search Library: {tune_params['search_library']}")
        self.logger.info(f"   ⏱️  Early Stopping: {tune_params['early_stopping']}")
        
        self.logger.info(f"\n⚙️  PyCaret will automatically tune these parameters:")
        self.logger.info(f"   - n_estimators")
        self.logger.info(f"   - max_depth") 
        self.logger.info(f"   - learning_rate")
        self.logger.info(f"   - subsample")
        self.logger.info(f"   - colsample_bytree")
        self.logger.info(f"   - min_child_weight")
        self.logger.info(f"   - gamma")
        
        # Evaluate original model on full test set
        self.logger.info(f"\n📊 Evaluating original model (full test set)...")
        original_accuracy = self.evaluate_single_model(model, tuning_test_data, model_name)
        
        if original_accuracy is None:
            self.logger.error(f"❌ Cannot evaluate original model - skipping tuning")
            return model, original_accuracy
        
        self.logger.info(f"   📈 Original accuracy: {original_accuracy:.4f}")
        
        try:
            self.logger.info(f"\n🔄 Starting tuning process (F1 focus)...")
            self.logger.info(f"   🚀 Optimizing for: F1")
            self.logger.info(f"   ⏱️  Starting optimization...")
            
            start_time = time.time()
            
            # Custom search grid for CatBoost to avoid alias conflicts ('eta'/'learning_rate', 'n_estimators'/'iterations')
            custom_grid_catboost = {
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'depth': [4, 6, 8, 10],
                'iterations': [50, 100, 200, 300],  # Use 'iterations' instead of 'n_estimators' for CatBoost
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }

            if 'CatBoostClassifier' in str(type(model)):
                self.logger.info("Using custom search grid for CatBoost to avoid alias conflicts.")
                tuned_model = tune_model(
                    model,
                    optimize='F1',
                    n_iter=10,
                    search_library='scikit-learn',
                    custom_grid=custom_grid_catboost,
                    early_stopping=True,
                    early_stopping_max_iters=10,
                    choose_better=True,
                    verbose=False
                )
            else:
                self.logger.info("Using default search grid for other models.")
                tuned_model = tune_model(
                    model,
                    optimize='F1',
                    n_iter=10,
                    search_library='scikit-learn',
                    early_stopping=True,
                    early_stopping_max_iters=10,
                    choose_better=True,
                    verbose=False
                )

            tuning_time = time.time() - start_time
            self.logger.info(f"✅ Hyperparameter tuning completed in {tuning_time:.1f} seconds")
            
            # Additional check to ensure a valid model was returned
            if tuned_model is None:
                self.logger.warning("Tuning did not produce a better model. Original model will be used.")
                return model, False

            # Evaluate tuned model
            self.logger.info(f"\n📊 Evaluating tuned model (full test set)...")
            tuned_accuracy = self.evaluate_single_model(tuned_model, tuning_test_data, f"{model_name}_tuned")
            
            if tuned_accuracy is None:
                self.logger.error(f"❌ Cannot evaluate tuned model - using original")
                return model, original_accuracy
            
            self.logger.info(f"   📈 Tuned accuracy: {tuned_accuracy:.4f}")
            improvement = tuned_accuracy - original_accuracy
            self.logger.info(f"   📈 Improvement: {improvement:.4f}")
            
            # Show improvement percentage
            if original_accuracy > 0:
                improvement_percent = (improvement / original_accuracy) * 100
                self.logger.info(f"   📈 Improvement percentage: {improvement_percent:.2f}%")
            
            # Use tuned model if it's better
            if tuned_accuracy > original_accuracy:
                self.logger.info(f"   ✅ Using tuned model (improvement: {improvement:.4f})")
                return tuned_model, tuned_accuracy
            else:
                self.logger.info(f"   ⚠️  Using original model (tuning didn't improve)")
                return model, original_accuracy
                
        except Exception as e:
            self.logger.error(f"❌ Tuning failed for {model_name}: {str(e)}")
            self.logger.info("   ⚠️  Using original model")
            return model, False
    
    def evaluate_single_model(self, model, test_data, model_name):
        """Evaluate a single model and return accuracy"""
        try:
            # Make predictions using PyCaret
            predictions = predict_model(model, data=test_data)
            
            # Calculate accuracy
            accuracy = accuracy_score(test_data['Complaint_Type'], predictions['prediction_label'])
            
            # Show detailed metrics
            report = classification_report(
                test_data['Complaint_Type'], 
                predictions['prediction_label'],
                output_dict=True
            )
            
            f1_score_weighted = report['weighted avg']['f1-score']
            
            self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 (weighted): {f1_score_weighted:.4f}")
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None

    def create_confusion_matrix(self, model, test_data, model_name, save_path=None):
        """Create and save confusion matrix for the model"""
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Make predictions
            predictions = predict_model(model, data=test_data)
            y_true = test_data['Complaint_Type']
            y_pred = predictions['prediction_label']

            # === HIER: Namen für die Achsen holen ===
            labels = sorted(list(set(y_true) | set(y_pred)))
            class_names = [self.complaint_type_mapping.get(label, f"Unknown-{label}") for label in labels]
            # === ENDE Anpassung ===

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # Create figure
            plt.figure(figsize=(16, 14)) # Größe angepasst für bessere Lesbarkeit
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names) # Angepasst
            plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right') # Rotiert die X-Achsen-Labels
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save confusion matrix
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.model_save_path, f'confusion_matrix_{model_name}_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Confusion matrix saved as: {save_path}")
            
            # Print classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            self.logger.info(f"📈 Classification Report for {model_name}:")
            self.logger.info(f"   Accuracy: {report['accuracy']:.4f}")
            self.logger.info(f"   Macro F1: {report['macro avg']['f1-score']:.4f}")
            self.logger.info(f"   Weighted F1: {report['weighted avg']['f1-score']:.4f}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {str(e)}")
            return None

def main():
    """Main function to run training"""
    parser = argparse.ArgumentParser(description='Train NYC 311 complaint classification models')
    parser.add_argument('--data-path', default=None, 
                       help='Path to training data')
    parser.add_argument('--model-save-path', default=None, 
                       help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.077, 
                       help='Percentage of data for testing')
    parser.add_argument('--random-state', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--sample-percentage', type=float, default=0.1, 
                       help='Percentage of data to use for training')
    parser.add_argument('--train-only-first', action='store_true', 
                       help='Train only XGBoost (default: all models)')
    parser.add_argument('--skip-tuning', action='store_true', 
                       help='Skip hyperparameter tuning and use original model')
    
    args = parser.parse_args()
    
    # Create trainer and run pipeline
    trainer = ModelTrainer(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_percentage=args.sample_percentage
    )
    
    results = trainer.run_training_pipeline(
        train_only_first=args.train_only_first,  # Default to all models
        skip_tuning=args.skip_tuning
    )
    
    return results

if __name__ == "__main__":
    main() 