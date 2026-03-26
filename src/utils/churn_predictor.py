"""
Student Dropout Prediction - ChurnPredictor Class
Implements sklearn pipeline for churn prediction with SMOTE and feature engineering
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


class ChurnPredictor:
    """
    Student dropout prediction using Logistic Regression with SMOTE balancing
    """
    
    def __init__(self, config):
        """Initialize predictor with configuration"""
        self.config = config
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load raw OULAD data from CSV files"""
        print("Loading OULAD data from CSV files...")
        
        # Load all tables
        student_info = pd.read_csv('data/studentInfo.csv')
        student_vle = pd.read_csv('data/studentVle.csv')
        student_assessment = pd.read_csv('data/studentAssessment.csv')
        assessments = pd.read_csv('data/assessments.csv')
        student_registration = pd.read_csv('data/studentRegistration.csv')
        
        print(f"✓ Loaded {len(student_info)} students")
        
        return {
            'student_info': student_info,
            'student_vle': student_vle,
            'student_assessment': student_assessment,
            'assessments': assessments,
            'student_registration': student_registration
        }
    
    def build_features(self, data_dict, window_days):
        """
        Build features for prediction at specified time window
        
        Args:
            data_dict: Dictionary of OULAD tables
            window_days: Prediction window in days
            
        Returns:
            pd.DataFrame: Features with 'final_result' column
        """
        student_info = data_dict['student_info']
        student_vle = data_dict['student_vle']
        student_assessment = data_dict['student_assessment']
        assessments = data_dict['assessments']
        student_registration = data_dict['student_registration']
        
        print(f"Building features for day {window_days}...")
        
        # Start with student info (demographics)
        features = student_info.copy()
        
        # === VLE FEATURES ===
        # Filter VLE data up to prediction window
        vle_early = student_vle[student_vle['date'] <= window_days].copy()
        
        # Aggregate VLE interactions per student
        vle_features = vle_early.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'sum_click': ['sum', 'mean', 'std'],
            'date': ['min', 'max', 'nunique']
        }).reset_index()
        
        vle_features.columns = ['code_module', 'code_presentation', 'id_student',
                                'total_clicks', 'avg_clicks_per_day', 'std_clicks',
                                'first_access', 'last_access', 'days_active']
        
        # Calculate engagement metrics (with safe division)
        vle_features['days_since_last_access'] = window_days - vle_features['last_access']
        vle_features['engagement_span'] = vle_features['last_access'] - vle_features['first_access']
        
        # Safe engagement intensity calculation (avoid division by zero)
        vle_features['engagement_intensity'] = np.where(
            vle_features['days_active'] > 0,
            vle_features['total_clicks'] / vle_features['days_active'],
            0
        )
        
        # Merge VLE features
        features = features.merge(vle_features, 
                                on=['code_module', 'code_presentation', 'id_student'], 
                                how='left')
        
        # Fill missing VLE data (students with no clicks)
        vle_cols = ['total_clicks', 'avg_clicks_per_day', 'std_clicks', 'days_active',
                    'days_since_last_access', 'engagement_span', 'engagement_intensity',
                    'first_access', 'last_access']
        features[vle_cols] = features[vle_cols].fillna(0)
        
        # === ASSESSMENT FEATURES ===
        # Get assessments due within prediction window
        assessments_early = assessments[assessments['date'] <= window_days].copy()
        
        if len(assessments_early) > 0:
            # Get submissions for those assessments
            assessment_features = student_assessment.merge(
                assessments_early[['id_assessment', 'code_module', 'code_presentation', 'date']],
                on='id_assessment',
                how='inner'
            )
            
            if len(assessment_features) > 0:
                # Aggregate assessment metrics
                assessment_agg = assessment_features.groupby(
                    ['code_module', 'code_presentation', 'id_student']
                ).agg({
                    'score': ['mean', 'std', 'min', 'max', 'count'],
                    'date_submitted': 'max'
                }).reset_index()
                
                assessment_agg.columns = ['code_module', 'code_presentation', 'id_student',
                                        'avg_score', 'std_score', 'min_score', 'max_score',
                                        'num_assessments_submitted', 'last_submission_day']
                
                # Calculate submission timeliness
                assessment_features['days_early'] = assessment_features['date'] - assessment_features['date_submitted']
                late_submissions = assessment_features.groupby(
                    ['code_module', 'code_presentation', 'id_student']
                )['days_early'].apply(lambda x: (x < 0).sum()).reset_index()
                late_submissions.columns = ['code_module', 'code_presentation', 'id_student', 'num_late_submissions']
                
                assessment_agg = assessment_agg.merge(late_submissions, 
                                                    on=['code_module', 'code_presentation', 'id_student'],
                                                    how='left')
                
                # Merge assessment features
                features = features.merge(assessment_agg,
                                        on=['code_module', 'code_presentation', 'id_student'],
                                        how='left')
        
        # Fill missing assessment data
        assessment_cols = ['avg_score', 'std_score', 'min_score', 'max_score', 
                        'num_assessments_submitted', 'num_late_submissions', 'last_submission_day']
        for col in assessment_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)
        
        # === REGISTRATION FEATURES ===
        reg_features = student_registration[['code_module', 'code_presentation', 'id_student', 
                                            'date_registration', 'date_unregistration']].copy()
        
        features = features.merge(reg_features,
                                on=['code_module', 'code_presentation', 'id_student'],
                                how='left')
        
        features['date_registration'] = features['date_registration'].fillna(0)
        
        # === FINAL CLEANUP: Handle any remaining NaN/inf ===
        # Fill any remaining NaN with 0
        features = features.fillna(0)
        
        # Replace inf with 0 in numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        print(f"✓ Feature set built: {features.shape[0]} students, {features.shape[1]} features")
        
        return features
    
    def prepare_data(self, features):
        """Split data into train/test sets"""
        # Remove non-feature columns
        drop_cols = ['id_student', 'code_module', 'code_presentation', 'final_result', 
                     'date_unregistration', 'last_submission_day']
        feature_cols = [col for col in features.columns if col not in drop_cols]
        
        # Handle categorical variables
        categorical_cols = features[feature_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            features[col] = pd.Categorical(features[col]).codes
        
        X = features[feature_cols]
        y = (features['final_result'] == 'Withdrawn').astype(int)
        
        # Double-check for NaN/inf (should be clean from build_features, but be safe)
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values, filling with 0...")
            X = X.fillna(0)
        
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            print(f"⚠️  Found {inf_count} infinite values, replacing with 0...")
            X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_names = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y if self.config['data']['stratify'] else None
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Dropout rate: {y.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def build_pipeline(self):
        """Build sklearn pipeline with preprocessing and model"""
        steps = []
        
        # Scaler
        if self.config['preprocessing']['scaling']['enabled']:
            steps.append(('scaler', StandardScaler()))
        
        # SMOTE
        if self.config['preprocessing']['smote']['enabled']:
            smote = SMOTE(random_state=self.config['preprocessing']['smote']['random_state'])
            steps.append(('smote', smote))
        
        # Classifier
        model_config = self.config['model']['hyperparameters']
        classifier = LogisticRegression(
            C=model_config['C'],
            solver=model_config['solver'],
            max_iter=model_config['max_iter'],
            random_state=model_config['random_state']
        )
        steps.append(('classifier', classifier))
        
        pipeline = Pipeline(steps)
        
        print("\nPipeline:")
        for name, _ in steps:
            print(f"  → {name}")
        
        return pipeline
    
    def fit(self):
        """Train the model pipeline"""
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        data_dict = self.load_data()
        features = self.build_features(data_dict, self.config['preprocessing']['window_days'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data(features)
        
        self.pipeline = self.build_pipeline()
        
        print("\nTraining model...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("✓ Training complete")
        
    def predict(self, X, use_threshold=True):
        """Make predictions"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() or load() first.")
        
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        
        if use_threshold:
            threshold = self.config['model']['threshold']
            predictions = (probabilities >= threshold).astype(int)
        else:
            predictions = self.pipeline.predict(X)
        
        return predictions, probabilities
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        if self.X_test is None:
            raise ValueError("Test data not available. Run fit() first.")
        
        threshold = self.config['model']['threshold']
        y_pred, y_pred_proba = self.predict(self.X_test, use_threshold=True)
        
        metrics = self.compute_metrics(self.y_test, y_pred, y_pred_proba)
        
        print(f"\nThreshold: {threshold}")
        print("\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def save(self, filepath=None):
        """Save trained pipeline to disk"""
        if self.pipeline is None:
            raise ValueError("No model to save. Call fit() first.")
        
        if filepath is None:
            filepath = self.config['output']['model_path']
        
        joblib.dump(self.pipeline, filepath)
        print(f"\n✓ Model saved: {filepath}")
    
    def load(self, filepath=None):
        """Load trained pipeline from disk"""
        if filepath is None:
            filepath = self.config['output']['model_path']
        
        self.pipeline = joblib.load(filepath)
        print(f"✓ Model loaded: {filepath}")
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_pred_proba):
        """Compute classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=['Stayed', 'Dropped'],
                zero_division=0
            )
        }