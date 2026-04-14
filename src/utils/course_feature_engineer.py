"""
Course Feature Engineer - Compute course-level features from OULAD data
Pure data transformation - takes raw DataFrames, returns feature DataFrame
"""

import pandas as pd
import numpy as np


class CourseFeatureEngineer:
    """
    Transform raw OULAD data into course feature vectors
    
    Input: Dictionary of raw DataFrames from OULADDataLoader
    Output: DataFrame with one row per course, computed features
    
    Single Responsibility: Feature engineering only (no data loading, no embeddings)
    """
    
    def __init__(self, config=None):
        """
        Initialize feature engineer
        
        Args:
            config: Optional config dict (for future extensibility)
        """
        self.config = config or {}
    
    def fit_transform(self, data):
        """
        Main method: Transform raw data to course features
        
        Args:
            data: Dictionary of DataFrames with keys:
                - 'courses': courses.csv
                - 'student_info': studentInfo.csv
                - 'student_vle': studentVle.csv
                - 'student_assessment': studentAssessment.csv
                - 'assessments': assessments.csv
                - 'student_registration': studentRegistration.csv
                
        Returns:
            DataFrame with columns:
                - code_module
                - code_presentation
                - duration_days
                - num_students
                - dropout_rate
                - avg_score
                - avg_clicks
                - num_assessments
                - num_resources
                - pct_female
                - avg_age
                - avg_dropout_day
        """
        print("Engineering course features...")
        
        courses = data['courses']
        student_info = data['student_info']
        student_vle = data['student_vle']
        student_assessment = data['student_assessment']
        assessments = data['assessments']
        student_registration = data['student_registration']
        
        course_features = []
        
        for _, course in courses.iterrows():
            features = self._compute_course_features(
                course,
                student_info,
                student_vle,
                student_assessment,
                assessments,
                student_registration
            )
            
            if features:
                course_features.append(features)
        
        df = pd.DataFrame(course_features)
        print(f"✓ Computed features for {len(df)} course presentations")
        
        return df
    
    def _compute_course_features(self, course, student_info, student_vle,
                                 student_assessment, assessments, student_registration):
        """
        Compute all features for a single course presentation
        
        Args:
            course: Single row from courses.csv
            [other args]: Full DataFrames for filtering
            
        Returns:
            Dictionary of features, or None if no students enrolled
        """
        code_module = course['code_module']
        code_presentation = course['code_presentation']
        
        # Filter students for this course
        course_students = student_info[
            (student_info['code_module'] == code_module) &
            (student_info['code_presentation'] == code_presentation)
        ]
        
        if len(course_students) == 0:
            return None
        
        # Initialize features
        features = {
            'code_module': code_module,
            'code_presentation': code_presentation,
            'duration_days': course['module_presentation_length'],
        }
        
        # Compute feature groups
        features.update(self._compute_student_demographics(course_students))
        features.update(self._compute_vle_features(code_module, code_presentation, student_vle))
        features.update(self._compute_assessment_features(
            code_module, code_presentation, student_assessment, assessments
        ))
        features.update(self._compute_dropout_timing(
            code_module, code_presentation, course_students, student_registration
        ))
        
        return features
    
    def _compute_student_demographics(self, course_students):
        """
        Compute demographic features from enrolled students
        
        Args:
            course_students: Filtered studentInfo DataFrame for this course
            
        Returns:
            Dictionary with: num_students, dropout_rate, pct_female, avg_age
        """
        return {
            'num_students': len(course_students),
            'dropout_rate': (course_students['final_result'] == 'Withdrawn').mean(),
            'pct_female': (course_students['gender'] == 'F').mean(),
            'avg_age': course_students['age_band'].apply(self._age_band_to_numeric).mean()
        }
    
    def _compute_vle_features(self, code_module, code_presentation, student_vle):
        """
        Compute VLE engagement features
        
        Args:
            code_module: Course code
            code_presentation: Presentation code
            student_vle: Full studentVle DataFrame
            
        Returns:
            Dictionary with: avg_clicks, num_resources
        """
        course_vle = student_vle[
            (student_vle['code_module'] == code_module) &
            (student_vle['code_presentation'] == code_presentation)
        ]
        
        if len(course_vle) == 0:
            return {
                'avg_clicks': 0,
                'num_resources': 0
            }
        
        # Average clicks per student
        avg_clicks = course_vle.groupby('id_student')['sum_click'].sum().mean()
        
        # Number of unique resources
        num_resources = course_vle['id_site'].nunique()
        
        return {
            'avg_clicks': avg_clicks,
            'num_resources': num_resources
        }
    
    def _compute_assessment_features(self, code_module, code_presentation,
                                    student_assessment, assessments):
        """
        Compute assessment performance features
        
        Args:
            code_module: Course code
            code_presentation: Presentation code
            student_assessment: Full studentAssessment DataFrame
            assessments: Full assessments DataFrame
            
        Returns:
            Dictionary with: num_assessments, avg_score
        """
        course_assessments = assessments[
            (assessments['code_module'] == code_module) &
            (assessments['code_presentation'] == code_presentation)
        ]
        
        num_assessments = len(course_assessments)
        
        if num_assessments == 0:
            return {
                'num_assessments': 0,
                'avg_score': 0
            }
        
        # Get student scores for this course's assessments
        course_scores = student_assessment.merge(
            course_assessments[['id_assessment']],
            on='id_assessment',
            how='inner'
        )
        
        avg_score = course_scores['score'].mean() if len(course_scores) > 0 else 0
        
        return {
            'num_assessments': num_assessments,
            'avg_score': avg_score
        }
    
    def _compute_dropout_timing(self, code_module, code_presentation,
                               course_students, student_registration):
        """
        Compute when students typically drop out
        
        Args:
            code_module: Course code
            code_presentation: Presentation code
            course_students: Filtered studentInfo for this course
            student_registration: Full studentRegistration DataFrame
            
        Returns:
            Dictionary with: avg_dropout_day
        """
        # Get registration data for this course
        reg_features = student_registration[
            (student_registration['code_module'] == code_module) &
            (student_registration['code_presentation'] == code_presentation)
        ]
        
        # Find withdrawn students
        withdrawn_students = course_students[
            course_students['final_result'] == 'Withdrawn'
        ]['id_student']
        
        if len(withdrawn_students) == 0:
            return {'avg_dropout_day': 0}
        
        # Get their unregistration dates
        withdrawn_reg = reg_features[
            reg_features['id_student'].isin(withdrawn_students)
        ]
        
        # date_unregistration = day when they dropped out
        dropout_days = withdrawn_reg['date_unregistration'].dropna()
        
        if len(dropout_days) == 0:
            return {'avg_dropout_day': 0}
        
        return {
            'avg_dropout_day': dropout_days.mean()
        }
    
    def _age_band_to_numeric(self, age_band):
        """
        Convert age band string to numeric midpoint
        
        Args:
            age_band: String ('0-35', '35-55', '55<=')
            
        Returns:
            Numeric age (midpoint of range)
        """
        mapping = {
            '0-35': 25,
            '35-55': 45,
            '55<=': 60
        }
        return mapping.get(age_band, 35)
    
    def get_feature_names(self):
        """
        Get list of feature names produced by this engineer
        
        Returns:
            List of feature column names
        """
        return [
            'code_module',
            'code_presentation',
            'duration_days',
            'num_students',
            'dropout_rate',
            'avg_score',
            'avg_clicks',
            'num_assessments',
            'num_resources',
            'pct_female',
            'avg_age',
            'avg_dropout_day'
        ]
    
    def get_numerical_feature_names(self):
        """
        Get list of numerical feature names (excluding identifiers)
        
        Returns:
            List of numerical feature names
        """
        return [
            'num_students',
            'dropout_rate',
            'avg_score',
            'avg_clicks',
            'duration_days',
            'num_assessments',
            'num_resources',
            'pct_female',
            'avg_age',
            'avg_dropout_day'
        ]