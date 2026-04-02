# src/utils/course_feature_engineer.py
"""
Compute course-level features from OULAD data
"""

import pandas as pd
import numpy as np


class CourseFeatureEngineer:
    """
    Transform raw OULAD data into course feature vectors
    
    Input: Raw DataFrames
    Output: DataFrame with one row per course, computed features
    """
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def fit_transform(self, data):

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
        return df
    
    def _compute_course_features(self, course, student_info, student_vle,
                                 student_assessment, assessments, student_registration):
        """Compute all features for a single course"""
        code_module = course['code_module']
        code_presentation = course['code_presentation']
        
        # Filter students for this course
        course_students = student_info[
            (student_info['code_module'] == code_module) &
            (student_info['code_presentation'] == code_presentation)
        ]
        
        if len(course_students) == 0:
            return None
        
        features = {
            'code_module': code_module,
            'code_presentation': code_presentation,
            'duration_days': course['module_presentation_length'],
        }
        
        # Student demographics
        features.update(self._compute_student_features(course_students))
        
        # VLE engagement
        features.update(self._compute_vle_features(
            code_module, code_presentation, student_vle
        ))
        
        # Assessment performance
        features.update(self._compute_assessment_features(
            code_module, code_presentation, student_assessment, assessments
        ))
        
        # Derived features
        features['difficulty_score'] = self._calculate_difficulty(
            features['dropout_rate'],
            features['avg_score']
        )
        
        return features
    
    def _compute_student_features(self, course_students):
        """Compute demographics from student data"""
        return {
            'num_students': len(course_students),
            'dropout_rate': (course_students['final_result'] == 'Withdrawn').mean(),
            'pct_female': (course_students['gender'] == 'F').mean(),
            'avg_age': course_students['age_band'].apply(self._age_band_to_numeric).mean()
        }
    
    def _compute_vle_features(self, code_module, code_presentation, student_vle):
        """Compute VLE engagement features"""
        course_vle = student_vle[
            (student_vle['code_module'] == code_module) &
            (student_vle['code_presentation'] == code_presentation)
        ]
        
        if len(course_vle) == 0:
            return {'avg_clicks': 0, 'num_resources': 0}
        
        return {
            'avg_clicks': course_vle.groupby('id_student')['sum_click'].sum().mean(),
            'num_resources': course_vle['id_site'].nunique()
        }
    
    def _compute_assessment_features(self, code_module, code_presentation,
                                    student_assessment, assessments):
        """Compute assessment features"""
        course_assessments = assessments[
            (assessments['code_module'] == code_module) &
            (assessments['code_presentation'] == code_presentation)
        ]
        
        num_assessments = len(course_assessments)
        
        if num_assessments == 0:
            return {'num_assessments': 0, 'avg_score': 0}
        
        # Get scores
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
    
    def _age_band_to_numeric(self, age_band):
        """Convert age band to numeric"""
        mapping = {'0-35': 25, '35-55': 45, '55<=': 60}
        return mapping.get(age_band, 35)
    
