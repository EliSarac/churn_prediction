# src/utils/data_loader.py
"""
Load OULAD datasets
Read CSVs and return DataFrames
"""

import pandas as pd
from pathlib import Path


class OULADDataLoader:
    """Load OULAD datasets from CSV files"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def load_all(self):
        """Load all OULAD datasets"""
        return {
            'courses': self.load_courses(),
            'student_info': self.load_student_info(),
            'student_vle': self.load_student_vle(),
            'student_assessment': self.load_student_assessment(),
            'assessments': self.load_assessments(),
            'student_registration': self.load_student_registration(),
            'vle': self.load_vle()
        }
    
    def load_courses(self):
        return pd.read_csv(self.data_dir / 'courses.csv')
    
    def load_student_info(self):
        return pd.read_csv(self.data_dir / 'studentInfo.csv')
    
    def load_student_vle(self):
        return pd.read_csv(self.data_dir / 'studentVle.csv')
    
    def load_student_assessment(self):
        return pd.read_csv(self.data_dir / 'studentAssessment.csv')
    
    def load_assessments(self):
        return pd.read_csv(self.data_dir / 'assessments.csv')
    
    def load_student_registration(self):
        return pd.read_csv(self.data_dir / 'studentRegistration.csv')
    
    def load_vle(self):
        return pd.read_csv(self.data_dir / 'vle.csv')