# generate course embeddings here
 
import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

class CourseEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model"""
        pass
    
    def load_data(self):
        """Load OULAD data and compute course statistics"""
        pass
    
    def create_course_features(self):
        """Engineer features for each course"""
        pass
    
    def generate_text_embeddings(self):
        """Use sentence transformer for text"""
        pass
    
    def create_numerical_features(self):
        """Normalize numerical features"""
        pass
    
    def combine_embeddings(self):
        """Concatenate text + numerical"""
        pass
    
    def save_embeddings(self, filepath):
        """Cache embeddings to disk"""
        pass
    
    def load_embeddings(self, filepath):
        """Load cached embeddings"""
        pass