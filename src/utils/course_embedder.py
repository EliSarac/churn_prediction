"""
Course Embedder - Generate embeddings from course features
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class CourseEmbedder:
    """Generate course embeddings from feature DataFrame"""
    
    def __init__(self, config_path='config/config_recommender.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.embeddings = None
        self.course_features = None
    
    def fit_transform(self, course_features):
        """Generate embeddings from course features DataFrame"""
        logger.info("Generating course embeddings")
        
        self.course_features = course_features
        
        text_embeddings = self._generate_text_embeddings(course_features)
        numerical_features = self._create_numerical_features(course_features)
        self.embeddings = self._combine_embeddings(text_embeddings, numerical_features)
        
        logger.info(f"Generated embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def _generate_text_embeddings(self, course_features):
        """Generate text embeddings using Sentence Transformer"""
        logger.debug("Generating text embeddings")
        
        if self.model is None:
            model_name = self.config['embeddings']['sentence_transformer']['model_name']
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
        
        texts = [self._create_text_description(row) for _, row in course_features.iterrows()]
        
        batch_size = self.config['embeddings']['sentence_transformer']['batch_size']
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.debug(f"Text embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _create_text_description(self, course_row):
        """Create text description from course features"""
        code = course_row['code_module']
        semester = course_row['code_presentation']
        num_students = int(course_row['num_students'])
        duration = int(course_row['duration_days'])
        
        text = f"Course {code} offered in semester {semester}. "
        text += f"Enrolled {num_students} students. "
        text += f"Duration {duration} days."
        
        return text
    
    def _create_numerical_features(self, course_features):
        """Extract and normalize numerical features"""
        logger.debug("Normalizing numerical features")
        
        feature_names = self.config['course_features']['include']
        numerical_data = course_features[feature_names].values
        normalized = self.scaler.fit_transform(numerical_data)
        
        logger.debug(f"Numerical features shape: {normalized.shape}")
        return normalized
    
    def _combine_embeddings(self, text_embeddings, numerical_features):
        """Concatenate text embeddings and numerical features"""
        logger.debug("Combining embeddings")
        
        combined = np.hstack([text_embeddings, numerical_features])
        
        expected_dim = self.config['embeddings']['total_dim']
        actual_dim = combined.shape[1]
        
        if actual_dim != expected_dim:
            logger.warning(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        return combined
    
    def get_embedding(self, code_module, code_presentation):
        """Get embedding for a specific course"""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call fit_transform() first.")
        
        mask = (
            (self.course_features['code_module'] == code_module) &
            (self.course_features['code_presentation'] == code_presentation)
        )
        
        indices = self.course_features[mask].index
        
        if len(indices) == 0:
            raise ValueError(f"Course {code_module}-{code_presentation} not found")
        
        return self.embeddings[indices[0]]