"""
Embedding Store - FAISS index for vector storage and search
"""

import faiss
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Manage FAISS index for embedding storage and similarity search
    
    Responsibilities:
    - Build FAISS index from embeddings
    - Save/load index to/from disk
    - Search for similar vectors
    - Manage ID mappings
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.index = None
        self.metadata = None
        self.index_to_id = {}
        self.id_to_index = {}
    
    def build_index(self, embeddings, metadata):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array (n_items, dim)
            metadata: DataFrame with item information (must have code_module, code_presentation)
        """
        logger.info(f"Building FAISS index for {len(embeddings)} items")
        
        dimension = embeddings.shape[1]
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create flat index with inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = metadata.reset_index(drop=True)
        
        # Create ID mappings
        for i, row in self.metadata.iterrows():
            course_id = f"{row['code_module']}-{row['code_presentation']}"
            self.index_to_id[i] = course_id
            self.id_to_index[course_id] = i
        
        logger.info(f"✓ Built index: {self.index.ntotal} vectors, {dimension} dimensions")
    
    def search(self, query_vector, k=10):
        """
        Search for k most similar items
        
        Args:
            query_vector: numpy array (dim,)
            k: number of results to return
            
        Returns:
            tuple: (course_ids, scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Reshape and normalize query
        query = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't ask for more than we have
        scores, indices = self.index.search(query, k)
        
        # Map to course IDs
        course_ids = [self.index_to_id[idx] for idx in indices[0]]
        
        logger.debug(f"Search returned {len(course_ids)} results")
        
        return course_ids, scores[0]
    
    def get_course_info(self, course_id):
        """
        Get metadata for a specific course
        
        Args:
            course_id: String like 'AAA-2013J'
            
        Returns:
            Series with course information
        """
        if course_id not in self.id_to_index:
            raise ValueError(f"Course {course_id} not found in index")
        
        idx = self.id_to_index[course_id]
        return self.metadata.iloc[idx]
    
    def save(self, output_dir='embeddings'):
        """
        Save index and metadata to disk
        
        Args:
            output_dir: Directory to save files
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_path / 'course_index.faiss'
        faiss.write_index(self.index, str(index_path))
        logger.info(f"✓ Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = output_path / 'course_metadata.parquet'
        self.metadata.to_parquet(metadata_path, compression='snappy')
        logger.info(f"✓ Saved metadata to {metadata_path}")
        
        # Save ID mappings
        mapping_path = output_path / 'id_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump({
                'index_to_id': self.index_to_id,
                'id_to_index': self.id_to_index
            }, f, indent=2)
        logger.info(f"✓ Saved ID mappings to {mapping_path}")
    
    def load(self, output_dir='embeddings'):
        """
        Load index and metadata from disk
        
        Args:
            output_dir: Directory containing saved files
        """
        output_path = Path(output_dir)
        
        # Load FAISS index
        index_path = output_path / 'course_index.faiss'
        self.index = faiss.read_index(str(index_path))
        logger.info(f"✓ Loaded FAISS index from {index_path}")
        
        # Load metadata
        metadata_path = output_path / 'course_metadata.parquet'
        self.metadata = pd.read_parquet(metadata_path)
        logger.info(f"✓ Loaded metadata from {metadata_path}")
        
        # Load ID mappings
        mapping_path = output_path / 'id_mapping.json'
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
            # Convert string keys back to integers for index_to_id
            self.index_to_id = {int(k): v for k, v in mappings['index_to_id'].items()}
            self.id_to_index = mappings['id_to_index']
        
        logger.info(f"✓ Loaded {self.index.ntotal} vectors")