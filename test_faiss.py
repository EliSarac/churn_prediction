"""
Test FAISS Embedding Store - Vector similarity search
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.embedding_store import EmbeddingStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_faiss_search():
    """Test FAISS index building and similarity search"""
    
    logger.info("="*80)
    logger.info("TESTING FAISS EMBEDDING STORE")
    logger.info("="*80)
    
    # Load embeddings and features from pipeline output
    logger.info("\n[STEP 1] Loading embeddings and features")
    embeddings = np.load('embeddings/course_embeddings.npy')
    course_features = pd.read_csv('embeddings/course_features.csv')
    
    logger.info(f"✓ Loaded embeddings: {embeddings.shape}")
    logger.info(f"✓ Loaded features: {course_features.shape}")
    
    # Build FAISS index
    logger.info("\n[STEP 2] Building FAISS index")
    store = EmbeddingStore()
    store.build_index(embeddings, course_features)
    
    # Save index
    logger.info("\n[STEP 3] Saving index to disk")
    store.save('embeddings')
    
    # Test search
    logger.info("\n[STEP 4] Testing similarity search")
    
    # Pick a test course
    test_course = course_features.iloc[0]
    test_id = f"{test_course['code_module']}-{test_course['code_presentation']}"
    test_embedding = embeddings[0]
    
    logger.info(f"\nQuery course: {test_id}")
    logger.info(f"  Students: {int(test_course['num_students'])}")
    logger.info(f"  Dropout rate: {test_course['dropout_rate']:.2%}")
    logger.info(f"  Avg dropout day: {test_course['avg_dropout_day']:.1f}")
    logger.info(f"  Avg score: {test_course['avg_score']:.1f}")
    
    # Search for similar courses
    similar_ids, scores = store.search(test_embedding, k=5)
    
    logger.info(f"\nTop 5 most similar courses:")
    for i, (course_id, score) in enumerate(zip(similar_ids, scores), 1):
        course_info = store.get_course_info(course_id)
        
        logger.info(f"\n{i}. {course_id} (similarity: {score:.4f})")
        logger.info(f"   Students: {int(course_info['num_students'])}")
        logger.info(f"   Dropout rate: {course_info['dropout_rate']:.2%}")
        logger.info(f"   Avg dropout day: {course_info['avg_dropout_day']:.1f}")
        logger.info(f"   Avg score: {course_info['avg_score']:.1f}")
    
    # Test loading from disk
    logger.info("\n[STEP 5] Testing load from disk")
    store2 = EmbeddingStore()
    store2.load('embeddings')
    
    # Verify same results
    similar_ids2, scores2 = store2.search(test_embedding, k=5)
    
    if similar_ids == similar_ids2:
        logger.info("✓ Load/save verified - identical results")
    else:
        logger.error("✗ Load/save mismatch!")
    
    # Test different queries
    logger.info("\n[STEP 6] Testing multiple queries")
    
    # Find courses with high dropout
    high_dropout = course_features[course_features['dropout_rate'] > 0.3]
    logger.info(f"\nCourses with >30% dropout: {len(high_dropout)}")
    
    if len(high_dropout) > 0:
        test_idx = high_dropout.index[0]
        test_id = f"{high_dropout.iloc[0]['code_module']}-{high_dropout.iloc[0]['code_presentation']}"
        
        logger.info(f"\nQuery: {test_id} (high dropout course)")
        logger.info(f"  Dropout rate: {high_dropout.iloc[0]['dropout_rate']:.2%}")
        
        similar_ids, scores = store.search(embeddings[test_idx], k=3)
        
        logger.info(f"\nSimilar high-dropout courses:")
        for i, (course_id, score) in enumerate(zip(similar_ids, scores), 1):
            course_info = store.get_course_info(course_id)
            logger.info(f"{i}. {course_id}: {course_info['dropout_rate']:.2%} dropout (score: {score:.4f})")
    
    # Success
    logger.info("\n" + "="*80)
    logger.info("✓ FAISS TEST SUCCESSFUL")
    logger.info("="*80)
    logger.info("\nFiles saved:")
    logger.info("  - embeddings/course_index.faiss")
    logger.info("  - embeddings/course_metadata.parquet")
    logger.info("  - embeddings/id_mapping.json")
    
    return True


if __name__ == '__main__':
    try:
        test_faiss_search()
    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)