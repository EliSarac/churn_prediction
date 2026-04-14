"""
Test Course Embedding Pipeline
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import OULADDataLoader
from src.utils.course_feature_engineer import CourseFeatureEngineer
from src.utils.course_embedder import CourseEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the complete pipeline: Data → Features → Embeddings"""
    
    logger.info("="*80)
    logger.info("TESTING COURSE EMBEDDING PIPELINE")
    logger.info("="*80)
    
    try:
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading OULAD data")
        loader = OULADDataLoader(data_dir='data')
        data = loader.load_all()
        
        logger.info(f"✓ Loaded {len(data)} datasets:")
        for name, df in data.items():
            logger.info(f"  - {name}: {len(df):,} rows")
        
        # Step 2: Engineer features
        logger.info("\n[STEP 2] Engineering course features")
        engineer = CourseFeatureEngineer()
        course_features = engineer.fit_transform(data)
        
        logger.info(f"✓ Generated features: {course_features.shape}")
        logger.info(f"  Columns: {list(course_features.columns)}")
        
        # Display sample
        logger.info("\n  Sample course features:")
        sample = course_features[['code_module', 'code_presentation', 'num_students', 
                                  'dropout_rate', 'avg_dropout_day']].head(5)
        print(sample.to_string(index=False))
        
        # Verify features
        logger.info("\n  Feature statistics:")
        numerical_features = course_features.select_dtypes(include=['float64', 'int64'])
        print(numerical_features.describe().round(2))
        
        # Step 3: Generate embeddings
        logger.info("\n[STEP 3] Generating embeddings")
        embedder = CourseEmbedder(config_path='config/config_recommender.yaml')
        embeddings = embedder.fit_transform(course_features)
        
        logger.info(f"✓ Generated embeddings: {embeddings.shape}")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  Expected dimension: 394 (384 text + 10 numerical)")
        
        # Verify embeddings
        logger.info("\n  Embedding statistics:")
        logger.info(f"  - Min value: {embeddings.min():.4f}")
        logger.info(f"  - Max value: {embeddings.max():.4f}")
        logger.info(f"  - Mean value: {embeddings.mean():.4f}")
        logger.info(f"  - Std dev: {embeddings.std():.4f}")
        
        # Test get_embedding
        logger.info("\n[STEP 4] Testing get_embedding method")
        test_course = course_features.iloc[0]
        code_module = test_course['code_module']
        code_presentation = test_course['code_presentation']
        
        embedding = embedder.get_embedding(code_module, code_presentation)
        logger.info(f"✓ Retrieved embedding for {code_module}-{code_presentation}")
        logger.info(f"  Embedding shape: {embedding.shape}")
        logger.info(f"  First 5 values: {embedding[:5]}")
        
        # Save outputs
        logger.info("\n[STEP 5] Saving outputs")
        output_dir = Path('embeddings')
        output_dir.mkdir(exist_ok=True)
        
        # Save features
        features_path = output_dir / 'course_features.csv'
        course_features.to_csv(features_path, index=False)
        logger.info(f"✓ Saved features to {features_path}")
        
        # Save embeddings
        import numpy as np
        embeddings_path = output_dir / 'course_embeddings.npy'
        np.save(embeddings_path, embeddings)
        logger.info(f"✓ Saved embeddings to {embeddings_path}")
        
        # Success summary
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE TEST SUCCESSFUL")
        logger.info("="*80)
        logger.info(f"\nGenerated:")
        logger.info(f"  - {len(course_features)} course feature vectors")
        logger.info(f"  - {len(embeddings)} course embeddings (394-dim)")
        logger.info(f"\nOutputs saved to: {output_dir}/")
        
        return True
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("✗ PIPELINE TEST FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
