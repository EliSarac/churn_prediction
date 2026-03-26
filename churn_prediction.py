
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.churn_predictor import ChurnPredictor


def load_config(config_path='config/config_churn.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments (key=value format)"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args


def main():
    """Main execution"""
    args = parse_args()
    config = load_config()
    
    step = args.get('step', config.get('step', 'train'))
    
    if step not in ['train', 'test']:
        print(f"Error: Invalid step '{step}'. Use 'train' or 'test'")
        sys.exit(1)
    
    predictor = ChurnPredictor(config)
    
    if step == 'train':
        print("\n" + "="*80)
        print("STEP: TRAIN")
        print("="*80)
        
        predictor.fit()
        predictor.evaluate()
        predictor.save()
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        
    elif step == 'test':
        print("\n" + "="*80)
        print("STEP: TEST")
        print("="*80)
        
        model_path = config['output']['model_path']
        
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            print("Run training first: uv run churn_prediction.py step=train")
            sys.exit(1)
        
        predictor.load()
        
        # Load data for testing
        dataset = predictor.load_data()
        features = predictor.build_features(dataset, config['preprocessing']['window_days'])
        predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = predictor.prepare_data(features)
        
        predictor.evaluate()
        
        print("\n" + "="*80)
        print("✓ TESTING COMPLETE")
        print("="*80)


if __name__ == '__main__':
    main()
