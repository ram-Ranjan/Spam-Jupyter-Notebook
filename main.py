"""
Main entry point for Hindi Spam Detection project
"""
import argparse
import sys
from pathlib import Path
import joblib

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


def train_pipeline(config: dict, logger):
    """
    Complete training pipeline

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*80)

    # 1. Load data
    logger.info("\n[1/7] Loading data...")
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    df = data_loader.prepare_target(df)

    summary = data_loader.get_data_summary(df)
    logger.info(f"Data summary: {summary}")

    # 2. Feature engineering
    logger.info("\n[2/7] Extracting features...")
    feature_engineer = FeatureEngineer(config)
    features_df = feature_engineer.extract_all_features(df, fit_tfidf=True)

    # Save processed data
    processed_path = config['data']['processed_path']
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    result_df = df[['Message']].copy()
    result_df = result_df.join(features_df)
    result_df['is_spam'] = df['is_spam']
    result_df.to_csv(processed_path, index=False, encoding='utf-8')
    logger.info(f"Processed data saved to {processed_path}")

    # 3. Split data
    logger.info("\n[3/7] Splitting data...")
    X = features_df
    y = df['is_spam']

    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # 4. Preprocess data
    logger.info("\n[4/7] Preprocessing data...")
    X_train_processed, y_train_processed, X_test_processed = trainer.preprocess_data(
        X_train, y_train, X_test
    )

    # 5. Train model
    logger.info("\n[5/7] Training model...")
    model = trainer.train(X_train_processed, y_train_processed)

    # 6. Evaluate model
    logger.info("\n[6/7] Evaluating model...")
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate_model(
        model,
        X_test_processed,
        y_test,
        feature_names=feature_engineer.get_feature_names()
    )

    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    for metric, value in metrics.items():
        print(f"{metric.upper():20s}: {value:.4f} ({value*100:.2f}%)")
    print("="*80)

    # 7. Save model and artifacts
    logger.info("\n[7/7] Saving model and artifacts...")
    trainer.save_model()

    # Save feature engineer artifacts
    model_save_config = config.get('model_save', {})
    vectorizer_path = model_save_config.get('vectorizer_path', 'models/tfidf_vectorizer.pkl')
    feature_names_path = model_save_config.get('feature_names_path', 'models/feature_names.pkl')

    joblib.dump(feature_engineer.vectorizer, vectorizer_path)
    logger.info(f"Vectorizer saved to {vectorizer_path}")

    joblib.dump(feature_engineer.get_feature_names(), feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("="*80)

    print(f"\nModel saved to: {model_save_config.get('model_path')}")
    print(f"Reports saved to: {config['evaluation']['report_path']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Hindi Spam Detection - Train and evaluate model'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data CSV file (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Override data path if provided
    if args.data:
        config['data']['raw_path'] = args.data

    # Setup logger
    log_config = config.get('logging', {})
    logger = setup_logger(
        'spam_detection',
        log_file=log_config.get('log_file'),
        level=log_config.get('level', 'INFO')
    )

    # Train model
    if args.train:
        try:
            train_pipeline(config, logger)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            print(f"\nError: Training failed - {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Train model:        python main.py --train")
        print("  Custom config:      python main.py --train --config my_config.yaml")
        print("  Custom data:        python main.py --train --data my_data.csv")
        print("\nFor predictions, use: python predict_cli.py")


if __name__ == '__main__':
    main()
