"""
Command-line interface for spam detection predictions
"""
import argparse
import sys
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.predict import SpamPredictor


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Hindi Spam Detection - Predict if messages are spam or ham'
    )

    parser.add_argument(
        'message',
        type=str,
        nargs='?',
        help='Message to classify (if not provided, will use interactive mode)'
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Path to CSV file with messages to classify'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions (for file mode)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Setup logger
    log_config = config.get('logging', {})
    logger = setup_logger(
        'spam_detection_cli',
        log_file=log_config.get('log_file'),
        level=log_config.get('level', 'INFO')
    )

    # Initialize predictor
    predictor = SpamPredictor(config)

    try:
        predictor.load_artifacts()
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        print(f"\nError: {e}")
        print("Please train the model first using: python main.py --train")
        sys.exit(1)

    # File mode
    if args.file:
        logger.info(f"Processing file: {args.file}")
        print(f"\nProcessing file: {args.file}")

        try:
            results_df = predictor.predict_from_file(args.file)

            # Save results if output path provided
            if args.output:
                results_df.to_csv(args.output, index=False, encoding='utf-8')
                print(f"Predictions saved to: {args.output}")
            else:
                # Print results
                print("\nPredictions:")
                print("-" * 80)
                for idx, row in results_df.iterrows():
                    print(f"{idx+1}. {row['prediction']}: {row['Message'][:60]}...")
                    if 'confidence' in row:
                        print(f"   Confidence: {row['confidence']:.2%}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            print(f"Error: {e}")
            sys.exit(1)

    # Single message mode
    elif args.message:
        result = predictor.predict_single(args.message)

        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"Message: {result['message']}")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.2%}")
        print("="*80)

    # Interactive mode
    else:
        print("\n" + "="*80)
        print("HINDI SPAM DETECTION - Interactive Mode")
        print("="*80)
        print("Enter messages to classify (type 'quit' or 'exit' to stop)")
        print("-"*80)

        while True:
            try:
                message = input("\nEnter message: ").strip()

                if message.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not message:
                    print("Please enter a message.")
                    continue

                result = predictor.predict_single(message)

                print(f"\n  Prediction: {result['prediction']}")
                if result['confidence']:
                    print(f"  Confidence: {result['confidence']:.2%}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


if __name__ == '__main__':
    main()
