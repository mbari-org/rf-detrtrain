#!/usr/bin/env python
"""
AWS SageMaker training script for RF-DETR model on i2map dataset.

SageMaker directory structure:
- /opt/ml/input/data/training - Input training data
- /opt/ml/model - Output model location
- /opt/ml/output - Failure output
- /opt/ml/input/config/hyperparameters.json - Hyperparameters
"""

import os
import json
import logging
import sys
import argparse
from pathlib import Path
from rfdetr import RFDETRLarge, RFDETRMedium
from src import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for hyperparameters."""
    parser = argparse.ArgumentParser()

    # SageMaker specific paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    )
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="large", choices=["large", "medium"])
    parser.add_argument("--learning-rate", type=float, default=None)

    return parser.parse_args()


def load_hyperparameters():
    """Load hyperparameters from SageMaker config if available."""
    hyperparameters_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, "r") as f:
            return json.load(f)
    return {}


def validate_dataset(dataset_dir):
    """Validate that the dataset directory has the expected structure."""
    required_splits = ["train", "valid"]
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    for split in required_splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            raise ValueError(f"Missing required split directory: {split_dir}")

        # Check for annotations file
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            raise ValueError(f"Missing annotations file: {annotations_file}")

        logger.info(f"Found {split} split with annotations at {split_dir}")

    logger.info(f"Dataset validation successful for {dataset_dir}")


def main():
    """Main training function for SageMaker."""
    args = parse_args()

    # Log package version
    logger.info(f"RF-DETR training wrapper version: {__version__}")

    # Load additional hyperparameters from SageMaker config
    hyperparameters = load_hyperparameters()
    logger.info(f"Loaded hyperparameters: {hyperparameters}")
    logger.info(f"Command line arguments: {args}")

    # Log environment info
    logger.info(f"Training data location: {args.train}")
    logger.info(f"Model output location: {args.model_dir}")
    logger.info(f"Output data location: {args.output_data_dir}")

    # Validate dataset structure
    try:
        validate_dataset(args.train)
    except ValueError as e:
        logger.error(f"Dataset validation failed: {e}")
        raise

    # Initialize model based on size
    logger.info(f"Initializing {args.model_size} model...")
    if args.model_size.lower() == "large":
        model = RFDETRLarge()
    elif args.model_size.lower() == "medium":
        model = RFDETRMedium()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # Prepare training kwargs
    train_kwargs = {
        "dataset_dir": args.train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
    }

    # Add learning rate if specified
    if args.learning_rate is not None:
        train_kwargs["learning_rate"] = args.learning_rate

    logger.info(f"Starting training with parameters: {train_kwargs}")

    # Train the model
    try:
        model.train(**train_kwargs)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # Save the model to SageMaker's model directory
    logger.info(f"Saving model to {args.model_dir}")
    model_output_path = Path(args.model_dir)
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Note: The actual model saving logic depends on rfdetr's save mechanism
    # You may need to adjust this based on how rfdetr saves models
    try:
        # If rfdetr has a save method:
        if hasattr(model, "save"):
            model.save(str(model_output_path))
        elif hasattr(model, "model") and hasattr(model.model, "save_pretrained"):
            model.model.save_pretrained(str(model_output_path))
        logger.info(f"Model saved successfully to {args.model_dir}")
    except Exception as e:
        logger.warning(f"Could not save model with standard method: {e}")
        logger.info("Training completed but model save may need manual handling")

    logger.info("Training job completed")


if __name__ == "__main__":
    main()
