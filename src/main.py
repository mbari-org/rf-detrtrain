#!/usr/bin/env python
"""
DDP-enabled local training entrypoint for RF-DETR.
 
SageMaker directory structure:
- /opt/ml/input/data/training - Input training data
- /opt/ml/model - Output model location
- /opt/ml/output - Failure output
- /opt/ml/input/config/hyperparameters.json - Hyperparameters
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rfdetr import RFDETRLarge, RFDETRMedium
from src import __version__


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_logging():
    """Configure logging; only rank 0 emits INFO, others WARN to reduce noise."""
    log_level = logging.INFO if is_main_process() else logging.WARN
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("rfdetr-ddp")



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



def validate_dataset(dataset_dir: str):
    required_splits = ["train", "valid"]
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    for split in required_splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            raise ValueError(f"Missing required split directory: {split_dir}")
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            raise ValueError(f"Missing annotations file: {annotations_file}")


def init_distributed():
    """Initialize torch.distributed from environment variables."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def set_device_from_env():
    """Set CUDA device based on local rank for DDP."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        # Fallback to rank 0 GPU if not launched with torch.distributed.run/launch
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device

    local_rank = int(local_rank_str)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device


def build_model(model_size: str):
    if model_size.lower() == "large":
        return RFDETRLarge()
    if model_size.lower() == "medium":
        return RFDETRMedium()
    raise ValueError(f"Unknown model size: {model_size}")


def load_hyperparameters():
    """Load hyperparameters from SageMaker config if available."""
    hyperparameters_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, "r") as f:
            return json.load(f)
    return {}


def main():
    # Report how many GPUs are available
    if is_main_process():
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Initialize distributed (if launched with torch.distributed)
    init_distributed()

    # After initializing distributed, configure logging based on rank
    logger = setup_logging()

    # Report version and environment from rank 0
    if is_main_process():
        logger.info(f"RF-DETR training wrapper version: {__version__}")

    args = parse_args()

    # Determine device for this process
    device = set_device_from_env() if args.device == "cuda" else torch.device(args.device)

    hyperparameters = load_hyperparameters()
    logger.info(f"Loaded hyperparameters: {hyperparameters}")
    logger.info(f"Command line arguments: {args}")

    if is_main_process():
        logger.info(f"Using device: {device}")
        logger.info(f"Training data location: {args.train}")
        logger.info(f"Model directory: {args.model_dir}") 
        logger.info(f"Output data location: {args.output_data_dir}")

    # Validate dataset structure (all ranks do quick check to avoid hanging)
    validate_dataset(args.train)

    # Build model and move to device
    model = build_model(args.model_size) 

    # Prepare training kwargs; scale batch size per process if needed
    train_kwargs = {
        "dataset_dir": args.train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "output_dir": args.model_dir,
    }
    if args.learning_rate is not None:
        train_kwargs["learning_rate"] = args.learning_rate

    if is_main_process():
        logger.info(f"Starting training with parameters: {json.dumps(train_kwargs, indent=2)}")
 
    # Train the model
    try:
        model.train(**train_kwargs)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # Save the model to SageMaker's model directory
    logger.info(f"Model saved to {args.model_dir}")
    logger.info("Training job completed")

    # Clean up distributed
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


