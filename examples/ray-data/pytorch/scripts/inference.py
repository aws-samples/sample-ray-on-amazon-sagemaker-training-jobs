"""
Example from https://docs.ray.io/en/latest/data/examples/pytorch_resnet_batch_prediction.html
"""

from argparse import ArgumentParser
import logging
from model import ResnetModel
import numpy as np
from PIL import Image
import ray
import sagemaker_training.environment
import sys
import time
from torchvision import transforms
from typing import Dict

S3_BUCKET_PATH = "s3://anonymous@air-example-data-2/imagenette2/train/"


# Configure logging to prevent duplicates in distributed environments
def setup_logging():
    """Set up logging configuration to prevent duplicate messages in Ray workers."""
    logger = logging.getLogger(__name__)

    # Prevent duplicate handlers in distributed environments
    if logger.handlers or hasattr(logger, "_configured"):
        return logger

    # Only configure logging once per process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],  # Only use stdout, not both stdout and stderr
        force=True,  # Override any existing configuration
    )

    # Mark as configured and prevent propagation to avoid duplicates
    logger._configured = True
    logger.propagate = False

    return logger


logger = setup_logging()


def parse_args():
    """Parse command line arguments for hyperparameters."""
    parser = ArgumentParser()

    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for inference"
    )

    parser.add_argument(
        "--subset", type=int, default=1000, help="Maximum number of examples"
    )

    # Parse only the arguments we care about and ignore the rest
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    return args


def preprocess_image(row: Dict[str, np.ndarray]):
    """Preprocess a single image."""

    # Define transform globally
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Convert numpy array to PIL Image
    image = Image.fromarray(row["image"])

    return {
        "transformed_image": transform(image),
    }


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    env = sagemaker_training.environment.Environment()

    num_gpus = int(ray.available_resources().get("GPU", 0))
    num_cpus = int(ray.available_resources().get("CPU", env.num_cpus))

    logger.info(f"Available resources: {num_cpus} CPUs, {num_gpus} GPUs")

    # Load dataset
    if args.subset is not None:
        logger.info(f"Loading {args.subset} images")
        ds = ray.data.read_images(S3_BUCKET_PATH, mode="RGB").limit(args.subset)
    else:
        ds = ray.data.read_images(S3_BUCKET_PATH, mode="RGB")

    # Preprocess images
    transformed_ds = ds.map(preprocess_image)

    # Configure inference based on available resources
    if num_gpus > 0:
        # GPU inference
        concurrency = min(num_gpus, 4)  # Use available GPUs, max 4
        num_gpus_per_actor = 1
        logger.info(f"Using GPU inference with {concurrency} actors")
    else:
        # CPU inference - reserve some CPUs for data loading
        concurrency = max(1, num_cpus - 2)
        num_gpus_per_actor = 0
        logger.info(f"Using CPU inference with {concurrency} actors")

    # Run batch inference
    predictions = transformed_ds.map_batches(
        ResnetModel,
        concurrency=concurrency,
        num_gpus=num_gpus_per_actor,
        batch_size=args.batch_size,
    )

    # Process results
    results = []
    for batch in predictions.iter_batches(batch_size=None):
        results.extend(batch["predictions"])
        if len(results) % 100 == 0:
            logger.info(f"Processed {len(results)} images")

    end_time = time.time()

    logger.info(f"Total images processed: {len(results)}")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Throughput: {len(results) / (end_time - start_time):.2f} images/sec")

    # Print dataset stats
    logger.info("Dataset statistics:")
    logger.info(predictions.stats())
