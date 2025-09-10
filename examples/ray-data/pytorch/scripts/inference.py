from actor import Actor
from argparse import ArgumentParser
import logging
import ray
import sagemaker_training.environment
import sys
import time
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from ray.data import ActorPoolStrategy


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
        "--batch_size", type=int, default=1000, help="Batch size for inference"
    )

    parser.add_argument(
        "--subset", type=int, default=None, help="Maximum number of examples"
    )

    # Parse only the arguments we care about and ignore the rest
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    return args


def preprocess(image_batch):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_tensor = torch.Tensor(image_batch["image"].transpose(0, 3, 1, 2))
    preprocessed_images = preprocess(torch_tensor).numpy()
    return {"image": preprocessed_images}


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    env = sagemaker_training.environment.Environment()

    start_time_without_metadata_fetching = time.time()
    num_gpus = int(ray.available_resources().get("GPU", 0))
    num_cpus = int(ray.available_resources().get("CPU", env.num_cpus))

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model_ref = ray.put(model)

    if args.subset is not None:
        logger.info(f"Loading {args.subset} images")

        ds = ray.data.read_parquet(
            "s3://air-example-data-2/10G-image-data-synthetic-raw-parquet/"
        ).limit(
            args.subset
        )  # Take only first 1000 images
    else:
        ds = ray.data.read_parquet(
            "s3://air-example-data-2/10G-image-data-synthetic-raw-parquet/"
        )

    ds = ds.map_batches(preprocess, batch_format="numpy")
    ds = ds.map_batches(
        Actor,
        batch_size=args.batch_size,
        compute=ActorPoolStrategy(size=num_cpus if num_gpus == 0 else num_gpus),
        num_gpus=1 if torch.cuda.is_available() and num_gpus > 0 else 0,
        batch_format="numpy",
        fn_constructor_kwargs={"model": model_ref},
        max_concurrency=2,
    )
    for _ in ds.iter_batches(batch_size=None, batch_format="pyarrow"):
        pass
    end_time = time.time()

    logger.info(f"Total time: {end_time - start_time}")
    logger.info(f"Throughput (img/sec): {ds.count()} / {(end_time - start_time)}")
    logger.info(
        f"Total time w/o metadata fetching (img/sec): {(end_time - start_time_without_metadata_fetching)}",
    )
    logger.info(
        f"Throughput w/o metadata fetching (img/sec) {ds.count()} / {(end_time - start_time_without_metadata_fetching)}",
    )

    logger.info(ds.stats())
