from argparse import ArgumentParser, Namespace
import csv
import emoji
import json
import logging
from model import MulticlassClassifier
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sys
from time import gmtime, strftime
import traceback
import ray
import torch


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

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH_INPUT = os.path.join(BASE_PATH, "input", "data", "processing")
PROCESSING_PATH_OUTPUT = os.path.join(BASE_PATH, "output", "data")


"""
    Read hyperparameters
"""


def __read_params():
    try:
        parser = ArgumentParser()

        parser.add_argument("--epochs", type=int, default=25)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--dataset_percentage", type=str, default=100)
        parser.add_argument(
            "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
        )
        parser.add_argument(
            "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
        )
        parser.add_argument(
            "--test", type=str, default=os.environ.get("SM_CHANNEL_TEST")
        )
        parser.add_argument(
            "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR")
        )

        # Parse only the arguments we care about and ignore the rest
        args, unknown = parser.parse_known_args()

        if unknown:
            logger.info(f"Ignoring unknown arguments: {unknown}")

        if len(vars(args)) == 0:
            with open(
                os.path.join(
                    "/", "opt", "ml", "input", "config", "hyperparameters.json"
                ),
                "r",
            ) as f:
                training_params = json.load(f)

            args = Namespace(**training_params)

        return args
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e


@ray.remote(num_cpus=2)
def clean_text_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Ray remote function to clean text data for a chunk of the dataframe
    """

    def clean_single_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.lstrip().rstrip()
        text = re.sub("\[.*?\]", "", text)
        text = re.sub("https?://\S+|www\.\S+", "", text)
        text = re.sub("\n", "", text)
        text = " ".join(filter(lambda x: x[0] != "@", text.split()))
        text = emoji.replace_emoji(text, "")
        text = text.replace("u'", "'")
        text = text.encode("ascii", "ignore").decode()

        word_list = text.split(" ")
        for word in word_list:
            if isinstance(word, bytes):
                word = word.decode("utf-8")

        text = " ".join(word_list)

        if not any(c.isalpha() for c in text):
            return ""
        else:
            return text

    # Apply cleaning to the text column
    df_chunk["text"] = df_chunk["text"].apply(clean_single_text)
    return df_chunk


@ray.remote(num_cpus=1)
def process_sentiment_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Ray remote function to process sentiment mapping for a chunk
    """
    # Clean sentiment column
    df_chunk["Sentiment"] = df_chunk["Sentiment"].map(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    df_chunk["Sentiment"] = df_chunk["Sentiment"].replace("", np.nan)
    df_chunk["Sentiment"] = df_chunk["Sentiment"].replace(" ", np.nan)

    # Map sentiment labels
    df_chunk["Sentiment"] = df_chunk["Sentiment"].map(
        {"Negative": 0, "Neutral": 1, "Positive": 2}
    )

    return df_chunk


@ray.remote(num_cpus=1)
def transform_data(df):
    """
    Transform data using Ray for distributed processing
    """
    try:
        # Select relevant columns
        df = df[["text", "Sentiment"]]
        logger.info("Original count: {}".format(len(df.index)))

        # Drop initial NaN values
        df = df.dropna()

        # Calculate optimal chunk size based on available CPUs
        available_cpus = int(ray.cluster_resources().get("CPU", 4))
        chunk_size = max(1000, len(df) // available_cpus)

        logger.info(f"Using {available_cpus} CPUs with chunk size {chunk_size}")

        # Split dataframe into chunks
        chunks = [
            df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)
        ]
        logger.info(f"Processing {len(chunks)} chunks in parallel")

        # Process text cleaning in parallel
        text_cleaning_futures = [clean_text_chunk.remote(chunk) for chunk in chunks]
        cleaned_chunks = ray.get(text_cleaning_futures)

        # Combine cleaned chunks
        df = pd.concat(cleaned_chunks, ignore_index=True)

        # Clean text column further
        df["text"] = df["text"].map(lambda x: x.strip() if isinstance(x, str) else x)
        df["text"] = df["text"].replace("", np.nan)
        df["text"] = df["text"].replace(" ", np.nan)

        # Process sentiment mapping in parallel
        sentiment_chunks = [
            df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)
        ]
        sentiment_futures = [
            process_sentiment_chunk.remote(chunk) for chunk in sentiment_chunks
        ]
        processed_chunks = ray.get(sentiment_futures)

        # Combine processed chunks
        df = pd.concat(processed_chunks, ignore_index=True)

        # Final cleanup
        df = df.dropna()
        df = df.rename(columns={"Sentiment": "labels"})
        df = df[["text", "labels"]]

        logger.info("Final count: {}".format(len(df.index)))
        return df

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))
        raise e


@ray.remote(num_cpus=2)
def train_func(args, train, test):
    X_train, y_train = train["text"], train["labels"].values
    X_test, y_test = test["text"], test["labels"].values

    # Create a bag-of-words vectorizer
    vectorizer = CountVectorizer()

    # Fit the vectorizer to your dataset
    vectorizer.fit(X_train)

    # Convert the input strings to numerical representations
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    X_train, y_train = torch.from_numpy(X_train).type(torch.float32), torch.from_numpy(
        y_train
    )
    X_test, y_test = torch.from_numpy(X_test).type(torch.float32), torch.from_numpy(
        y_test
    )

    model = MulticlassClassifier(X_train.shape[1], 20, 3)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    for epoch in range(args.epochs):
        logger.info("Epoch: {}".format(epoch))
        # Forward pass: compute predicted y by passing x to the model
        y_pred = model(X_train)

        # Compute and print loss
        loss = criterion(y_pred, y_train)
        logger.info(f"Training Loss: {loss:.4f}")

        # Compute the accuracy
        _, predicted = torch.max(y_pred, dim=1)
        correct = (predicted == y_train).sum().item()
        accuracy = correct / len(y_train)
        logger.info(f"Training Accuracy: {accuracy:.4f}")

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    with torch.no_grad():
        y_pred = model(X_test)
        # Compute and print loss
        loss = criterion(y_pred, y_test)
        logger.info(f"Evaluation Loss: {loss:.4f}")

        # Compute the accuracy
        _, predicted = torch.max(y_pred, dim=1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / len(y_test)
        logger.info(f"Evaluation Accuracy: {accuracy:.4f}")

    logger.info("Save model in {}".format(args.model_dir))

    torch.save(model.state_dict(), "{}/model.pth".format(args.model_dir))

    logger.info("Save vectorizer in {}".format(args.model_dir))

    with open("{}/vectorizer.pkl".format(args.model_dir), "wb") as f:
        pickle.dump(vectorizer, f)


def extract_data(file_path, percentage=100):
    """
    Extract data from CSV files
    """
    try:
        files = [
            f
            for f in listdir(file_path)
            if isfile(join(file_path, f)) and f.endswith(".csv")
        ]
        logger.info("Files found: {}".format(files))

        frames = []
        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                encoding="utf-8",
                on_bad_lines="skip",
            )

            df = df.head(int(len(df) * (percentage / 100)))
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Total rows loaded: {len(df)}")
        return df

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))
        raise e


if __name__ == "__main__":
    try:
        args = __read_params()

        logger.info(f"Training arguments: {args}")

        # Log Ray cluster information
        cluster_resources = ray.cluster_resources()
        cluster_nodes = ray.nodes()

        logger.info(f"Ray cluster resources: {cluster_resources}")
        logger.info(f"Available CPUs: {cluster_resources.get('CPU', 'Unknown')}")
        logger.info(f"Available GPUs: {cluster_resources.get('GPU', 'Unknown')}")
        logger.info(
            f"Available Memory: {cluster_resources.get('memory', 'Unknown')} bytes"
        )

        # Extract data
        df = extract_data(PROCESSING_PATH_INPUT, 100)

        # Transform data with Ray parallelization
        df = ray.get(transform_data.remote(df))

        # Split data with fixed random state for reproducibility
        data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)

        ray.get(train_func.remote(args, data_train, data_test))

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise e
