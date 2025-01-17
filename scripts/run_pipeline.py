import argparse
import logging
import sys
import os

from pipelines.training_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline")

    parser.add_argument("--data_path", type=str, default="data/raw/olist_customers_dataset.csv", help="Path to dataset")
    # parser.add_argument("--model_path", type=str, default="artifacts/checkpoints/model.pkl", help="Path to save/load the model")
    # parser.add_argument("--train", action="store_true", help="Flag to train the model")
    # parser.add_argument("--evaluate", action="store_true", help="Flag to evaluate the model")

    args = parser.parse_args()

    train_pipeline(args.data_path)
