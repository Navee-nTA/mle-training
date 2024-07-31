import argparse
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from HousePricePrediction.logger import get_logger

import mlflow
import mlflow.sklearn


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url, housing_path, logger):
    """
    Fetch and extract housing data.

    Downloads the housing data from the provided URL, extracts it, and saves it
    into the specified directory.

    Parameters
    ----------
    housing_url : str
        URL to download the housing data from.
    housing_path : str
        Directory path to save the downloaded and extracted data.
    logger : logging.Logger
        Logger instance for logging the process.

    Returns
    -------
    None
    """
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info(f"Data downloaded and extracted to {housing_path}")


def load_housing_data(housing_path):
    """
    Load housing data from the specified directory.

    Parameters
    ----------
    housing_path : str
        Directory path to load the housing data from.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def main(args):
    """
    Main function to ingest housing data.

    This function handles the data ingestion process including downloading,
    extracting, loading, processing the housing data, and saving the processed
    data.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments passed to the script.

    Returns
    -------
    None
    """
    log_file = args.log_path if args.log_path.lower() != "none" else None
    logger = get_logger(
        "ingest", log_file, args.log_level, args.no_console_log
    )
    fetch_housing_data(HOUSING_URL, args.raw_data_path, logger)
    housing = load_housing_data(args.raw_data_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    housing.to_csv(
        os.path.join(args.processed_data_path, "housing.csv"),
        index=False,
    )
    logger.info(f"Data processed and saved to {args.processed_data_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw",
        help="Path to save raw data",
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="data/processed",
        help="Path to save processed data",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Specify log level",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="None",
        help="Use a log file or not, give path",
    )
    parser.add_argument(
        "--no_console_log",
        action="store_true",
        help="Not to write logs to console",
    )
    args = parser.parse_args()
    main(args)
    with mlflow.start_run():
        mlflow.log_artifact(args.processed_data_path)
