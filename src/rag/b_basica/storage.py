"""
Created by Datoscout at 27/03/2024
solutions@datoscout.ec

This module provides functions to upload and download files (specifically pandas DataFrames) to and from AWS S3 or local storage, depending on the environment configuration. It abstracts the storage backend, allowing seamless switching between cloud and local storage for data persistence.
"""

# Standard imports
import os
import pickle
from io import BytesIO

# Third party imports
# import boto3
import pandas as pd

# External imports
from loguru import logger

# Internal imports
from src.config.settings import (
    AWS_ACCESS_KEY_ID,
    AWS_FOLDER,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    DATA_PATH,
    IS_AWS_AVAILABLE,
    S3_BUCKET_NAME,
)

# from botocore.exceptions import ClientError




def upload_file(file_name: str, data: pd.DataFrame) -> bool:
    """
    Uploads a pandas DataFrame to AWS S3 or local storage as a pickle file.

    Args:
        file_name (str): The name of the file (without extension) to save.
        data (pd.DataFrame): The DataFrame to be saved.

    Returns:
        bool: True if the upload was successful, False otherwise.

    The function checks if AWS is available (IS_AWS_AVAILABLE). If so, it uploads the file to S3 using the provided AWS credentials and configuration. Otherwise, it saves the file locally under the 'embeddings' directory inside DATA_PATH.
    """
    if IS_AWS_AVAILABLE:
        # Initialize a session using AWS credentials
        s3_client = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        ).client("s3")

        try:
            # Serialize the DataFrame to pickle format
            pickle_data = pickle.dumps(data)
            # Construct the S3 file key (path in the bucket)
            s3_file_key = f"{AWS_FOLDER}/{file_name}.pkl"
            # Upload the serialized data to S3
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_file_key, Body=pickle_data)
        except ClientError as e:
            # Log AWS client errors
            logger.error(e)
            return False
        return True
    else:
        # Local storage alternative
        try:
            # Create the 'embeddings' directory if it doesn't exist
            data_dir = os.path.join(DATA_PATH, "embeddings")
            os.makedirs(data_dir, exist_ok=True)

            # Construct the full file path
            file_path = os.path.join(data_dir, f"{file_name}.pkl")
            # Save the DataFrame to a local pickle file
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to local file: {file_path}")
            return True
        except Exception as e:
            # Log any error that occurs during local saving
            logger.error(f"Error saving to local storage: {e}")
            return False


def download_file(file_name: str) -> pd.DataFrame | bool:
    """
    Downloads a pandas DataFrame from AWS S3 or local storage.

    Args:
        file_name (str): The name of the file (without extension) to load.

    Returns:
        Union[pd.DataFrame, bool]: The loaded DataFrame if successful, or an empty DataFrame (or False) if not found or on error.

    The function checks if AWS is available (IS_AWS_AVAILABLE). If so, it downloads the file from S3 using the provided AWS credentials and configuration. Otherwise, it loads the file from local storage under the 'embeddings' directory inside DATA_PATH.
    """
    if IS_AWS_AVAILABLE:
        # Initialize a session using AWS credentials
        s3_client = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        ).client("s3")

        # Create a BytesIO object as an in-memory binary stream to hold the downloaded data
        file_stream = BytesIO()

        try:
            # Construct the S3 file key (path in the bucket)
            s3_file_key = f"{AWS_FOLDER}/{file_name}.pkl"
            # Download the file from S3 into the in-memory stream
            s3_client.download_fileobj(Bucket=S3_BUCKET_NAME, Key=s3_file_key, Fileobj=file_stream)
            # Move the stream position to the start
            file_stream.seek(0)

            # Deserialize the DataFrame from the stream
            data = pickle.load(file_stream)

        except ClientError as e:
            # Log AWS client errors
            logger.error(e)
            data = pd.DataFrame()
        return data
    else:
        # Local storage alternative
        try:
            # Construct the directory and file path
            data_dir = os.path.join(DATA_PATH, "embeddings")
            file_path = os.path.join(data_dir, f"{file_name}.pkl")

            # Check if the file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()

            # Load the DataFrame from the local pickle file
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded data from local file: {file_path}")
            return data
        except Exception as e:
            # Log any error that occurs during local loading
            logger.error(f"Error loading from local storage: {e}")
            return pd.DataFrame()
