"""
Created by Datoscout at 28/03/2024
solutions@datoscout.ec

This module provides utility functions for PDF context extraction, text processing, hashing, and DataFrame storage/retrieval for the RAG chatbot application.
"""

# Standard imports
import base64
import hashlib
import re
import tempfile
from io import BytesIO

# Third party imports
import fitz
import pandas as pd

# Internal imports
from src.config.parameters import MAX_PAGES
from src.rag.b_basica.storage import download_file, upload_file


def extract_context(uploaded_file, max_pages: int = MAX_PAGES) -> dict:
    """
    Extracts text content from an uploaded PDF file, up to a maximum number of pages.

    Args:
        uploaded_file: The file-like object representing the uploaded PDF.
        max_pages (int): The maximum number of pages to extract from the PDF.

    Returns:
        dict: A dictionary mapping page numbers to lists of paragraphs extracted from each page.
    """
    # Read the uploaded file into a BytesIO buffer
    file_buffer = BytesIO(uploaded_file.read())
    # Reset buffer pointer to the beginning
    file_buffer.seek(0)
    # Encode the file buffer to base64 (for compatibility)
    encoded_pdf = base64.b64encode(file_buffer.read()).decode("utf-8")
    # Decode the base64 PDF file back to binary
    pdf_binary_file = base64.b64decode(encoded_pdf)
    pdf_file = BytesIO(pdf_binary_file)
    if pdf_file is not None:
        # Save the uploaded file to a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

    # Extract text from the temporary PDF file
    final_text = extract_text_from_pdf_fitz(temp_file_path, max_pages)

    return final_text


def identify_paragraphs(text: str) -> list:
    """
    Splits a string into paragraphs based on the occurrence of at least two newline (\n) characters,
    possibly separated by spaces.

    Args:
        text (str): The input text to split into paragraphs.

    Returns:
        list: A list of paragraphs (strings) with newlines replaced by spaces and stripped of whitespace.
    """
    # Split the text based on the pattern of two or more newlines (with optional spaces)
    parts = re.split(r"(\n\s*\n)+", text)
    # Filter out any empty strings or strings that only contain whitespace
    parts = [part.replace("\n", " ").strip() for part in parts if part.strip()]
    return parts


def extract_text_from_pdf_fitz(pdf_file: str, max_nb_pages: int = 5) -> dict:
    """
    Extracts text from a PDF file using PyMuPDF (fitz), up to a specified number of pages.

    Args:
        pdf_file (str): Path to the PDF file.
        max_nb_pages (int): Maximum number of pages to extract.

    Returns:
        dict: A dictionary mapping page numbers to lists of paragraphs.
    """
    # Open the PDF document
    doc = fitz.open(pdf_file)
    # Get an iterator for the specified range of pages
    pages = doc.pages(0, max_nb_pages)
    full_text = {}
    for pn, page in enumerate(pages):
        # Extract text from each page
        page_text = page.get_text()
        if page_text:
            # Split the page text into paragraphs
            full_text[pn] = identify_paragraphs(page_text)

    return full_text


def hash_string(string_to_hash: str, algorithm: str = "sha256") -> str:
    """
    Hashes a string using the specified algorithm (md5, sha1, or sha256).

    Args:
        string_to_hash (str): The string to hash.
        algorithm (str): The hashing algorithm to use ('md5', 'sha1', or 'sha256').

    Returns:
        str: The hexadecimal hash digest of the input string.
    """
    # Select the hash algorithm
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError("Unsupported algorithm. Choose 'md5', 'sha1' or 'sha256'.")

    # Convert string to bytes and update the hash object
    hash_obj.update(string_to_hash.encode("utf-8"))

    # Return the hexadecimal representation of the digest
    return hash_obj.hexdigest()


def store_dataframe(filename: str, df: pd.DataFrame) -> bool:
    """
    Stores a pandas DataFrame using a hashed filename (for privacy/uniqueness).

    Args:
        filename (str): The original filename to hash.
        df (pd.DataFrame): The DataFrame to store.

    Returns:
        bool: True if the storage was successful, False otherwise.
    """
    # Hash the filename for storage
    filename_hashed = hash_string(filename, "md5")
    # Upload the DataFrame using the hashed filename
    is_ok = upload_file(filename_hashed, df)
    return is_ok


def retrieve_dataframe(filename: str) -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame using a hashed filename.

    Args:
        filename (str): The original filename to hash.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if not found.
    """
    # Hash the filename for retrieval
    filename_hashed = hash_string(filename, "md5")
    # Download the DataFrame using the hashed filename
    db = download_file(filename_hashed)
    return db
