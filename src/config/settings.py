# External imports
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Environment variables
load_dotenv()

# AWS CREDENTIALS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
AWS_REGION = os.getenv("AWS_REGION", None)
S3_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", None)
AWS_FOLDER = os.getenv("AWS_FOLDER", None)

IS_AWS_AVAILABLE = False
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET_NAME and AWS_FOLDER:
    IS_AWS_AVAILABLE = True

# OPENAI CREDENTIALS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", None)
OPENAI_COMPLETIONS_MODEL = os.getenv("OPENAI_COMPLETIONS_MODEL", None)
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL", None)
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID", None)

# DEEPSEEK CREDENTIALS
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", None)
DEEPSEEK_COMPLETIONS_MODEL = os.getenv("DEEPSEEK_COMPLETIONS_MODEL", None)

# HUGGINGFACE CREDENTIALS
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
logger.info(f"Project root: {PROJECT_ROOT}")
