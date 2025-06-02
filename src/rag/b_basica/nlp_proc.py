"""
Modified by [Tu Nombre] for Claro Technical Support Chatbot
Support for PDF, Excel, and CSV files with RAG
"""

# Standard imports
import random
import time
from typing import Dict, List

# Third party imports
import numpy as np
import pandas as pd

# Internal imports
from src.config.parameters import NAIVE_RAG_THRESHOLD
from src.models_ia.call_model import generate_answer, get_embeddings


def compute_embeddings(full_text: dict[int, list[str]]) -> pd.DataFrame:
    """
    Computes embeddings for text from multiple sources (PDF pages or Excel/CSV rows).
    Enhanced to handle metadata about source type.

    Args:
        full_text: Dictionary where keys are page/segment numbers and values are text chunks.
                  Format: {0: ["text1", "text2"], 1: ["text3"]}

    Returns:
        pd.DataFrame with columns: ['page', 'paragraph', 'embeddings', 'text', 'source']
    """
    data = []
    for page, paragraphs in full_text.items():
        embeddings = [(get_embeddings(pg), pg) for pg in paragraphs]
        for i, (emb, text) in enumerate(embeddings):
            data.append(
                {
                    "page": page,
                    "paragraph": i,
                    "embeddings": emb,
                    "text": text,
                    "source": "PDF",  # Default, can be changed by extract_context()
                }
            )

    return pd.DataFrame(data)


def response_generator(question: str, embeddings: pd.DataFrame):
    """
    Generates responses using RAG, now with support for tabular data references.

    Args:
        question: User query
        embeddings: DataFrame with context embeddings and metadata

    Yields:
        Response tokens with source references
    """
    # Calculate question embedding
    q_emb = get_embeddings(question)

    # Compute similarities
    embeddings["similarities"] = embeddings["embeddings"].apply(
        lambda x: cosine_similarity(x[0], q_emb[0])
    )
    # Get top 4 most relevant chunks
    embeddings = embeddings.sort_values("similarities", ascending=False).head(4)
    result = embeddings.loc[embeddings.similarities > NAIVE_RAG_THRESHOLD]

    if result.empty:
        response = (
            "No encontré información relevante en los documentos. Por favor reformula tu pregunta."
        )
    else:
        context = []
        source_info = {}

        for i, row in result.iterrows():
            context.append(row["text"])
            src_type = row.get("source", "PDF")
            page = row["page"]

            if src_type not in source_info:
                source_info[src_type] = {}
            if page not in source_info[src_type]:
                source_info[src_type][page] = []

            source_info[src_type][page].append(row["paragraph"])

        # Format context based on source type
        if any(src in source_info for src in ["Excel", "CSV"]):
            # Tabular data response
            response = "Datos relevantes encontrados:\n"
            for text in context:
                response += f"- {text}\n"
        else:
            # PDF text response
            response = generate_answer(question, "\n".join(context))

        # Add references
        response += "\n\nFuentes:\n"
        for src_type, pages in source_info.items():
            for page, paras in pages.items():
                paras_str = ", ".join(map(str, paras))
                response += f"- {src_type}, Página {page}: Secciones {paras_str}\n"

    # Stream response
    for word in response.split():
        yield word + " "
        time.sleep(0.03)


def cosine_similarity(a: np.array, b: np.array) -> float:
    """Computes cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Debug/testing functions
def debug_embeddings():
    """Test function for embeddings computation."""
    test_data = {
        0: [
            "Router Cisco ASR-903 supports SFP-10G-LR in slots 1-4",
            "Inventory shows 3 available SFP modules in Bogotá",
        ],
        1: ["RBS-001 Location: Bogotá, SFP: SFP-10G-LR, Status: Active"],
    }

    df = compute_embeddings(test_data)
    print("Embeddings computed successfully:")
    print(df.head())
    return df


if __name__ == "__main__":
    # Test the modified functionality
    print("=== Testing NLP Proc Module ===")
    test_df = debug_embeddings()

    # Test question answering
    test_question = "¿Qué módulos SFP están disponibles en Bogotá?"
    print("\nTesting response generator:")
    for token in response_generator(test_question, test_df):
        print(token, end="", flush=True)
