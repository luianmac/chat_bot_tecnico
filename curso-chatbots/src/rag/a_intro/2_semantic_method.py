"""
Embedding-based RAG Example
----------------------------
This script demonstrates how to use embeddings for semantic similarity
in the context of Retrieval-Augmented Generation (RAG).

Created by Datoscout at 09/08/2024
solutions@datoscout.ec
"""

# Standard imports

# Third party imports
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

# Import helper functions
from src.rag.a_intro.helpers_semantic import (
    compute_embeddings,
    do_radar_plot,
    do_scatter_plot,
    load_sentences,
)


def initialize_model() -> tuple[AutoTokenizer, AutoModel]:
    """Initialize and return the embedding model and tokenizer."""
    model_str = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModel.from_pretrained(model_str)
    return tokenizer, model


def generate_sentence_embeddings(sentences_df, tokenizer, model):
    """
    Generate embeddings for all sentences in the dataframe.

    Args:
        sentences_df: DataFrame containing sentences organized by categories
        tokenizer: Pre-trained tokenizer
        model: Pre-trained embedding model

    Returns:
        all_embeddings: List of embeddings for each sentence
        all_labels: List of category labels corresponding to each embedding
        results: DataFrame of similarity scores
    """
    all_embeddings = []
    all_labels = []
    results = pd.DataFrame(columns=sentences_df.columns)

    for column in sentences_df.columns:
        for idx, sentence in enumerate(sentences_df[column]):
            # Generate embeddings for the sentence
            sentence_embeddings = compute_embeddings(sentence, tokenizer, model)

            # Store embeddings and labels
            all_embeddings.append(sentence_embeddings)
            all_labels.append(column)

            # Calculate similarity with reference text
            similarity = cosine_similarity([text_reference_embeddings], [sentence_embeddings])

            results.loc[idx, column] = similarity[0][0]

    return np.array(all_embeddings), all_labels, results


def reduce_dimensions_umap(embeddings, n_components=2):
    """
    Reduce the dimensionality of embeddings using UMAP.

    Args:
        embeddings: High-dimensional embeddings to reduce
        n_components: Number of dimensions in the reduced space

    Returns:
        DataFrame with reduced dimensions and labels
    """
    # Scale features to [0,1] range
    X_scaled = MinMaxScaler().fit_transform(embeddings)

    # Initialize and fit UMAP
    mapper = UMAP(n_components=n_components, metric="cosine", random_state=42)
    reduced_embeddings = mapper.fit_transform(X_scaled)

    # Create a DataFrame of reduced dimensions
    df_reduced = pd.DataFrame(reduced_embeddings, columns=["X", "Y"])

    return df_reduced


if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model = initialize_model()

    # # Define reference text (in Spanish)
    text_reference = (
        "Disciplina científica centrada en el análisis de grandes fuentes de datos "
        "para extraer información, comprender la realidad y descubrir patrones para tomar decisiones."
    )

    print("\nTexto de Referencia:")
    print(text_reference)

    # Compute embeddings for reference text
    text_reference_embeddings = compute_embeddings(text_reference, tokenizer, model)

    # Load example sentences
    print("\nCargando frases de ejemplo...")
    sentences_df = load_sentences()

    # Generate embeddings and compute similarities
    print("\nGenerando embeddings y calculando similitudes...")
    all_embeddings, all_labels, results = generate_sentence_embeddings(
        sentences_df, tokenizer, model
    )

    # Create radar plot of similarities
    print("\nCreando gráfico de radar de similitudes de categoría...")
    annotation = text_reference.replace("información, comprender", "información,<br>comprender")
    do_radar_plot(results, sentences_df, "Similaridad al Texto de Referencia")

    # Reduce dimensions using UMAP
    print("\nReduciendo dimensiones de embeddings usando UMAP...")
    df_emb = reduce_dimensions_umap(all_embeddings)
    df_emb["label"] = all_labels

    # Create scatter plot
    print("\nCreando gráfico de dispersión de embeddings de documentos...")
    do_scatter_plot(df_emb)

    print("\nConclusión:")
    print("Este script demuestra cómo los embeddings pueden capturar la similitud")
    print("semántica entre textos, incluso a través de diferentes idiomas.")
    print("La visualización muestra cómo los textos semánticamente similares")
    print("se agrupan juntos en el espacio de embeddings.")
