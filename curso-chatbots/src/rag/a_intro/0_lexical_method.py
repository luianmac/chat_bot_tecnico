"""
BM25 for Text Similarity in RAG
This script demonstrates how to use BM25 for determining text similarity
in the context of Retrieval-Augmented Generation (RAG).
"""

# Standard imports
import re
import string

# Third party imports
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Helper Functions
def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing, removing punctuation and stopwords."""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Remove stopwords
    stop_words = set(stopwords.words("spanish"))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    return " ".join(filtered_tokens)


def tokenize_texts(texts: list[str]) -> list[list[str]]:
    """Tokenize a list of texts into words."""
    return [text.split() for text in texts]


def create_bm25_index(corpus: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    """Create a BM25 index from a corpus of texts."""
    # Preprocess texts
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]

    # Tokenize preprocessed texts
    tokenized_corpus = tokenize_texts(preprocessed_corpus)

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


def get_bm25_scores(
    query: str, bm25_index: BM25Okapi, tokenized_corpus: list[list[str]]
) -> np.ndarray:
    """Get BM25 scores for a query against the corpus."""
    # Preprocess and tokenize query
    preprocessed_query = preprocess_text(query)
    tokenized_query = tokenize_texts([preprocessed_query])[0]

    # Get BM25 scores
    scores = bm25_index.get_scores(tokenized_query)

    return scores


def visualize_bm25_scores(scores: np.ndarray, corpus: list[str], query: str) -> None:
    """Visualize BM25 scores for a query against the corpus."""
    plt.figure(figsize=(12, 6))

    # Create a dataframe for better visualization
    df = pd.DataFrame(
        {
            "Document": [f"Doc {i + 1}" for i in range(len(corpus))],
            "Score": scores,
            "Text": [doc[:120] + "..." if len(doc) > 120 else doc for doc in corpus],
        }
    )

    # Sort by score
    df = df.sort_values("Score", ascending=False)

    # Create the plot
    ax = sns.barplot(x="Score", y="Document", data=df, palette="viridis")

    # Add document text as annotations
    for i, row in df.iterrows():
        ax.text(0.01, i, row["Text"], fontsize=8, ha="left", va="center")

    plt.title(f'BM25 Scores for Query: "{query}"')
    plt.tight_layout()
    plt.show()


def explain_bm25_calculation(
    query: str, doc: str, bm25_index: BM25Okapi, tokenized_corpus: list[list[str]], doc_idx: int
) -> dict:
    """Explain the BM25 calculation for a specific document."""
    # Preprocess and tokenize query and document
    preprocessed_query = preprocess_text(query)
    tokenized_query = tokenize_texts([preprocessed_query])[0]

    # Get document length and average document length
    doc_len = len(tokenized_corpus[doc_idx])
    avg_doc_len = bm25_index.avgdl

    # Get parameters
    k1 = bm25_index.k1
    b = bm25_index.b

    # Calculate term frequencies in the document
    term_freqs = {}
    for term in set(tokenized_query):
        term_freqs[term] = tokenized_corpus[doc_idx].count(term)

    # Retrieve inverse document frequencies
    idfs = {}
    for term in set(tokenized_query):
        # n_docs_with_term = sum(1 for doc in tokenized_corpus if term in doc)
        idf = bm25_index.idf[term] if term in bm25_index.idf else 0
        idfs[term] = idf

    # Calculate term scores
    term_scores = {}
    for term in set(tokenized_query):
        if term in term_freqs and term in idfs:
            tf = term_freqs[term]
            idf = idfs[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            term_scores[term] = idf * numerator / denominator
        else:
            term_scores[term] = 0

    # Calculate total score
    total_score = sum(term_scores.values())

    return {
        "query": query,
        "document": doc,
        "tokenized_document": tokenized_corpus[doc_idx],
        "document_length": doc_len,
        "average_document_length": avg_doc_len,
        "k1": k1,
        "b": b,
        "term_frequencies": term_freqs,
        "inverse_document_frequencies": idfs,
        "term_scores": term_scores,
        "total_score": total_score,
    }


def compare_documents_bm25(corpus: list[str], queries: list[str]) -> pd.DataFrame:
    """Compare multiple documents against multiple queries using BM25."""
    # Create BM25 index
    bm25_index, tokenized_corpus = create_bm25_index(corpus)

    # Initialize results dataframe
    results = pd.DataFrame(index=range(len(queries)), columns=range(len(corpus)))

    # Fill results with BM25 scores
    for i, query in enumerate(queries):
        scores = get_bm25_scores(query, bm25_index, tokenized_corpus)
        for j, score in enumerate(scores):
            results.iloc[i, j] = score

    # Set column names as document snippets
    results.columns = [f"Doc {i + 1}: {doc[:35]}..." for i, doc in enumerate(corpus)]

    # Set row names as queries
    results.index = [f"Query: {query}" for query in queries]

    return results


def visualize_similarity_heatmap(results: pd.DataFrame) -> None:
    """Visualize document-query similarity as a heatmap."""
    plt.figure(figsize=(14, 5))

    # Ensure data is numeric
    results_numeric = results.astype(float)

    # Create heatmap
    sns.heatmap(results_numeric, annot=True, cmap="YlGnBu", fmt=".2f")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=25, ha="right")

    # Adjust bottom margin to accommodate rotated labels
    plt.subplots_adjust(bottom=0.4, left=0.2)

    plt.title("BM25 Document-Query Similarity")
    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    # Define a reference text and a collection of texts
    text_reference = """La ciencia de datos es la disciplina científica enfocada en analizar grandes fuentes de
    datos para extraer información, comprender la realidad y descubrir patrones para la toma de decisiones.
    """

    corpus = [
        "La ciencia de datos combina estadística, matemáticas y programación para analizar información.",
        "El aprendizaje automático es un subconjunto de la inteligencia artificial enfocado en construir sistemas que aprenden de los datos.",
        "Big data se refiere a conjuntos de datos extremadamente grandes que pueden ser analizados para revelar patrones y tendencias.",
        "La estadística es la disciplina que se ocupa de la recolección, organización, análisis e interpretación de datos.",
        "La minería de datos es el proceso de descubrir patrones en grandes conjuntos de datos.",
        "La toma de decisiones basada en el análisis de datos mejora significativamente los resultados empresariales.",
        "Python es un lenguaje de programación popular utilizado para ciencia de datos y aprendizaje automático.",
        "La visualización de datos ayuda a comunicar de manera clara y efectiva las ideas complejas de los datos.",
        "La inteligencia artificial busca simular procesos de inteligencia humana mediante máquinas.",
        "El método científico es un enfoque sistemático para la investigación y el descubrimiento.",
    ]

    # Create queries for comparison
    queries = [
        "¿Qué es la ciencia de datos?",
        "¿Cómo funciona el aprendizaje automático?",
        "¿Cuáles son los beneficios del análisis de datos?",
        "¿Qué herramientas se utilizan en la ciencia de datos?",
        "Enfoques científicos para los datos",
    ]

    # Add the reference text to the corpus
    all_texts = [text_reference] + corpus

    # Create BM25 index
    bm25_index, tokenized_corpus = create_bm25_index(all_texts)

    # Example 1: Calculate BM25 scores for a single query
    query = "¿Cómo ayuda la ciencia de datos en la toma de decisiones?"
    scores = get_bm25_scores(query, bm25_index, tokenized_corpus)

    # Print results
    print("BM25 Scores for query:", query)
    for i, (text, score) in enumerate(zip(all_texts, scores)):
        print(f"Documento {i}: Score = {score:.4f}")
        print(f"Text: {text[:100]}...\n")

    # Visualize the scores
    visualize_bm25_scores(scores, all_texts, query)

    # Example 2: Explain BM25 calculation for the highest scoring document
    best_doc_idx = np.argmax(scores)
    explanation = explain_bm25_calculation(
        query, all_texts[best_doc_idx], bm25_index, tokenized_corpus, best_doc_idx
    )

    print("\nBM25 Explicación de cálculo:")
    print(f"Consulta: {explanation['query']}")
    print(f"Documento: {explanation['document']}")
    print(f"Longitud del Documento: {explanation['document_length']} tokens")
    print(f"Longitud Media del Documento: {explanation['average_document_length']:.2f} tokens")
    print(f"Parámetros: k1={explanation['k1']}, b={explanation['b']}")
    print("\nFrecuencias de Términos:")
    for term, freq in explanation["term_frequencies"].items():
        print(f"  {term}: {freq}")
    print("\nFrecuencias de Documentos Inversos:")
    for term, idf in explanation["inverse_document_frequencies"].items():
        print(f"  {term}: {idf:.4f}")
    print("\nScores de Términos:")
    for term, score in explanation["term_scores"].items():
        print(f"  {term}: {score:.4f}")
    print(f"\nScore Total: {explanation['total_score']:.4f}")

    # Example 3: Compare multiple documents against multiple queries
    comparison_results = compare_documents_bm25(all_texts, queries)

    print("\nMatriz de Similaridad Documento-Consulta:")
    print(comparison_results)

    # Visualize the comparison
    visualize_similarity_heatmap(comparison_results)

    # Example 4: Show how BM25 can be used in RAG
    print("\nRAG Ejemplo con BM25:")
    user_query = "¿Cuál es la relación entre la ciencia de datos y la toma de decisiones?"

    # 1. Retrieve relevant documents using BM25
    retrieval_scores = get_bm25_scores(user_query, bm25_index, tokenized_corpus)
    top_k = 3  # Number of documents to retrieve
    top_indices = np.argsort(retrieval_scores)[::-1][:top_k]

    print(f"Consulta del Usuario: {user_query}")
    print("\nDocumentos Recuperados:")
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. Score: {retrieval_scores[idx]:.4f}")
        print(f"   Documento: {all_texts[idx]}")

    # 2. Simulating augmented generation with retrieved context
    print(
        "\nEn un sistema RAG, estos documentos principales se enviarían a un LLM para generar una respuesta."
    )
    print(
        "El LLM utilizaría el contexto recuperado para proporcionar una respuesta más precisa e informada."
    )
    print("El ejemplo de prompt para el LLM sería:")
    print("-" * 80)
    print(f"Dada la consulta del usuario: '{user_query}'")
    print("Y la siguiente información relevante:")
    for i, idx in enumerate(top_indices):
        print(f"{i}- {all_texts[idx]}")
    print("\nPor favor, proporciona una respuesta completa a la consulta del usuario.")
    print("-" * 80)
