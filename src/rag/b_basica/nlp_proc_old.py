"""
Created by Datoscout at 28/03/2024
solutions@datoscout.ec
"""

# Standard imports
import random
import time

# Third party imports
import numpy as np
import pandas as pd

# Internal imports
from src.config.parameters import NAIVE_RAG_THRESHOLD
from src.models_ia.call_model import generate_answer, get_embeddings


# Streamed response emulator
def response_generator(question: str, embeddings: pd.DataFrame):
    q_emb = get_embeddings(question)  # question's embeddings
    embeddings["similarities"] = embeddings["embeddings"].apply(
        lambda x: cosine_similarity(x[0], q_emb[0])
    )
    embeddings = embeddings.sort_values("similarities", ascending=False).head(4)
    result = embeddings.loc[embeddings.similarities > NAIVE_RAG_THRESHOLD].sort_values(
        "similarities", ascending=False
    )

    if result.empty:
        response = "Relevant context not found. (err 1)"
    else:
        context = []
        source = {}
        for i, row in result.iterrows():
            context.append(row["text"])
            if row["page"] in source:
                source[row["page"]].append(i)
            else:
                source[row["page"]] = [i]

        text = "\n".join(context)
        response = generate_answer(question, text)
        src = ""
        for key in source:
            prgphs = ", ".join([str(pp) for pp in source[key]])
            src += f"Página {key}: Sección {prgphs}, "

        # Remove trailing comma and space
        src = src.rstrip(", ")
        response += "\nRefs:\n(" + src + ".)"

    for word in response.split():
        yield word + " "
        time.sleep(0.03)


# Streamed response emulator
def response_random_generator():
    response = random.choice(
        [
            "Hola! ¿Cómo puedo ayudarte hoy?",
            "Hola, humano! ¿Necesitas ayuda?",
            "¿Necesitas ayuda?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def compute_embeddings(full_text: dict) -> pd.DataFrame:
    data = []  # Initialize an empty list to collect data
    # Iterate through each page and its paragraphs
    for page, paragraphs in full_text.items():
        embeddings = [
            (get_embeddings(pgs), pgs) for pgs in paragraphs
        ]  # Get embeddings for all paragraphs in a page
        for i, embedding in enumerate(embeddings):
            # Append a dictionary for each row of data we want in the DataFrame
            data.append(
                {
                    "page": page,
                    "paragraph": i,
                    "embeddings": embedding[0],
                    "text": embedding[1],
                }
            )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    return df


def cosine_similarity(a: np.array, b: np.array) -> float:
    return np.array(a) @ np.array(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def debug_embeddings():
    # filename = "source_file.pdf"
    f_text = {
        0: [
            """
    - We start by tokenizing the inputs (the context- this is your database) and computing embeddings (this are numerical vectors that represents the sentences in the text and contain semantic meaning). Context's embeddings are stored in a vector database.
    """,
            """
    - It then uses questions's embeddings to identify the best section of the database to use to answer to the questions. The best section is identified thanks to a similarity score computed over the embeddings.
    """,
            """
    - For embeddings, we can use the "text-embedding-3-large" model, and "gpt-3.5-turbo-instruct" for generating responses - (which provides a summarization of the context provided), the call to the APIs for this model are quite cheap: $0.13 and $1.50/$2 (input/output) for 1M tokens for the first and second models respectively, where a token is roughly a word. For a better management of the tone of the answers we could use ChatGPT-4 (bigger model, more expensive but much more adaptable to your needs).
    """,
            """
    I recommend first attempting to get good results with prompt engineering and prompt chaining, if the results are not satisfying we can use fine-tuning to improve model's performance, either the same OpenAI models, or to use a model from existing open-source ones.
    """,
        ]
    }

    df = compute_embeddings(f_text)

    return df


if __name__ == "__main__":
    debug_embeddings()
