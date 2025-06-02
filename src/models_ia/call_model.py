"""
Created by Datoscout at 27/03/2024
solutions@datoscout.ec
"""

# Third party imports
from loguru import logger
from openai import OpenAI

# Internal imports
from src.config.settings import OPENAI_API_KEY, OPENAI_COMPLETIONS_MODEL, OPENAI_EMBEDDINGS_MODEL

# Configure logging
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(
    text: list,
    model: str = OPENAI_EMBEDDINGS_MODEL,
    dimension: int = 1536,
) -> list:
    """
    text = ["this is a text", "this is divided into parts"]
    model = EMBEDDINGS_MODEL
    dimension = 1536 dimension of the small model
    """

    logger.info(f"Getting embeddings for {text} with model {model}")

    if not model:
        raise KeyError("model not provided")

    response = client.embeddings.create(input=text, model=model, dimensions=dimension)
    return [data.embedding for data in response.data]


def generate_answer(question: str, context: str):
    """
    temperature=0,  # Controls the randomness in the output generation. The hotter, the more random.
                      A temperature of 1 is a standard setting for creative or varied outputs.
    max_tokens=500, # The maximum length of the model's response.
    top_p=1,        # (or nucleus sampling) this parameter controls the cumulative probability distribution
                      of token selection. A value of 1 means no truncation, allowing all tokens to be considered
                      for selection based on their probability.
    frequency_penalty=0,  # Adjusts the likelihood of the model repeating the same line verbatim.
                            Setting this to 0 means there's no penalty for frequency, allowing the model to freely
                            repeat information as needed.
    presence_penalty=0,  # Alters the likelihood of introducing new concepts into the text.
                           A penalty of 0 implies no adjustment, meaning the model is neutral about introducing
                           new topics or concepts.
    generate_answer("how old is Eduardo", "Eduardo was born in 1982, since then he has been growing more and more handsome")
    """

    prompt = f"""Answer the following question using only the context provided.
    Answer in the style of a professional person and be clear, concise and to the point.
    Identify the language of the question and answer in the same language.
    If you don't know the answer for certain, using the same language of the question say I don't know.
    Question: {question}
    """

    response = client.chat.completions.create(
        model=OPENAI_COMPLETIONS_MODEL,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ],
    )

    response_ = response.model_dump()
    return response_["choices"][0]["message"]["content"]
