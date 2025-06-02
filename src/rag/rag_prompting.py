"""
Created by Datoscout on 18/02/2025
solutions@datoscout.ec

This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain.

It loads a text file, splits it into smaller chunks, converts it into vector embeddings,
stores it in a FAISS vector database, and queries the database using an LLM with a structured prompt.
"""

# External imports
import os
import sys

# from langchain.callbacks import StdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# Internal imports
from src.config.settings import (
    DATA_PATH,
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
    OPENAI_EMBEDDINGS_MODEL,
)

# Set up data file path
DATA_FILE = os.path.join(DATA_PATH, "Alicia en el país de las maravillas.txt")


class RAGChatbot:
    def __init__(self, data_file: str):
        """Initialize the RAG chatbot with OpenAI embeddings, FAISS retriever, and a structured prompt."""
        logger.info("Inicializando Chatbot RAG")

        try:
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDINGS_MODEL, openai_api_key=OPENAI_API_KEY
            )
            self.llm = ChatOpenAI(
                model=OPENAI_COMPLETIONS_MODEL, openai_api_key=OPENAI_API_KEY, max_tokens=128
            )
        except Exception as e:
            logger.error(f"Error al inicializar LLM o embeddings: {e}")
            sys.exit(1)

        self.vectorstore = self.create_vectorstore(data_file)
        self.retriever = self.configure_retriever()
        self.prompt_template = self.create_prompt_template()

    def create_vectorstore(self, data_file: str):
        """Loads text from a file, splits it into chunks, and stores it in a FAISS vector store."""
        logger.info(f"Cargando texto desde {data_file}")

        try:
            loader = TextLoader(file_path=data_file, encoding="utf-8")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = loader.load_and_split(text_splitter=text_splitter)
        except Exception as e:
            logger.error(f"Error al cargar o dividir documentos: {e}")
            raise

        logger.info("Creando almacén vectorial FAISS")
        try:
            vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
        except Exception as e:
            logger.error(f"Error al crear almacén vectorial FAISS: {e}")
            raise
        return vectorstore

    def configure_retriever(self):
        """Configures the retriever with optimized search parameters."""
        retriever = self.vectorstore.as_retriever()
        # Update search parameters: fetch more documents then narrow down to the top-k after re-ranking.
        retriever.search_kwargs.update(
            {
                "fetch_k": 20,  # Fetch 20 documents before re-ranking
                "maximal_marginal_relevance": True,  # Ensure diverse results
                "k": 10,  # Return top 10 most relevant documents
            }
        )
        logger.info(
            "Retriever configurado con parámetros de búsqueda: %s", retriever.search_kwargs
        )
        return retriever

    def create_prompt_template(self) -> ChatPromptTemplate:
        """Creates a structured prompt template for the RAG chain."""
        logger.info("Creando plantilla de prompt estructurado")
        template = (
            "Responde la pregunta basada únicamente en el siguiente contexto:\n"
            "{context}\n\n"
            "Pregunta: {question}\n\n"
            "Responde de manera completa, proporcionando detalles relevantes del contexto."
        )
        try:
            prompt = ChatPromptTemplate.from_template(template)
        except Exception as e:
            logger.error(f"Error al crear plantilla de prompt: {e}")
            raise
        return prompt

    def generate_answer(self, question: str) -> str:
        """
        Retrieves relevant context using the retriever, constructs the prompt,
        and generates an answer using the LLM.
        """
        logger.info(f"Procesando pregunta: {question}")
        try:
            # Retrieve context using the retriever. Assume `invoke` returns text context.
            context = self.retriever.invoke(question)
            logger.debug("Contexto recuperado: {}", context)

            # Prepare the prompt with the retrieved context and question
            prompt_input = {"context": context, "question": question}
            formatted_prompt = self.prompt_template.format(**prompt_input)
            logger.debug("Prompt formateado: {}", formatted_prompt)

            # Call the language model with the formatted prompt.
            # Optionally, add callbacks or additional parameters if needed.
            raw_response = self.llm.invoke(formatted_prompt)
            logger.debug("Respuesta de LLM sin procesar: {}", raw_response)

            # Parse the raw response into a clean string answer
            answer = StrOutputParser().parse(raw_response)
        except Exception as e:
            logger.error(f"Error durante la generación de la respuesta: {e}")
            answer = "Lo siento, ocurrió un error al generar la respuesta."
        return answer


def main():
    logger.info("Iniciando aplicación de Chatbot RAG")
    chatbot = RAGChatbot(DATA_FILE)

    # Example questions to test the chatbot
    questions = [
        "¿De qué trata la historia de Alicia en el país de las maravillas?",
        "¿Quién es el Hatter Loco?",
        "¿Qué consejo da el Gato de Cheshire a Alicia?",
        "¿Qué camino deberías tomar si no sabes a dónde vas?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        answer = chatbot.generate_answer(question)
        print(f"A{i}: {answer.content}")


if __name__ == "__main__":
    main()
