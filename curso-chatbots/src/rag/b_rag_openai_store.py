"""
Created by Datoscout on 12/02/2025
solutions@datoscout.ec
"""

# Standard imports
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

# Third party imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
from src.config.settings import (
    DATA_PATH,
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
    OPENAI_EMBEDDINGS_MODEL,
)

# Initialize OpenAI client
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_COMPLETIONS_MODEL, max_tokens=64)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL, show_progress_bar=True)

# Define the path to the text file (Alice in Wonderland)
text = os.path.join(DATA_PATH, "Alicia en el país de las maravillas.txt")

# Load the text file using LangChain's TextLoader
loader = TextLoader(file_path=text, encoding="utf-8")

# This helps improve retrieval efficiency and accuracy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Each chunk will have 500 characters
    chunk_overlap=0,  # No overlap between consecutive chunks
)
# Apply text splitting to the loaded document
data = loader.load_and_split(text_splitter=text_splitter)

vectorstore = InMemoryVectorStore.from_documents(data, embedding=embeddings)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "fetch_k": 10, "maximal_marginal_relevance": True}
)

# Create a prompt template
template = """
Eres un asistente de IA que responde preguntas basadas en la información recuperada.

Contexto: {context}
Pregunta: {question}

Responde la pregunta usando únicamente el contexto proporcionado.
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create the chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# Example usage
question = "¿qué camino debo seguir para salir de aquí?"
# question = "¿es buena idea retroceder al día de ayer?"
response = chain.invoke(question)
print(f"Pregunta: {question}")
print(f"Respuesta: {response}")

# For comparison with the original code
retrieved_documents = retriever.invoke(question)
for i, doc in enumerate(retrieved_documents):
    print(f"{i + 1}. Documento recuperado: \n{doc.page_content}\n\n")
