# External imports
import time

# Third-party imports
from openai import OpenAI

from src.config.parameters import EXIT_WORDS

# Internal imports
# from src.config.settings import OPENAI_API_KEY, OPENAI_ASSISTANT_ID

OPENAI_API_KEY = "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"
OPENAI_ASSISTANT_ID = "asst_e07lEcDq9z8It8DZ4y3FAdu3"

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")

# Create a new thread for each conversation
thread = client.beta.threads.create()


def submit_message(thread, user_message: str):
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=OPENAI_ASSISTANT_ID,
    )


def get_response(thread) -> list:
    return client.beta.threads.messages.list(thread_id=thread.id, order="desc")


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.1)
    return run


def print_assistant_response(messages: list) -> None:
    """Printing helper for the latest assistant response"""
    for m in messages:
        if m.role == "assistant":
            print(f"🤖: {m.content[0].text.value}")
            break


def decode_assistant_response(messages: list) -> str:
    response = ""
    for m in messages:
        if m.role == "assistant":
            response = m.content[0].text.value
            break
    return response


def list_assistants() -> list:
    """
    Retrieves and displays all available OpenAI assistants.

    Returns:
        list: A list of assistant objects
    """
    assistants = client.beta.assistants.list(
        order="desc",
        limit=100,
    )

    print("🤖 Asistentes Disponibles:")
    for assistant in assistants.data:
        print(f"ID: {assistant.id} | Nombre: {assistant.name} | Modelo: {assistant.model}")

    return assistants.data


if __name__ == "__main__":
    # Uncomment to list available assistants
    # list_assistants()

    while True:
        user_input = input("🧑: ")
        if any(keyword in user_input.lower() for keyword in EXIT_WORDS):
            print("🤖: ¡Chao! Sesión de chat finalizada.")
            break

        # Submit the user's message to the existing thread
        run = submit_message(thread, user_input)

        # Wait for the assistant to respond
        run = wait_on_run(run, thread)

        # Display the assistant's response
        assistant_response = get_response(thread)
        print_assistant_response(assistant_response)
