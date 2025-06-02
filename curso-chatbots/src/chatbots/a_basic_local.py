# External imports
# None needed here

from src.config.parameters import EXIT_WORDS

# Internal imports
from src.local_llm.client_local import LocalAI

# Initialize the LocalAI client
# client = LocalAI(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", by_api=False)
client = LocalAI(model_id="microsoft/Phi-3-mini-4k-instruct", by_api=False)


def initial_message(welcome_message):
    system_message = "Eres un asistente muy útil. Y hablas español."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": welcome_message},
    ]
    return messages


def format_response(response):
    return response.choices[0].message.content


def generate_response(message_history):
    # Generate a response from LocalAI
    response = client.chat.completions.create(
        messages=message_history,
    )
    return response


def main() -> None:
    welcome_message = (
        "¡Hola! Soy tu chatbot local. Pregúntame lo que quieras y haré lo mejor para ayudarte."
    )
    print(f"🤖: {welcome_message}")

    message_history = initial_message(welcome_message)

    while True:
        # Get user input
        print("🧑: ", end="")
        user_input = input()
        message_history.append({"role": "user", "content": user_input})

        # Check if the conversation is complete
        if any(exit_keyword in user_input.lower() for exit_keyword in EXIT_WORDS):
            print("🤖: ¡Chao! Sesión de chat finalizada.")
            break

        # Generate and display the bot's response
        response = generate_response(message_history)
        bot_response = format_response(response)
        print(f"🤖: {bot_response}")

        # Add the bot's response to the chat history
        message_history.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    # message = "¿Cuál es la mejor manera de desplegar modelos LLMs?"
    main()
