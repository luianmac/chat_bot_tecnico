# Standard imports
import os

import anthropic

# Third party imports
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=apikey)

response = client.responses.create(model="gpt-4o-mini", input="Cuéntame un chiste sobre elefantes")

print(response)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(
    api_key=anthropic_api_key,
)
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Cuéntame un chiste sobre elefantes"}],
)
print(message.content)
