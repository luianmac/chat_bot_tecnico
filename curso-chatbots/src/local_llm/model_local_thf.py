from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Se define el nombre del modelo que se va a cargar desde Hugging Face. Este modelo es un LLM pequeño (360 millones de parámetros) entrenado para seguir instrucciones (Instruct).
# checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
)  # que convierte texto en tokens comprensibles para el modelo
model = AutoModelForCausalLM.from_pretrained(
    checkpoint
)  # El modelo de lenguaje causal (CausalLM), capaz de generar texto palabra por palabra.

messages = [
    {"role": "system", "content": "You are a helpful assistant, you speak spanish."},
    {"role": "user", "content": "Cuál es la mejor manera de cocinar cebollas?"},
]
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False
)  # Esto transforma el chat en un formato de entrada adecuado para el modelo, usando la plantilla estándar definida por el tokenizador.


pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer
)  # combina el modelo y el tokenizador para generar texto fácilmente con una sola llamada.
response = pipe(
    input_text, return_full_text=False, max_new_tokens=200
)  # La opción return_full_text=False indica que solo se devuelva el texto generado, no la entrada original.
bot = response[0]["generated_text"]
if bot.startswith("assistant\n"):
    bot = bot[len("assistant\n") :].strip()

print(bot)
