import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


if device == "cuda":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA compute capability:", torch.cuda.get_device_capability(0))
    print("Memory total (MB):", torch.cuda.get_device_properties(0).total_memory / 1024**2)


# model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"


"""
Why float16 on GPU?
- Speed: Modern GPUs (especially NVIDIA with Tensor Cores) are highly optimized for float16
(also called FP16) computation. It's faster and uses less memory.
- Efficiency: FP16 allows twice the throughput compared to FP32 on GPUs, enabling faster inference
with larger batch sizes or sequences.
- Support: Transformers like SmolLM2 are typically trained and optimized to run efficiently in FP16 on GPU.

Why float32 on CPU?
- Compatibility: Most CPUs do not support fast FP16 operations. If you force FP16 on CPU, it can be slower
and error-prone, or not work at all.
- Stability: FP32 (float32) has higher numerical precision, which avoids instability or degradation during
model inference on CPU.
"""
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

"""
- max_new_tokens=200: Limita la salida generada a un máximo de 200 tokens nuevos, asegurando
que las respuestas sean concisas y no excedan la longitud deseada.
- temperature=0.7: Controla el grado de aleatoriedad en la selección de tokens. Un valor de 0.7
introduce una variabilidad moderada, equilibrando creatividad y coherencia.
Valores más bajos hacen la salida más determinista; valores más altos, más creativa y diversa.
- do_sample=True: Activa el muestreo durante la generación, permitiendo que el modelo seleccione
tokens basándose en su distribución de probabilidad. Esto fomenta respuestas más variadas y naturales.
- repetition_penalty=1.5: Penaliza la repetición de palabras o frases ya generadas, reduciendo su
probabilidad de ser reutilizadas. Un valor superior a 1.0 aplica esta penalización, y valores
más altos la refuerzan.
- repetition_penalty=1.5: Penaliza la repetición de palabras o frases ya generadas, reduciendo su
probabilidad de ser reutilizadas. Un valor superior a 1.0 aplica esta penalización, y valores más
altos la refuerzan.
- no_repeat_ngram_size=3: Evita que el modelo repita cualquier secuencia de 3 palabras consecutivas
(trigramas), lo que mejora la fluidez y originalidad del texto generado.
- return_full_text=False: Hace que se retorne únicamente el texto generado por el modelo, sin incluir
el prompt original. Esto es útil cuando sólo se desea la respuesta del modelo.
"""
# Create a text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.5,
    # no_repeat_ngram_size=3,
    return_full_text=False,
)

# Wrap the pipeline in LangChain's LLM class
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize ChatHuggingFace with the LLM
chat_model = ChatHuggingFace(llm=llm)

# Define the conversation messages
messages = [
    SystemMessage(
        content="Eres un asistente útil que habla español. Provees respuestas cortas y directas."
    ),
    HumanMessage(content="¿Cuál es la mejor manera de desplegar modelos LLMs?"),
]

# Generate and print the assistant's response
response = chat_model.invoke(messages)
print(response.content)
