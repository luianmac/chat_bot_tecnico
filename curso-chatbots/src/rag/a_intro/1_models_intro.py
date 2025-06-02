"""
Created by Datoscout at 09/08/2024
solutions@datoscout.ec
"""

# External imports
import torch
from huggingface_hub import InferenceClient
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

# Internal imports
from src.config.settings import HUGGINGFACE_TOKEN

# check whether a GPU is available
is_gpu = torch.cuda.is_available()
if is_gpu:
    print("GPU is available")

# Variables
model_str = "distilbert-base-multilingual-cased"
text = "Machine Learning es el mejor [MASK] de la historia"


def local_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelForMaskedLM.from_pretrained(model_str)

    encoded_input = tokenizer(text, return_tensors="pt")
    len(encoded_input["input_ids"][0])

    # See the tokens
    tkns = [tokenizer.decode([token_id]).strip() for token_id in encoded_input["input_ids"][0]]
    print("Tokens: ", tkns)

    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Get the predicted token ID
    masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
    # predicted_token_id = model_output.logits[0, masked_index].argmax(axis=-1)  # top one
    logits = model_output.logits[0, masked_index, :]

    # Get the top 5 token IDs with the highest probability
    top_5_token_ids = logits.topk(5, dim=-1).indices[0].tolist()

    # Convert the top 5 token IDs to corresponding words
    top_5_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_5_token_ids]

    # Display the original text and the top 3 predictions
    print(f"Texto Original: {text}")
    print("Top 5 Predicciones:")
    for i, token in enumerate(top_5_tokens, start=1):
        completed_text = text.replace(tokenizer.mask_token, f"*{token}*")
        print(f"{i}: {completed_text}")


def local_pipeline():
    unmasker = pipeline("fill-mask", model="distilbert-base-multilingual-cased")
    print(unmasker("Hello I'm a [MASK] model."))
    return


def api_model():
    # Call to the HuggingFace API
    # https://huggingface.co/docs/inference-providers/en/providers/hf-inference
    client = InferenceClient(provider="hf-inference", token=HUGGINGFACE_TOKEN)
    try:
        model_ = "emilyalsentzer/Bio_ClinicalBERT"
        data = client.fill_mask(text=text, model=model_)
        # https://huggingface.co/distilbert/distilbert-base-multilingual-cased
        # This model isn't deployed by any Inference Provider.
        for i, ans in enumerate(data):
            completed_text = ans.sequence  # Access sequence attribute directly
            print(f"{i}: {completed_text}")
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")


if __name__ == "__main__":
    api_model()
    local_model()
