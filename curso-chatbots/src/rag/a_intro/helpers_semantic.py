"""
Created by Analitika at 12/08/2024
contact@analitika.fr
"""

# Standard imports
import json
import os

# Third party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import AutoModel, AutoTokenizer

# Internal imports
from src.config.settings import DATA_PATH


def load_sentences():
    src = os.path.join(DATA_PATH, "short_sentences.json")
    sens = pd.DataFrame()
    with open(src, encoding="utf-8") as f:
        sens = pd.DataFrame(json.load(f))
    return sens


def compute_embeddings(text_: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    # Prepare text
    encoded_input = tokenizer(text_, return_tensors="pt")
    print(text_, len(encoded_input["input_ids"][0]))
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Obtain the embeddings from the last hidden state
    return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()


def do_radar_plot(results, sentences_df, annotation):
    # Get the sentences (index) and their similarity values for the radar plot
    theta = [f"frase {_}" for _ in sentences_df.index]

    # Create radar plot
    fig = go.Figure()

    for col in sentences_df.columns:
        # Select the top sentences and corresponding similarities for each column
        top_sentences_i = results[col].tolist()

        # Add the first curve
        fig.add_trace(go.Scatterpolar(r=top_sentences_i, theta=theta, fill="toself", name=col))

    # Update the layout of the radar plot
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        title="Medidas de similaridad semántica: <br>" + annotation + "<br>",
        annotations=[
            {
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "text": "",
                "showarrow": False,
                "font": {"size": 12},
                "align": "left",
            }
        ],
    )

    # Show the plot
    fig.show()


def do_scatter_plot(df_emb):
    # Create a scatter plot with different symbols for each label
    fig = px.scatter(
        df_emb,
        x="X",
        y="Y",
        color="label",  # Different colors for different labels
        symbol="label",  # Different symbols for different labels
        title="UMAP Proyección de Embeddings",
        labels={"X": "UMAP Dimension 1", "Y": "UMAP Dimension 2"},
    )

    # Customize layout for better visualization
    fig.update_traces(marker={"size": 10, "line": {"width": 2}})
    fig.update_layout(legend_title_text="Categorías de Frases", width=800, height=600)

    # Show the plot
    fig.show()
