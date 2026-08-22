"""
Gradio demo for the fine-tuned ViT blood cell classifier.

Inference-only — no training code here, so this file is safe to deploy
directly to a Hugging Face Space (Space SDK: Gradio, entry point: app.py
or app/gradio_demo.py depending on your Space settings).

Usage (local):
    python app/gradio_demo.py

Model resolution order:
    1. Local checkpoint at results/<training.output_dir from config.yaml>
    2. Hugging Face Hub repo at hub.repo_id (from config.yaml), if
       hub.push_to_hub is true or the repo exists publicly
"""

import os

import gradio as gr
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import ViTForImageClassification, ViTImageProcessor

# --- Load config ---------------------------------------------------------

HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(HERE, "..", "configs", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

local_dir = os.path.join(HERE, "..", cfg["training"]["output_dir"])
hub_repo_id = cfg["hub"]["repo_id"]

if os.path.isdir(local_dir) and os.listdir(local_dir):
    model_path = local_dir
    print(f"Loading model from local checkpoint: {model_path}")
else:
    model_path = hub_repo_id
    print(f"Local checkpoint not found. Loading from Hugging Face Hub: {model_path}")

# --- Load model ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
id2label = model.config.id2label

# --- Preprocessing (mirrors eval-time transforms used in training) --------

size = cfg["preprocessing"].get("image_size", processor.size["height"])
eval_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ]
)


def classify_blood_cell(image: Image.Image):
    if image is None:
        return None
    image = image.convert("RGB")
    pixel_values = eval_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return {id2label[i]: float(probs[i]) for i in range(len(id2label))}


# --- UI ---------------------------------------------------------------------

demo = gr.Interface(
    fn=classify_blood_cell,
    inputs=gr.Image(type="pil", label="Upload a blood cell microscopy image"),
    outputs=gr.Label(num_top_classes=8, label="Predicted class"),
    title="ViT Blood Cell Classifier",
    description=(
        f"Fine-tuned `{cfg['model']['checkpoint']}` on "
        f"[{cfg['dataset']['name']}](https://huggingface.co/datasets/{cfg['dataset']['name']}) "
        "(8 classes: basophil, eosinophil, erythroblast, lymphocyte, monocyte, "
        "neutrophil, plasma cell, platelet). For portfolio/demo purposes only — "
        "not a diagnostic tool."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
