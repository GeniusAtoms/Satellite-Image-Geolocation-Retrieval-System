"""
CLIP image embedding module.
Converts any image into a 512-dim L2-normalized float32 vector.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling object
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_image_batch(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling object
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
