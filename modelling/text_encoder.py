from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig


class TextEncoder:
    def __init__(self, projection_dim, hidden_size=512):
        self.clip_model = CLIPTextModel(
            config=CLIPTextConfig(projection_dim=projection_dim, hidden_size=hidden_size)
        )
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
