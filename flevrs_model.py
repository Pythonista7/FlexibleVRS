import torch
from torch import nn

from modelling.image_pipeline import FleVRSImageEncoderDecoder
from modelling.relationship_decoder import Mask2FormerDecoder
from modelling.text_encoder import TextEncoder


class FleVRS(nn.Module):
    def __init__(self, num_queries=100, d_model=640):
        super().__init__()
        self.num_queries = num_queries

        self.image_pipeline = FleVRSImageEncoderDecoder()
        self.text_encoder = TextEncoder(projection_dim=d_model)
        self.relationship_decoder = Mask2FormerDecoder(
            dim=d_model,
            ffn_dim=2048
        )
        self.latent_query = nn.Embedding(num_queries, d_model)

    def forward(self, input_image, input_text=None, training=False):
        mask_features, transformer_encoder_features, multi_scale_features = self.image_pipeline(input_image)
        queries = self.latent_query
        if input_text != None:
            text_tokens = self.clip_tokenizer(input_text, return_tensors="pt", padding=True,
                                              truncation=True).input_ids.to(
                input_image.device)
            text_features = self.clip_model(text_tokens).last_hidden_state
            queries = torch.cat([queries, text_features.permute(1, 0, 2)], dim=0)

        out = self.relationship_decoder.forward(queries, multi_scale_features)
        return out


if __name__ == "__main__":
    model = FleVRS()
