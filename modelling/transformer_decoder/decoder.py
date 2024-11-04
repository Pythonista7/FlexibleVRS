from torch import nn
from torch.nn import Conv2d
import fvcore.nn.weight_init as weight_init

from position_encoding import PositionEmbeddingSine




class SelfAttention(nn.Module):
    def __init__(self, d_model: int, no_of_heads: int):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(d_model, no_of_heads)

    def forward(self, x):
        out = self.attn_block(x, x, x)
        out = out + x
        out = nn.LayerNorm(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, no_of_heads: int):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(d_model, no_of_heads)

    def forward(self, q, k, v):
        """
        :param q: Image features
        :param k: Image features
        :param v: Query features
        :return:
        """
        x = self.attn_block(q, k, v)
        x = x + v
        x = nn.LayerNorm(x)
        return x


class DecoderUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = None

    def _init_layers(self, d_model: int, no_of_heads: int):
        self.layers = nn.ModuleList()
        self.layers.append(
            CrossAttention(d_model, no_of_heads)
        )
        self.layers.append(
            SelfAttention(d_model, no_of_heads)
        )

    def forward(self):
        pass


class RelationshipDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            no_of_layers: int,
            in_channels: int,
            enforce_input_project: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.no_of_layers = no_of_layers
        self.layers: nn.ModuleList

        # use these against the pixel features
        n_steps = d_model // 2
        self.sin_pos_embedding_for_pixel_features = PositionEmbeddingSine(n_steps, normalize=True)
        self.scale_embedding_for_pixel_features = nn.Embedding(self.num_feature_levels, d_model)  # How to use this?

        # level embedding (we always use 3 scales) , i.e: we use 3 scale levels of features from the pixel decoder
        # which is used in the transformer
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, d_model)

        # Text features
        self.num_of_latent_queries = 100
        # learnable query features
        self.query_feat = nn.Embedding(self.num_of_latent_queries, d_model)
        # learnable query p.e.
        self.query_embed = nn.Embedding(self.num_of_latent_queries, d_model)

        # Will be used to project the input features to the d_model size
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != d_model or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, d_model, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

    def _init_layers(self, d_model: int, no_of_heads: int):
        self.layers = nn.ModuleList()
        for i in range(self.no_of_layers):
            # First Cross Attention as used in Mask2Former
            self.layers.append(
                nn.MultiheadAttention(d_model, no_of_heads)
            )
            # Then Self Attention
            self.layers.append(
                nn.MultiheadAttention(d_model, no_of_heads)
            )

    # TODO: Assuming the mask features are output of the pixel decoder
    def forward(self, text_features, multi_scale_pixel_decoder_features, mask_features):
        if self.layers is None or len(self.layers) == 0:
            self._init_layers(self.d_model, no_of_heads=5)

        positional_encoding_for_pixel_features = self.sin_pos_embedding_for_pixel_features(
            multi_scale_pixel_decoder_features)
        scale_level_embedding_for_pixel_features = self.scale_embedding_for_pixel_features(
            multi_scale_pixel_decoder_features)

        # Unpacking list 2 at a time with index
        for layer_idx, cross_attn_layer, self_attn_layer in enumerate(zip(self.layers[::2], self.layers[1::2])):
            level_idx = layer_idx % self.num_feature_levels
            pixel_feature_for_level = multi_scale_pixel_decoder_features[level_idx]
            x = cross_attn_layer(pixel_feature_for_level, pixel_feature_for_level, text_features)
            # Add Pixel Decoder features for that level
            x += multi_scale_pixel_decoder_features[0]

        pass
