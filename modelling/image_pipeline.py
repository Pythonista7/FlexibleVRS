import torch
from fvcore.nn import weight_init
from torch import nn

from modelling.backbone.davit import DaViTPixelDecoder
from modelling.backbone.focalnet_backbone import FocalNetMultiScale


class FleVRSImageEncoderDecoder(nn.Module):
    def __init__(
            self,
            num_queries=100,
            d_model=640,
            conv_dim=1024,  # This is the output dimension of the pixel decoder (1,1024,1,1)
            mask_dim=640
    ):
        super().__init__()
        self.num_queries = num_queries

        self.image_encoder = FocalNetMultiScale()
        self.pixel_decoder = DaViTPixelDecoder(in_channels=768)
        self.text_features = None
        self.features = None
        self.outputs = None

        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        weight_init.c2_xavier_fill(self.mask_features)

    def forward_features(self, features):
        # Forward through the image encoder
        image_features = self.image_encoder(features)

        # Forward through the pixel decoder
        pixel_decoder_outputs = self.pixel_decoder(image_features["stride_32"])

        # Process pixel decoder outputs
        multi_scale_features = []
        for i, feat in enumerate(pixel_decoder_outputs):
            if i < 3:  # Assuming you want to use the first 3 scales
                multi_scale_features.append(feat)

        # Generate mask features
        mask_features = self.mask_features(pixel_decoder_outputs[-1])

        return mask_features, pixel_decoder_outputs[0], multi_scale_features

    def forward(self, x):
        # features = self.image_encoder(x)
        mask_features, transformer_encoder_features, multi_scale_features = self.forward_features(x)
        return mask_features, transformer_encoder_features, multi_scale_features


if __name__ == '__main__':
    model = FleVRSImageEncoderDecoder()
    input_image = torch.randn(1, 3, 640, 640)
    mask_features, transformer_encoder_features, multi_scale_features = model(input_image)
    print(mask_features.shape)
    print(transformer_encoder_features.shape)
    print([f.shape for f in multi_scale_features])
