import torch
import torch.nn as nn
from transformers import FocalNetModel, FocalNetConfig


class FocalNetMultiScale(nn.Module):
    def __init__(self, image_size=640, num_channels=3, embed_dim=96):
        super(FocalNetMultiScale, self).__init__()

        # Load the FocalNet backbone with desired configuration
        config = FocalNetConfig(image_size=image_size, num_channels=num_channels, embed_dim=embed_dim, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],output_hidden_states=True)
        self.backbone = FocalNetModel(config)

    def forward(self, x):
        # Forward through the backbone
        outputs = self.backbone.forward(x)

        # outputs is a list containing features from different layers (resolution decreasing)
        stride_features = outputs[3]  # Fourth stage has the feature maps we need

        return {
            "stride_4": stride_features[0],  # Low-resolution feature at stride 4
            "stride_8": stride_features[1],  # Feature map at stride 8
            "stride_16": stride_features[2],  # Feature map at stride 16
            "stride_32": stride_features[3]  # Feature map at stride 32
        }


# Example usage
# if __name__ == '__main__':
#     model = FocalNetMultiScale(image_size=640, num_channels=3, embed_dim=96)
#     input_image = torch.randn(2, 3, 640, 640)
#     multi_scale_features = model(input_image)
#     print({k: v.shape for k, v in multi_scale_features.items()})
    """
    {
        'stride_4': torch.Size([2, 96, 160, 160]), 
        'stride_8': torch.Size([2, 192, 80, 80]), 
        'stride_16': torch.Size([2, 384, 40, 40]), 
        'stride_32': torch.Size([2, 768, 20, 20])
    }
    """
