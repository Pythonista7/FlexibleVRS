import timm
import torch
from torch import nn


class DaViTPixelDecoder(nn.Module):
    """
    from huggingface timm : https://huggingface.co/timm/davit_base.msft_in1k
    """

    def __init__(self, in_channels):
        super().__init__()
        self.davit = timm.create_model(
            'davit_base.msft_in1k',
            pretrained=True,
            features_only=True,
            in_chans=in_channels)

    def forward(self, input_features):
        out = self.davit(input_features)
        return out


if __name__ == '__main__':
    model = DaViTPixelDecoder(in_channels=3)
    input_image = torch.randn(2, 3, 640, 640)
    features = model(input_image)
    print(features)

"""
    Image Encoder Output = {
        'stride_4': torch.Size([2, 96, 160, 160]), 
        'stride_8': torch.Size([2, 192, 80, 80]), 
        'stride_16': torch.Size([2, 384, 40, 40]), 
        'stride_32': torch.Size([2, 768, 20, 20])
    }
    """

# Pixel decoder output = [
#   torch.Size([2, 128, 160, 160]),
#   torch.Size([2, 256, 80, 80]),
#   torch.Size([2, 512, 40, 40]),
#   torch.Size([2, 1024, 20, 20]),
# ]
