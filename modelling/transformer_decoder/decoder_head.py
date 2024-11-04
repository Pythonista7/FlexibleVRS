from torch import nn


class DecoderHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        pass