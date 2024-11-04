import torch
from torch import nn
from torchvision.models import swin_v2_t
from torchviz import make_dot


class SwinImageEncoder(nn.Module):
    """
    From torch vision : https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html
    """

    def __init__(self):
        super().__init__()
        self.backbone = swin_v2_t()

    def forward(self, x):
        features = self.backbone(x.float())
        return features


if __name__ == '__main__':
    model = SwinImageEncoder()
    rand_img = torch.randn(1, 3, 640, 640)
    preds = model(rand_img)
    # make_dot(preds, params=dict(list(model.named_parameters()))).render("swin", format="png")
    print(preds.shape)  # [B, 1000] , [batch_size, output_dim_of_swin_v2]
