# from typing import Tuple, List
#
# import torch
# from transformers import AutoImageProcessor, FocalNetModel, FocalNetConfig
# from torch import nn
#
#
# class FocalNetImageEncoder(nn.Module):
#
#     def __init__(self):
#         """
#
#         :param mean: List containing the #channels mean values for the dataset, e.g. [0.485, 0.456, 0.406] for RGB image and for b/w images [0.5]
#         :param std: List containing the #channels std values for the dataset, e.g. [0.229, 0.224, 0.225] for RGB image and for b/w images [0.5]
#         """
#         super().__init__()
#         self.focal_net_config = FocalNetConfig(
#             image_size=640,
#             encoder_stride=[4, 8, 16, 32],
#             out_features=["stage1", "stage2", "stage3", "stage4"],
#
#         )
#         self.model = FocalNetModel(config=self.focal_net_config)
#         self.image_processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny")
#
#     def prepare(self, image):
#         inputs = self.image_processor(image, return_tensors="pt")
#         return inputs
#
#     def forward(self, inputs):
#         # Just Inference
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         last_hidden_states = outputs.last_hidden_state
#         return last_hidden_states
#
#
# if __name__ == '__main__':
#     model = FocalNetImageEncoder()
#
#     rand_img = torch.randn(3, 640, 640)
#     # TODO: normalize the pixel values between 0 and 1 using mean and std of the dataset
#     rand_img = (rand_img - rand_img.min()) / (rand_img.max() - rand_img.min())
#     inputs = model.prepare(rand_img)
#     preds = model(inputs)
#     print(preds.shape)
