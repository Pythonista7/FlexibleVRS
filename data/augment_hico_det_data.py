import ast

import numpy as np
import torch
import torchvision.ops
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
def show_masked_image(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Handle different mask formats
    if mask.ndim == 2:
        # If mask is already 2D, use it directly
        mask_2d = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:  # (1, H, W) format
            mask_2d = mask.squeeze()
        else:  # (H, W, C) or (C, H, W) format
            mask_2d = mask.mean(axis=-1) if mask.shape[-1] < 5 else mask.mean(axis=0)
    else:
        raise ValueError("Unexpected mask shape")

    # Normalize mask to 0-1 range for visualization
    if mask.dtype != bool:
        mask_2d = (mask_2d - mask_2d.min()) / (mask_2d.max() - mask_2d.min())

    ax[1].imshow(mask_2d, alpha=0.5)
    ax[1].set_title("Mask")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def bbox_hico_to_torch_fmt(bbox):
    fmt_bbox = [bbox[0], bbox[2], bbox[1], bbox[3]]
    return fmt_bbox


def generate_masks(sample, mask_predictor):
    image = np.array(sample['image'])
    mask_predictor.set_image(image)

    all_masks = []

    # Think of this as a list of pairs of human and object bounding boxes
    objects = ast.literal_eval(sample['objects'])

    for obj in objects:
        human_bbox = bbox_hico_to_torch_fmt(obj["bbox_human"])
        object_bbox = bbox_hico_to_torch_fmt(obj["bbox_object"])

        human_mask, h_score, _ = mask_predictor.predict(box=np.asarray(human_bbox), multimask_output=False)
        object_mask, o_score, _ = mask_predictor.predict(box=np.asarray(object_bbox), multimask_output=False)

        all_masks.append(
            {
                "human_mask": human_mask,
                "human_mask_score": h_score,
                "object_mask": object_mask,
                "object_mask_score": o_score
            }
        )

    # Perform NMS here if needed and output nms_masks a dict similar to all_masks
    h_nms_masks = torchvision.ops.nms(
        boxes=torch.tensor([obj["bbox_human"] for obj in objects], dtype=torch.float32),
        scores=torch.tensor([obj["human_mask_score"][0] for obj in all_masks], dtype=torch.float32),
        iou_threshold=0.1  # Picked from the paper https://arxiv.org/pdf/2408.08305
    )

    o_nms_masks = torchvision.ops.nms(
        boxes=torch.tensor([obj["bbox_object"] for obj in objects], dtype=torch.float32),
        scores=torch.tensor([obj["object_mask_score"][0] for obj in all_masks], dtype=torch.float32),
        iou_threshold=0.1  # Picked from the paper https://arxiv.org/pdf/2408.08305
    )

    nms_masks = {
        # picking the first mask from the nms list along with the score
        "nms_human_mask": {k: v for k, v in all_masks[h_nms_masks[0]].items() if k in ["human_mask", "human_mask_score"]} ,
        "nms_object_mask": {k: v for k, v in all_masks[o_nms_masks[0]].items() if k in ["object_mask", "object_mask_score"]} ,
        # TODO: use clip model and the list_actions.csv to get the verb and object names and encode them to generate this training data.
        # "target_object_class": None, # embedding vec from clip model for the object_class encoding
        # "target_subject_class": None, # embedding vec from clip model for the subject_class encoding; default to person for now.
        # "target_predicate_class": None, # embedding vec from clip model for the vnaming class encoding
    }

    text_targets = [
        {
            "caption": f"<s>Person</s><p>{verb}</p><o>{obj}</o>",
            # """
            #     need to use different templates like
            #     "A photo of [predicate-ing]" or “A photo of something [predicate-ing] (something)”
            #     to improve captions for open vocab VRS.
            # """
            "subject_class": "person",
            "object_class": obj,
            "predicate_class": verb
        } for obj, verb in ast.literal_eval(sample["positive_captions"])
    ]

    return all_masks, nms_masks, text_targets


class CustomHicoDetProcessor:
    def __init__(self, model_type="vit_b", checkpoint="data/sam_vit_b_01ec64.pth"):
        self.DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.backends.cuda.is_built() else 'cpu'
        self.MODEL_TYPE = model_type
        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=checkpoint)
        self.sam.to(device=self.DEVICE)
        self.mask_predictor = SamPredictor(self.sam)
        print(
            f"Initialized MaskPredictor: Device : {self.DEVICE} , Model Type : {self.MODEL_TYPE} , Checkpoint : {checkpoint}")

    def process_sample(self, sample):
        # print(f"Processing sample ...")
        try:
            all_masks, nms_masks, text_targets = generate_masks(sample, self.mask_predictor)
        except Exception as e:
            print(f"Error generating mask for sample: {e} , sample: {sample}")
            return None
        # print(f"Completed generating mask for sample!")
        sample['all_masks'] = all_masks
        # sample['nms_masks'] = nms_masks
        # sample['text_targets'] = text_targets

        sample["target"] = {
            "subject_mask": nms_masks["nms_human_mask"]["human_mask"],
            "object_mask": nms_masks["nms_object_mask"]["object_mask"],
            "text_targets": text_targets
        }

        # show_masked_image(sample['image'], nms_masks["nms_human_masks"][0]["human_mask"])
        # show_masked_image(sample['image'], nms_masks["nms_object_masks"][0]["object_mask"])
        #
        # raise NotImplementedError("Implement the rest of the processing here")
        return sample
