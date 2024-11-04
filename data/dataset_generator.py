import ast

from datasets import load_dataset

from augment_hico_det_data import CustomHicoDetProcessor

SAMPLE_SIZE = 1
# Load up the dataset
train_hico_det_dataset = load_dataset(path="zhimeng/hico_det", split="train").select(range(SAMPLE_SIZE))
mask_predictor = CustomHicoDetProcessor()


def check_valid_data(sample, idx):
    # Missing bounding boxes or object values
    for obj in ast.literal_eval(sample["objects"]):
        if any([v is None or v == [] for v in obj.values()]):
            print(f"Sample has missing values: {idx}")
            return False

    if sample["image"] is None or ast.literal_eval(sample["size"])[-1] != 3:
        print(f"Sample has missing image or a black and white image: {idx}")
        return False
    return True


def split_row(sample):
    new_samples = []
    target_texts = sample["target"]["text_targets"]
    for tt in target_texts:
        new_sample = sample.copy()
        new_sample["text"] = tt["caption"]
        new_sample["subject_text"] = tt["subject_class"]
        new_sample["object_text"] = tt["object_class"]
        new_sample["predicate_text"] = tt["predicate_class"]
        new_samples.append(new_sample)

    new_samples_dict = {k: [v[k] for v in new_samples] for k in new_samples[0].keys()}
    return new_samples_dict

"""
TODO: 
> remove redundancy in the nms_masks, currently both human and object masks are being stored in the same dict with duplications.
> Convert the mask from bool to int for saving to diskspace of the dataset.
"""


def generate_nms_mask_dataset(dataset):
    print(f"Generating NMS masked dataset from {len(dataset)} samples")
    nms_dataset = (
        dataset
        .filter(
            lambda x, idx: check_valid_data(x, idx), with_indices=True
        )
        .map(
            lambda x: mask_predictor.process_sample(x)
        )
    )
    nms_dataset.save_to_disk(F"data/small_{SAMPLE_SIZE}_train_hico_det_masked_dataset")
    return True


generate_nms_mask_dataset(
    dataset=train_hico_det_dataset
)
