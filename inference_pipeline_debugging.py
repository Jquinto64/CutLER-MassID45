import logging
from typing import List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)
import sys
sys.path.insert(0, "/scratch/ssd004/scratch/jquinto/CutLER")
sys.path.insert(0, "/scratch/ssd004/scratch/jquinto/CutLER/cutler")

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import modeling
from config import add_cutler_config

from detectron2.modeling import ROI_HEADS_REGISTRY
# ROI_HEADS_REGISTRY.register(modeling.CustomCascadeROIHeads)
import cv2

cfg = get_cfg()
cfg.set_new_allowed(True) 
add_cutler_config(cfg)
cfg.merge_from_file("/scratch/ssd004/scratch/jquinto/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml")
cfg.MODEL.WEIGHTS = "/scratch/ssd004/scratch/jquinto/CutLER/cutler_cascade_final.pth"

# set model device
# cfg.MODEL.DEVICE = "self.device.type"
cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.freeze()

# init predictor
model = DefaultPredictor(cfg)
category_mapping={"1": "b"}

# detectron2 category mapping
category_names = list(category_mapping.values())
image = np.array(
    cv2.imread(
        "/h/jquinto/MaskDINO/datasets/lifeplan_256/val2017/GRLLN6_0_824_2163_1080_2419.png", 
        cv2.IMREAD_COLOR
))

# if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
#     # convert RGB image to BGR format
#     image = image[:, :, ::-1]

prediction_result = model(image)
# print(prediction_result)
# original_predictions = prediction_result

# PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
print(model.aug)
print()

# Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
print(model.cfg.INPUT.IMAGE_SIZE)
print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
print(sample_preds)
print(sample_preds.shape)
print(sample_preds.dtype)

"""
>>> print(prediction_result['instances'][0]._fields['pred_masks'])
tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0')
>>> print(prediction_result['instances'][0]._fields['pred_masks'].size())
torch.Size([1, 256, 256])

- Note: we can see here that the output mask predictions are a float32 array that 
        is the same size as the sample image that we are predicting on.
>>> np.unique(sample_preds)
array([0., 1.], dtype=float32)
- Appears to be in RLE format -> indicating we have an RLE mask that by definition is pixel-based,
so we can't have sub-pixel mask coordinates:

The process is:

Upsample by 5x and round to nearest integer using +.5 trick
Get dense boundary points at this higher resolution
Downsample back by dividing by scale
Apply floor/ceil and boundary checks
Convert to final integer coordinates
So decimal coordinates are first scaled up for better precision during boundary calculation, but ultimately get converted to integers through this upscale-then-downscale process with rounding.

This explains why super-resolution could help - it effectively increases the resolution at which this rounding occurs, allowing for more precise boundary definitions.


"""