# Copyright (c) Meta Platforms, Inc. and affiliates.

import config
import engine
import modeling
import structures
import tools
import demo 

# dataset loading
from . import data  # register all new datasets
from data import datasets  # register all new datasets
# from data.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

from solver import *

# from .data import register_all_imagenet