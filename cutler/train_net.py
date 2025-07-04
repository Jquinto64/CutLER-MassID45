#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py

"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from config import add_cutler_config
from detectron2.data import MetadataCatalog, build_detection_train_loader
from engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.engine import hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.data import transforms as T
from evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
import data # register new datasets
from data.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from data.dataset_mapper import DatasetMapper
from PIL import Image
import modeling.roi_heads

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, no_segm=cfg.TEST.NO_SEGM))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    # @classmethod
    # def build_train_loader(cls, cfg):
    #     # coco instance segmentation lsj new baseline
    #     # augs = [
    #     #     T.RandomFlip(
    #     #         horizontal=True,
    #     #         vertical=False,
    #     #     ),
    #     #     T.RandomRotation(angle=[90, 180, -90, -180], sample_style = 'choice'),
    #     #     T.ResizeShortestEdge(short_edge_length=(800, 800), max_size = cfg.INPUT.IMAGE_SIZE, sample_style = 'choice', interp=Image.LANCZOS),
    #     #     T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
    #     # ]
    #     # mapper = DatasetMapper(
    #     #     cfg,
    #     #     is_train=True,
    #     #     augmentations=augs
    #     # )
    #     # return build_detection_train_loader(cfg, mapper=mapper)
    #     mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
    #     return build_detection_train_loader(cfg, mapper=mapper)
    
from detectron2.data.datasets import register_coco_instances

def register_custom_coco_dataset(args, dataset_path: str = "/scratch/ssd004/scratch/jquinto/CutLER/datasets/lifeplan/") -> None:
   annotations_path = dataset_path + "annotations/"
   register_coco_instances(
       "lifeplan_train",
       {},
       annotations_path + "instances_train2017.json",
       dataset_path + "train2017",
   )
   if args.eval_only:
    register_coco_instances(
        "lifeplan_valid",
        {},
        annotations_path + "instances_val2017.json",
        dataset_path + "val2017", ## NOTE: we generally do not want to test on the tiled test set
    )
   else: 
    register_coco_instances(
        "lifeplan_valid",
        {},
        annotations_path + "instances_val2017.json",
        dataset_path + "val2017",
    )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_custom_coco_dataset(args)
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # FIXME: brute force changes to test datasets and evaluation tasks
    if args.test_dataset != "": cfg.DATASETS.TEST = ((args.test_dataset),)
    if args.train_dataset != "": cfg.DATASETS.TRAIN = ((args.train_dataset),)
    cfg.TEST.NO_SEGM = args.no_segm
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print(args)
    # args.opts = postprocess_args(args.opts)
    # rint = random.randint(0, 10000)
    # args.dist_url = args.dist_url.replace('12399', str(12399 + rint))
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
