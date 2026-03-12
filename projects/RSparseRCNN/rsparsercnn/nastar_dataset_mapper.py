# NASTaR Dataset Mapper for R-Sparse R-CNN
# Adapted from dataset_mapper.py by Kamirul Kamirul
#
# This mapper handles GeoTIFF (.tif) images from the NASTaR dataset.
# NASTaR uint8 TIFs are single-channel (grayscale) and need to be
# converted to 3-channel (RGB) for the ResNet backbone.

import copy
import logging
import os

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BoxMode

__all__ = ["NASTaRDatasetMapper"]

logger = logging.getLogger(__name__)


def read_tif_image(file_name):
    """
    Read a GeoTIFF image and return as a 3-channel uint8 numpy array (H, W, 3).
    Handles both uint8 and uint16/float32 TIFs.
    """
    try:
        import tifffile
        img = tifffile.imread(file_name)
    except ImportError:
        # Fallback to PIL if tifffile not available
        from PIL import Image
        img = np.array(Image.open(file_name))

    # Handle different dtypes
    if img.dtype == np.float32 or img.dtype == np.float64:
        # Float backscatter data — normalise to 0-255
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == np.uint16:
        # 16-bit — scale to 8-bit
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to 3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    return img


def rotate_bbox(annotation, transforms):
    annotation["bbox"] = transforms.apply_rotated_box(
        np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def get_shape_augmentations(cfg):
    return [
        T.ResizeShortestEdge(
            short_edge_length=(128, 800),
            max_size=1333,
            sample_style='range'
        ),
        T.RandomFlip(),
    ]


class NASTaRDatasetMapper:

    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.cfg = cfg

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        file_name = dataset_dict["file_name"]

        # Use custom TIF reader for .tif files, standard reader otherwise
        if file_name.lower().endswith((".tif", ".tiff")):
            image = read_tif_image(file_name)
        else:
            image = utils.read_image(file_name, format=self.img_format)

        image, image_transforms = T.apply_transform_gens(
            get_shape_augmentations(self.cfg), image)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        annotations = [
            rotate_bbox(obj, image_transforms)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances_rotated(
            annotations, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict