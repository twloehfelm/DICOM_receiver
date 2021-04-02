# from https://towardsdatascience.com/detectron2-the-basic-end-to-end-tutorial-5ac90e2f90e3

# version inspection
import detectron2
print(f"Detectron2 version is {detectron2.__version__}")

# import some common detectron2 utilities
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T
from detectron2.utils.visualizer import GenericMask
import cv2
import requests
import numpy as np
import pydicom
import torch, torchvision
from pathlib import Path
import os

class LiverPredictor:
  def __init__(self, cfg):
    self.cfg = cfg
    self.model = build_model(self.cfg)
    self.model.eval()

    liver_metadata = Metadata()
    liver_metadata.set(thing_classes = ['liver'])
    self.metadata = liver_metadata

    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    self.transform_gen = T.ResizeShortestEdge(
      [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    self.input_format = cfg.INPUT.FORMAT

  def __call__(self, original_image):
    """
    Args:
      original_image (np.ndarray): a single channel image.
    Returns:
      predictions (dict):
        the output of the model for one image only.
        See :doc:`/tutorials/models` for details about the format.
    """
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
      height, width = original_image.shape[:2]
      #image = self.transform_gen.get_transform(original_image).apply_image(original_image)
      image = original_image
      image = torch.as_tensor(image.astype("float32"))

      inputs = {"image": image, "height": height, "width": width}
      predictions = self.model([inputs])[0]
      return predictions

def prepare_predictor():
  #create config
  cfg = get_cfg()
  # below path applies to current installation location of Detectron2
  cfgFile = "/usr/local/lib/python3.9/site-packages/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
  cfg.merge_from_file(cfgFile)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80  # set threshold for this model
  cfg.MODEL.WEIGHTS = "/app/model_final.pth"
  cfg.MODEL.DEVICE = "cpu"  # we use a CPU Detectron copy
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (liver)
  cfg.INPUT.FORMAT = "F" #32-bit single channel floating point pixels
  cfg.INPUT.MASK_FORMAT = "bitmask" # Needed to change this from the default "polygons"

  # create predictor
  predictor = LiverPredictor(cfg)
  classes = predictor.metadata.thing_classes

  return (predictor, classes)
