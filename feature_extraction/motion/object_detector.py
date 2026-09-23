import os

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Predictor:
    def __init__(self, confidence_threshold=0.5, weights_path=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model_final_f10217.pkl")
        self.cfg.MODEL.WEIGHTS = weights_path

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, img):
        """Run detection on a single image and return (boxes[N,4], classes[N])."""
        if torch.is_tensor(img):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0).detach().cpu().numpy()
            else:
                img = img.detach().cpu().numpy()

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        with torch.no_grad():
            outputs = self.predictor(img)
            instances = outputs["instances"]
            bboxes = instances.get("pred_boxes").tensor
            classes = instances.get("pred_classes")

        return bboxes, classes
