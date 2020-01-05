from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, VOC_ROOT_LOCAL

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    # cv2.resize: default uses linear interpolation, and doesnt preserve aspact ratio: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#void%20resize(InputArray%20src,%20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,%20int%20interpolation)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        # if modeltype == 'SSD300':
        #     size = 300
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
