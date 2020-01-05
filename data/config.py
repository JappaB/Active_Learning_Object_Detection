# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123) # channel means for Pascal VOC

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'dataset_mean': (104, 117, 123),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2], # Not discussed in paper, see blogpost: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    'clip': True,
    'name': 'VOC',
}


voc_1_class = {
    'num_classes': 2,
    'dataset_mean': (104, 117, 123),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2], # Not discussed in paper, see blogpost: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    'clip': True,
    'name': 'VOC_1_class',
}

voc_6_class = {
    'num_classes': 7,
    'dataset_mean': (104, 117, 123),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2], # Not discussed in paper, see blogpost: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    'clip': True,
    'name': 'VOC_6_class',
}
