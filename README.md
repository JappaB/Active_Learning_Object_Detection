# Active Learning for Object Detection With Localization Uncertainty from Sampling-Based Probabilistic Bounding Boxes
In this repo the code belonging to my master thesis titled: "Active Learning for Object Detection With Localization Uncertainty from Sampling-Based Probabilistic Bounding Boxes" can be found. I also uploaded the pdf of my thesis [here](https://github.com/JappaB/Active_Learning_Object_Detection/blob/master/Thesis_Jasper_Bakker_Active_Deep_Learning_for_Object_Detection_With_Sampling_Based_Probabilistic_Bounding_Boxes_compressed.pdf). As a very short summary: I researched the use of a localization uncertainty, obtained trough an ensemble of object detectors to select more informative images to be labeled. It shows promissing results on Pascal VOC 2007, but has not been used on other datasets. Please let me know your experiences if you use it on different datasets.

As a basis for my repository I used the excellent repository by Max de Groot and Ellis Brown [PyTorch implementation of the SSD detector](https://github.com/amdegroot/ssd.pytorch), retrieved on 19-02-2019. However, as I used the then newest stable version of PyTorch (1.0.1), I did change some of their code to be able to run it. Note that their repo is probably more suitable if you just want to use an SSD written in PyTorch and don't want to perform acive learning. Some parts of this readme are directly copy-pasted from Max de Groot and Ellis Brown their repo as my work is built upon their code anyways.

After finishing my thesis, in order to make it more useable for others, I cleaned the code a bit and wrote this readme. I hope this helps, however, bear in mind that the code is research code and should be viewed as such. Currently I'm traveling trough Central and South America. I know the code is still not very beautiful, please post issues if you are serious about using it and don't understand certain parts. I'll see what I can do when I'm back.


### Table of Contents
- <a href='#get_started'>Getting Started</a>
- <a href='#datasets'>Datasets</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;


## SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>


## Getting started
- I supplied a list of the conda environment I used for my experiments in the [requirements](https://github.com/JappaB/Active_Learning_Object_Detection/blob/master/requirements) file for reproducability. The most important packages are probably: PyTorch, NumPy, SciPy, cv2 and hdbscan. 
- Clone this repository.
- Then download the dataset by following the [instructions](#datasets) below. Note that the Active Learning code has only been completely implemented for the Pascal VOC 2007 dataset.
- As the SSD uses a reduced VGG-16 backbone, download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:	https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth and put them in a directory called 'weights'
- By default, we assume you have downloaded the file in the `Active_Learning_Object_detection/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
- Note that a GPU is highly recommended for training the SSD.

- As there are many parser argument options, I provide two sample run scripts in the [run_scripts](https://github.com/JappaB/Active_Learning_Object_Detection/tree/master/run_scripts/scripts) directory to get a headstart. I provided one for the six classes I used in my experiments and one for a single class of interest (thus background vs non-background). To use them, you also need to copy the appropriate imageset files to the imageset folder. You can find the imageset files in `data/imageset_files` and they need to be copied to `~/data/VOCdevkit/VOC2007/ImageSets/Main/`.

- You are required to give a list of paths to the currently best networks. I provided a script `create_initial_networks.py` to generate these if you don't have any yet. The current settings of this script correspond to the sample run script with the single class. NOTE: A single saved network requires approximately 100MB of storage. Make sure you have enough room before running the script.

- Finally, if you don't want to use one of the provided scripts, the entry point for active learning is the `active_learning_main.py` file. 


## What can I find where?
For active learning the two most important folders are the `active_learning_dir` and `active_learning_package`. In the first the (intermediate) results of the runs (e.g. which images to label next) will be saved and in the second the code for the active learning can be found.


## Datasets
To make things easy, we provide bash scripts to handle the dataset (Pascal VOC) downloads  and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### Use a pre-trained SSD network for detection
#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth

## Authors
Active learning part:
* [**Jasper Bakker**](https://github.com/jappab)

SSD, Dataloaders, etc. (check their excellent repo at [PyTorch implementation of the SSD detector](https://github.com/amdegroot/ssd.pytorch)):
* [**Max deGroot**](https://github.com/amdegroot)
* [**Ellis Brown**](http://github.com/ellisbrown)

