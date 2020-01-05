import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Also implemented a version predicting a standard deviation per bounding box coordinate, following:
    CVPR 2019 paper:
    Bounding Box Regression with Uncertainty for Accurate Object Detection
    by Yihui He, Chenchen Zhu, Jianren Wang. Marios Savvides, Xiangyu Zhang

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, modeltype, base, extras, head, num_classes, default_forward, merging_method, sampling_strategy, sample_select_forward, sample_select_nms_conf_thresh, cfg,forward_vgg_base_only):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg, modeltype)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())

        # todo: convert to x1y1x2y2 format here if necessary


        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.modeltype = modeltype
        if self.modeltype == 'SSD300KL':
            self.loc_std = nn.ModuleList(head[2])
        self.conf = nn.ModuleList(head[1])
        if self.modeltype in ['SSD300','SSD300KL']:
            self.size = 300
        else:
            raise NotImplementedError()

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)


            if sample_select_forward and merging_method in ['bsas','hbdscan','pre_nms_avg']:
                conf_tresh = sample_select_nms_conf_thresh # merging of boxes can be expensive, to have less boxes, we can apply a more agressive conf treshold
            else:
                conf_tresh = 0.01
            # Active Learning parameters added to enable experiments with and usage of Active Learning
            self.detect = Detect(num_classes, 0, 200, conf_tresh, 0.45, # default values in paper: num_classes,0,200,0.01,0.45
                                 default_forward,
                                 merging_method,
                                 sampling_strategy,
                                 modeltype)

        self.forward_vgg_base_only = forward_vgg_base_only
        

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        sources = list()
        loc = list()
        conf = list()
        if self.modeltype == 'SSD300KL':
            loc_std = list()
        # apply vgg up to conv4_3 relu
        for k in range(23):
            # print('debug: apply vgg')
            x = self.vgg[k](x)

        if self.forward_vgg_base_only:
            return x
        # TODO: Why apply L2norm already? => because conv4_3 has larger scale than the rest
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7 TODO: Why FC layers? => Doesn't use FC layers, UP TO FC layers..
        for k in range(23, len(self.vgg)):
            # print('debug2: apply vgg')
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            # print('debug3: apply extra layers')
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1: #TODO: Why only every second layer of the extra layers? => because thats how the paper states it. It has conv blocks of 2 conv layers
                sources.append(x)

        if self.modeltype != 'SSD300KL':
            # apply multibox head to source layers
            for (x, l, c) in zip(sources, self.loc, self.conf):
                # print('debug4: apply multibox head')
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print('debug foward 1')
            if self.phase == "test":
                # if self.sampling_strategy != 'p-max_localization-stability' :
                output = self.detect(loc.view(loc.size(0), -1, 4),  # loc preds
                                     self.softmax(conf.view(conf.size(0), -1,self.num_classes)),  # conf preds
                                     self.priors.type(type(x.data)),  # default boxes
                                     )
                # else:
                #     output = self.detect()

                # training phase => no merging or other forwards used
            else:
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors
                )
        else:
            # apply multibox head to source layers
            for (x, l, c, std) in zip(sources, self.loc, self.conf, self.loc_std):
                # print('debug4: apply multibox head')
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
                loc_std.append(std(x).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            loc_std = torch.cat([o.view(o.size(0), -1) for o in loc_std], 1)

            if self.phase == "test":
                # during training alpha = log(sigma^2), during testing, this needs to be converted back
                loc_std = torch.exp(loc_std)

                output = self.detect(loc.view(loc.size(0), -1, 4),  # loc preds
                                     self.softmax(conf.view(conf.size(0), -1,self.num_classes)),  # conf preds
                                     self.priors.type(type(x.data)),  # default boxes
                                     torch.abs(loc_std.view(loc_std.size(0), -1, 4)) # alphas (predicted log of std deviations of loc preds)
                                     )
            else:
                # during training, alpha = log(sigma^2) is predicted
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                    torch.abs(loc_std.view(loc_std.size(0), -1, 4)) #alphas
                )

        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C': #TODO: ceil mode not used in https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py => impacts output shape
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # A TROUS algorithm (dilated conv)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers



def multibox(vgg, extra_layers, cfg, num_classes, model_type):
    #cfg = number of boxes per feature map location

    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    if model_type != 'SSD300KL':
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3,
                                     padding=1)]  # 4 is for the 4 corners of the bounding box
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3,
                                      padding=1)]  # out = #boxes*classes (per feature map)

        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)

    else:
        """
        Also predict a standard deviation per bounding box coordinate, from CVPR 2019 paper:
        Bounding Box Regression with Uncertainty for Accurate Object Detection
        by Yihui He, Chenchen Zhu, Jianren Wang. Marios Savvides, Xiangyu Zhang
        """
        loc_std_layers = []
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3,
                                     padding=1)]  # 4 is for the 4 corners of the bounding box
            loc_std_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3,
                                     padding=1)]  # 4 is for the 4 corners of the bounding box

            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3,
                                      padding=1)]  # out = #boxes*classes (per feature map)

        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]

            loc_std_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]

            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]

        return vgg, extra_layers, (loc_layers, conf_layers, loc_std_layers)


# 300D is SSD300 with dropout layers to be able to make Bayesian using MC-Dropout
# TODO: upconvolution first and then downconvolution?? NOPE => zijn de channels!
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
    # '300D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512,'D', 'M',
    #         512, 512, 512,'D']
}
#todo: should the dropout layers be inbetween base and extra? and also between
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
    '300D': [256, 'S', 512, 128, 'D', 'S', 256, 128, 256, 128, 256]
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
    # '300D': [4, 6, 6, 6, 4, 4]
}


def build_ssd(phase, model_type='SSD300', num_classes=21, default_forward = True, merging_method = None, sampling_strategy = None, sample_select_forward = False, sample_select_nms_conf_thresh = None, cfg = None, forward_vgg_base_only = False):
    " Active learning parameter here is the sample selection part"

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if model_type not in ['SSD300','SSD300KL']:
        print("ERROR: You specified size " + repr(model_type) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    if model_type in ['SSD300','SSD300KL']:  # if wished add other SSD models with input dim 300 to this list
        size = 300

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes, model_type) #cfg
    return SSD(phase, model_type, base_, extras_, head_, num_classes, default_forward, merging_method, sampling_strategy, sample_select_forward, sample_select_nms_conf_thresh, cfg,forward_vgg_base_only)
