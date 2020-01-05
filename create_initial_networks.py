import argparse
import os
import torch

import active_learning_package.helpers as helpers
from data import config

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# parser arguments from train.py
parser = argparse.ArgumentParser(description='Active Learning With Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--modeltype', default='SSD300',choices=['SSD300', 'SSD300KL'],
                    help='Which model to use: standard SSD or the SSD with  uncertainty in the bounding box regression and KL loss ') #SSD300KL doesn't work well
parser.add_argument('--dataset', default='VOC07_1_class', choices=['VOC07', 'VOC12','VOC07_1_class','VOC07_1_class','VOC07_6_class'],
                    type=str, help='VOC07_1_class is with one class of interest and the background class')
parser.add_argument('--sample_select_nms_conf_thresh', default = 0.01, type = float,
                    help = 'The conf threshold used in before non maximum suppression. Only detections with a confidence above '
                           'this threshold for a certain class will go trough nms')
parser.add_argument('--paths_to_weights', default=None,type=str, nargs='+',
                    help='These are the weights that ere used the initial evaluation of the unlabeled dataset') # if no trained model is given, this will return an error when loading the model.
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')

parser.add_argument('--ensemble_size', default=3,type=int)
parser.add_argument('--num_classes', default=1,type=int,
                    help='number of classes of interest (so excluding background class')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset in ['VOC12','VOC07']:
        args.cfg = config.voc # adapted from pytorch SSD code

    elif args.dataset == 'VOC07_1_class':
        args.cfg = config.voc_1_class

    elif args.dataset == 'VOC07_6_class':
        args.cfg = config.voc_6_class

    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'
        torch.set_default_tensor_type('torch.FloatTensor')

    args.device = device

    args.num_classes = args.num_classes + 1
    print('Creating ',args.ensemble_size,' number of SSDs for ',args.num_classes,' (+ 1 background class) number of classes')
    print('...')




    for i in range(args.ensemble_size):
        # make net
        net = helpers.build_sample_selection_net(args,
                                           args.num_classes)

        args.experiment_dir = os.getcwd()+'/'
        path = 'weights/initial_net_'+str(i)

        # save net
        helpers.save_weights(weights=net,
                             args=args,
                             path=path)
        print()


    print('Initial nets created!')