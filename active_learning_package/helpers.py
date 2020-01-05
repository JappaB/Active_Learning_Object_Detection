import json
import os
import random
import copy
import time
from collections import OrderedDict
import pickle
import random

import pandas as pd

from data import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torch.utils.data import Sampler
import torchvision.models as models


from utils import augmentations
from data.voc0712 import VOCAnnotationTransform2

def write_labeled(args,
                  # annotations,
                  image_idxs):
    """
    helper for writing the list of labeled idx and/or bounding boxes from the
    labeled and unlabeld json files.

    If we are annotating all objects, the unlabeled_idx_file should have


    :param unlabeled_idx_file:
    :param annotate_all_objects:
    :return:
    """

    # Create the labeled_idx file
    if not os.path.isfile(args.path_to_labeled_idx_file) and not args.seed_set_file:
        # Create the dictionary
        label_dict = {}
        label_dict['dataset_name'] = args.dataset

        # (ordered) lists to be able to link image idx and list id
        label_dict['image_idx_list'] = list(image_idxs)

        # Write to file
        with open(args.path_to_labeled_idx_file, "w+") as json_file:
            # Convert into JSON and write to outfile
            json.dump(label_dict, json_file)

    else:
        # Load current json (using read labeled) as python dict
        label_dict = read_labeled(args.path_to_labeled_idx_file,
                                  args.annotate_all_objects,
                                  args.dataset,
                                  args)

        # Append new labels and bounding boxes to the dict
        # (ordered) lists to be able to link image id and list id
        label_dict['train_set'] += image_idxs

        # Write to file
        with open(args.path_to_labeled_idx_file, "w") as json_file:
            # Convert into JSON and write to outfile
            json.dump(label_dict, json_file)
            print('Written new labels to labeled dict json file')


def read_labeled(path_to_labeled_idx_file,
                 annotate_all_objects,
                 dataset_name,
                 args):
    """:
    helper for loading the list of labeled idx and/or bounding boxes from the
    labeled and unlabeld json files.

    If we are annotating all objects, the unlabeled_idx_file should have
    """

    # Open file and load the current labeled if it exists already
    if not os.path.isfile(path_to_labeled_idx_file) and not args.seed_set_file:
        raise NotImplementedError()
        # I already have seed set, but in slightly different format (with an extra dict)

        # # return an empty structure labeled_dict structure
        # # Create the dictionary
        # label_dict = {}
        # label_dict['dataset_name'] = dataset_name
        #
        # # flag if all objects are annotated in this experiment, if so, the bounding boxes are just the hardest ones
        # label_dict['annotate_all_objects'] = args.annotate_all_objects
        #
        # # (ordered) lists to be able to link image idx and
        # label_dict['image_idx_list'] = []
        # label_dict['val_set'] = []


    elif not os.path.isfile(path_to_labeled_idx_file) and args.seed_set_file:
        with open(args.seed_set_file) as json_file:
            label_dict = json.load(json_file)
    else:
        with open(path_to_labeled_idx_file) as json_file:
            label_dict = json.load(json_file)

    return label_dict

def save_detections(args,
                    output, # detections
                    ensemble_idx,
                    num_unlabeled_images):

    path = args.experiment_dir + 'sample_selection/al-iteration_' + str(args.al_iteration) + '_detections_.pickle'

    if not os.path.exists(path):
        if args.merging_method == 'pre_nms_avg':
            detections = {}
            detections['loc_data'] = torch.zeros(args.ensemble_size, num_unlabeled_images, 8732, 4, device='cpu')
            detections['conf_data'] = torch.zeros(args.ensemble_size, num_unlabeled_images,8732, args.cfg['num_classes'], device='cpu')
        else:
            detections = {}
            detections['detections'] = torch.zeros(args.ensemble_size, num_unlabeled_images, args.cfg['num_classes'], 200, args.cfg['num_classes'] + 4 , device = 'cpu')
            detections['num_boxes_per_class'] = torch.zeros(args.ensemble_size, num_unlabeled_images, args.cfg['num_classes'], device = 'cpu')
    # open detection file and append
    else:
        detections = unpickle(path)

    if args.merging_method == 'pre_nms_avg':
        detections['loc_data'][ensemble_idx] = torch.cat(output[0])
        detections['conf_data'][ensemble_idx] = torch.cat(output[1])
    else:
        detections['detections'][ensemble_idx] = torch.cat(output[0])
        detections['num_boxes_per_class'][ensemble_idx] = torch.stack(output[1])


    with open(path,'wb') as f:
        pickle.dump(detections, f, protocol=4) # protocol 4 allows for large (>4 gb) files to be saved

    print('Saved outputs of model : {:d}/{:d}'.format(ensemble_idx+1,args.ensemble_size))

    return path

def save_observations(args,
                      al_iter,
                      new_observations):

    path = args.experiment_dir+ 'sample_selection/observations-iter_' + str(args.al_iteration) + '_.pickle'
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            pickle.dump(new_observations, f)
        observations = new_observations
    else:
        old_observations = unpickle(path)
        for i in range(len(old_observations)):
            # update the dicts
            old_observations[i].update(new_observations[i])
        observations = old_observations
        with open(path, 'wb') as f:
            pickle.dump(observations, f)

    print('Saved new observations')

    return observations, path



def load_detections(args):
    path = args.experiment_dir + 'sample_selection/al-iteration_' + str(args.al_iteration) + '_detections_.pickle'
    detections = unpickle(path)

    return detections

def write_summary(args,
                  timers,
                  write
                  ):
    """
    Every active learning iteration this function updates a file that makes a summary of the progress
    """
    # create summary file if it doesn't exist yet
    if not os.path.isfile(args.experiment_dir+'summary.json'):
        summary = {}

        # settings:
        summary['sampling_strategy'] = args.sampling_strategy
        summary['merging_strategy'] = str(args.merging_method) # turned to string in case its None
        summary['samples_per_iter'] = args.samples_per_iter
        summary['budget_in_images'] = str(not(args.budget_measured_in_objects))

        summary['training_hyper_params'] = {}
        summary['training_hyper_params']['trained_models'] = args.trained_models

        # object classes sampled
        summary['object_classes_sampled'] = [0 for cl in range(args.num_classes)]



    else:
        # load current summary
        summary = read_summary(args)


    if write == 'sample_selection':
        # sample selection must always be the first step of the al_iteration
        summary[str(args.al_iteration)] = {}
        summary[str(args.al_iteration)]['sample_selection'] = {}

        # sample selection: timer, num images, num objects,
        summary[str(args.al_iteration)]['sample_selection']['timer'] = timers['sample_selection'].total_time
        # summary[str(args.al_iteration)]['sample_selection']['num_images_viewed'] = args.summary['sample_selection']['num_images_viewed']
        summary[str(args.al_iteration)]['sample_selection']['images_selected'] = args.summary['sample_selection']['images_selected']
        summary[str(args.al_iteration)]['sample_selection']['object_classes_selected'] = args.summary['sample_selection']['object_classes_selected']
        summary[str(args.al_iteration)]['sample_selection']['num_objects_selected'] = sum(l_item for l_item in args.summary['sample_selection']['object_classes_selected'])


        # total images selected
        if args.al_iteration > 0:
            summary[str(args.al_iteration)]['sample_selection']['total_selected_images'] = summary[str(args.al_iteration - 1)]['sample_selection']['total_selected_images'] \
                                                                                           + args.summary['sample_selection']['images_selected']
        else:
            summary[str(args.al_iteration)]['sample_selection']['total_selected_images'] = args.summary['sample_selection']['images_selected']



        # update the total number of classes sampled
        list1 = summary[str(args.al_iteration)]['sample_selection']['object_classes_selected']
        list2 = summary['object_classes_sampled']
        summary[str(args.al_iteration)]['sample_selection']['total_classes_selected'] = [a + b for a, b in zip(list1, list2)]
        summary['object_classes_sampled'] = summary[str(args.al_iteration)]['sample_selection']['total_classes_selected']

    elif write == 'train_model':
        if str(args.al_iteration) not in summary:
            summary[str(args.al_iteration)] = {}
        summary[str(args.al_iteration)]['train_model'] = {}
        # train model, timer, num images, num objects, iterations, losses
        summary[str(args.al_iteration)]['train_model']['timer'] = timers['train_model'].total_time
        summary[str(args.al_iteration)]['train_model']['total_iterations'] = args.summary['train_model']['total_iterations']
        summary[str(args.al_iteration)]['train_model']['losses'] = {}
        for ensemble_idx in range(args.ensemble_size):
            summary[str(args.al_iteration)]['train_model']['losses'][ensemble_idx] = {}
            summary[str(args.al_iteration)]['train_model']['losses'][ensemble_idx]['val_loc_loss'] = args.summary['train_model']['losses'][ensemble_idx]['val_loc_loss']
            summary[str(args.al_iteration)]['train_model']['losses'][ensemble_idx]['val_conf_loss'] = args.summary['train_model']['losses'][ensemble_idx]['val_conf_loss']
            summary[str(args.al_iteration)]['train_model']['losses'][ensemble_idx]['train_loc_loss']= args.summary['train_model']['losses'][ensemble_idx]['train_loc_loss']
            summary[str(args.al_iteration)]['train_model']['losses'][ensemble_idx]['train_conf_loss'] = args.summary['train_model']['losses'][ensemble_idx]['train_conf_loss']



        # total time will be overwritten if evaluation is done as well
        summary[str(args.al_iteration)]['timer_al_iteration'] = timers['full_al_iteration'].total_time

    elif args.eval_every_iter and write == 'eval_model':
        summary[str(args.al_iteration)]['eval_model'] = {}


        # total time - overwrites the total time that has been written in train_model summary
        summary[str(args.al_iteration)]['timer_al_iteration'] = timers['full_al_iteration'].total_time


        # eval model, timer, num_images, num_objects, map_scores
        summary[str(args.al_iteration)]['eval_model']['timer'] = timers['eval_model'].total_time
        summary[str(args.al_iteration)]['eval_model']['num_images_eval'] = args.summary['eval_model']['num_images_eval']
        summary[str(args.al_iteration)]['eval_model']['num_objects_eval'] = args.summary['eval_model']['num_objects_eval']
        summary[str(args.al_iteration)]['eval_model']['APs'] = args.summary['eval_model']['APs']

    else:
        # al-iteration timer
        # total time - overwrites the total time that has been written in train_model summary
        summary[str(args.al_iteration)]['timer_al_iteration'] = timers['full_al_iteration'].total_time


    # Write to file
    with open(args.experiment_dir+'summary.json', "w+") as json_file:
        # Convert into JSON and write to outfile
        json.dump(summary, json_file)
            

def read_summary(args):
    with open(args.experiment_dir+'summary.json') as json_file:
        summary = json.load(json_file)

    return summary


def retrieve_all_annotations(args,
                             train_dataset,
                             labeled_idx,
                             default_boxes):
    """
    returns the annotations for the labeled_idx
    # and optionally only for the bounding boxes that correspond to the (hard) predicted bounding boxes
    """

    # a list of annotations, index of the annotation is the index of the labeled_idx list
    annotations = []

    # a list to keep track of which classes have been annotated, index = class
    classes = [0 for i in range(args.num_classes)]

    if args.dataset == 'VOC07':
        if args.annotate_all_objects:
            for img_idx in labeled_idx:
                # retrieve annotations for the image (x1,x2,y1,y2,class_num)
                img_annotations = train_dataset.pull_anno_using_imageset_id((train_dataset.ids[0][0],img_idx))
                annotations.append(img_annotations)

                for anno in img_annotations[1]:
                    classes[int(anno[4])] += 1

        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return annotations, classes


def data_subset(dataset, idx_to_include):
    # todo: subset with not all objects labeled

    subset = copy.copy(dataset)
    idx_to_include = set(idx_to_include)
    subset.ids = [i for i in dataset.ids if i[1] in idx_to_include]
    subset.size = len(idx_to_include)

    return subset

def cross_val_splits(args,
                     labeled_images):

    splits = []

    # shuffle so if there is an order in the dataset, it is now shuffled and distributed among the different splits
    random.shuffle(labeled_images.ids)
    labeled_image_ids = labeled_images.ids

    # create indices for the cross validation
    num_samples = len(labeled_image_ids)
    val_indices = np.floor(np.linspace(0, num_samples, args.num_crossval_folds + 1))

    for i in range(args.num_crossval_folds):
        split_ids = [idx[1] for idx in labeled_image_ids[int(val_indices[i]):int(val_indices[i+1])]]
        splits.append(split_ids)



    return splits

def load_evalset(args, imageset, idx=None):
    """
    Only works for Pascal VOC at the moment

    :param args:
    :param imageset: must be list of tuples (year,imageset)
    :param idx:
    :return:
    """

    if (args.subset_test != None and idx == None):
        idx = []
        indices = list(np.arange(args.subset_test))
        for im_set in imageset:
            idx_subset_imgset = get_imageset_idx_from_imageset_files(args,
                                                                     indices,
                                                                     im_set)
            idx += idx_subset_imgset
    elif (args.subset_test == None and idx == None):
        idx = []
        indices = None
        for im_set in imageset:
            idx_subset_imgset = get_imageset_idx_from_imageset_files(args,
                                                                     indices,
                                                                     im_set,
                                                                     all_indices = True)


    if args.dataset == 'VOC07':
        testset = VOCDetection(args.dataset_root, imageset, BaseTransform(300, args.cfg['dataset_mean']), VOCAnnotationTransform(), idx=idx)

    elif args.dataset == 'VOC07_1_class':
        target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)
        testset = VOCDetection(args.dataset_root, imageset, BaseTransform(300, args.cfg['dataset_mean']),
                               idx=idx, num_classes=1, object_class_number = args.object_class_number,
                               target_transform=target_transform)

    elif args.dataset == 'VOC07_6_class':
        target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        testset = VOCDetection(args.dataset_root, imageset, BaseTransform(300, args.cfg['dataset_mean']),
                               idx=idx,
                               num_classes=6,
                               # object_class_number = args.object_class_number,
                               target_transform=target_transform)

    return testset


def load_sample_select_set(args,imageset, transform = 'base', idx = None, keep_difficult = True):
    if transform == 'ssd':
        augmentation = augmentations.SSDAugmentation(args.cfg['min_dim'],
                                                     (104, 117, 123))
    elif transform == 'base':
        augmentation = BaseTransform(300, args.cfg['dataset_mean'])
    elif transform == 'no_transform':
        augmentation = BaseTransform(300, (0,0,0))
    else:
        raise NotImplementedError()

    if (args.subset_train != None and idx == None):
        idx = []
        indices = list(np.arange(args.subset_train))
        for im_set in imageset:
            idx_subset_imgset = get_imageset_idx_from_imageset_files(args,
                                                                     indices,
                                                                     im_set)
            idx += idx_subset_imgset

    if args.dataset == 'VOC07':
        testset = VOCDetection(args.dataset_root,
                               imageset,
                               transform=augmentation,
                               idx = idx,
                               target_transform=VOCAnnotationTransform(keep_difficult=keep_difficult))
    elif args.dataset == 'VOC07_1_class':
        target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        testset = VOCDetection(args.dataset_root,
                               imageset,
                               transform=augmentation,
                               idx = idx,
                               target_transform=target_transform,
                               num_classes=1,
                               object_class_number = args.object_class_number,
                               )

    elif args.dataset == 'VOC07_6_class':
        target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        testset = VOCDetection(args.dataset_root,
                               imageset,
                               transform=augmentation,
                               idx = idx,
                               target_transform=target_transform,
                               num_classes=6,
                               # object_class_number = args.object_class_number
                               )

    return testset


def load_trainset(args, imageset, transform = 'ssd', idx=None):
    """
    Datasets used in the train function (train and val), uses SSD augmentation (all augmentations)
    """


    if transform == 'ssd':
        augmentation = augmentations.SSDAugmentation(args.cfg['min_dim'],
                                                     (104, 117, 123))
    elif transform == 'base':
        augmentation = BaseTransform(300, args.cfg['dataset_mean'])

    elif transform == 'no_transform':
        augmentation = BaseTransform(300, (0,0,0))

    else:
        raise NotImplementedError()


    if (args.subset_train != None and idx == None):
        idx = []
        indices = list(np.arange(args.subset_train))
        for im_set in imageset:
            idx_subset_imgset = get_imageset_idx_from_imageset_files(args,
                                                                     indices,
                                                                     im_set)
            idx += idx_subset_imgset


    if args.dataset == 'VOC07':
        trainset = VOCDetection(args.dataset_root,imageset,
                                transform=augmentation,
                                idx=idx)
    elif args.dataset == 'VOC07_1_class':
        if transform=='no_transform':
            target_transform = VOCAnnotationTransform2(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        else:
            target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        trainset = VOCDetection(args.dataset_root, imageset,
                                transform=augmentation,
                                idx=idx,
                                num_classes=1,
                                object_class_number = args.object_class_number,
                                target_transform = target_transform)

    elif args.dataset == 'VOC07_6_class':
        target_transform = VOCAnnotationTransform(keep_difficult=True,
                                                  class_to_ind=args.class_to_ind)

        trainset = VOCDetection(args.dataset_root, imageset,
                                transform=augmentation,
                                idx=idx,
                                num_classes=6,
                                # object_class_number = args.object_class_number,
                                target_transform = target_transform
                                )
    elif args.dataset == 'VOC12':
        raise NotImplementedError()
        # trainset = VOCDetection(args.dataset_root, [('2012', 'train')], augmentation)
    else:
        raise NotImplementedError()

    return trainset



def build_sample_selection_net(args,
                               ensemble_idx = 0,
                               merging_method = 'pre_nms_avg',
                               default_forward = False,
                               sampling_strategy = None,
                               forward_vgg_base_only = False):

    net = build_ssd('test', # phase: train or test
                    args.modeltype,
                    args.num_classes,
                    default_forward,
                    merging_method,
                    sampling_strategy,
                    sample_select_forward = True,
                    sample_select_nms_conf_thresh = args.sample_select_nms_conf_thresh,
                    cfg = args.cfg,
                    forward_vgg_base_only = forward_vgg_base_only)



    # if no trained net yet (e.g. first active learning iteration without warmed up network),
    # initialize a neural net with only the base weights
    if args.modeltype != 'SSD300KL':
        if args.paths_to_weights == None:
            vgg_weights = torch.load(args.basenet)
            net.vgg.load_state_dict(vgg_weights)
            # initialize newly added layers' weights with xavier method
            net.extras.apply(weights_init)
            net.loc.apply(weights_init)
            net.conf.apply(weights_init)
        else:
            # dict saved as a data parallel model (train model). But test-net doesn't run with data parallel (todo)
            # so convert weights so it can be loaded by
            net.load_state_dict(torch.load(args.paths_to_weights[ensemble_idx], map_location=args.device))
    else:
        if args.paths_to_weights == None:
            vgg_weights = torch.load(args.basenet)
            net.vgg.load_state_dict(vgg_weights)
            # initialize newly added layers' weights with xavier method
            net.extras.apply(weights_init)
            net.loc.apply(weights_init)
            net.conf.apply(weights_init)
            #https://pytorch.org/docs/master/nn.init.html
            print('adding loc std layers...')

            net.loc_std.apply(loc_std_init)
        else:
            # dict saved as a data parallel model (train model). But test-net doesn't run with data parallel (todo)
            # so convert weights so it can be loaded by
            net.load_state_dict(torch.load(args.paths_to_weights[ensemble_idx], map_location=args.device))
    # set net to eval
    net = net.eval()

    if args.device == 'cuda':
        net = net.cuda()
        cudnn.benchmark = True

    print('Finished loading testing model!')

    return net

def build_eval_net(args,
                   default_forward = True,
                   merging_method = None):

    net = build_ssd('test',  # phase: train or test
                    args.modeltype,
                    args.num_classes,
                    default_forward,
                    merging_method,
                    cfg = args.cfg
                    )

    net.load_state_dict(torch.load(args.path_to_eval_weights, map_location=args.device))

    # set net to eval
    net.eval()
    if args.device == 'cuda':
        net = net.cuda()
        cudnn.benchmark = True

    print('Finished loading evaluation model!')

    return net


def build_train_net(args,
                    ensemble_idx = 0):
    ssd_net = build_ssd('train',
                        args.modeltype,
                        args.num_classes,
                        default_forward=None,
                        merging_method=None,
                        cfg = args.cfg
                        )

    if args.trained_models and not args.train_from_basenet_every_iter:
        # continue training from pre-trained model trained on seed-set
        print('Resuming training, loading {}...'.format(args.trained_models[ensemble_idx]))
        ssd_net.load_weights(args.trained_models[ensemble_idx])

    # don't use net trained on seed set but just the vgg basenet weights
    else:
        if args.modeltype != 'SSD300KL':

            vgg_weights = torch.load(args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
        else:
            vgg_weights = torch.load(args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
            print('adding loc std layers...')
            ssd_net.loc_std.apply(loc_std_init)





    if args.device == 'cuda':
        ssd_net = torch.nn.DataParallel(ssd_net)
        ssd_net.cuda()
        cudnn.benchmark = True  # Makes implementation of Neural Network faster if input size is fixed https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

    # set net to train
    ssd_net.train()

    return ssd_net



def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def loc_std_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.0001)
        m.bias.data.zero_()

def save_weights(weights,
                 args,
                 path,
                 mode = 'neural_net'):
    """
    #see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686

    """
    if mode == 'neural_net':
        folder = args.experiment_dir + 'weights/'
        print('saving nn weights!')

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    else:
        raise NotImplementedError()

    if args.device == 'cuda' and mode == 'neural_net':  # data parallism is used, so the state dict needs to be saved without the module prefix (see thread above)
        torch.save(weights.module.state_dict(), path)
    else:
        torch.save(weights.state_dict(), path)




    return path

def early_stopping(args,
                   net,
                   criterion,
                   val_batch_iterator,
                   val_conf_loss_list,
                   val_loc_loss_list,
                   val_loss):
    """
    Determines whether to continue training or not based on the validation losses
    """

    converged = False

    # calculate validation losses
    # load data
    images, targets = next(val_batch_iterator)

    # create variables for images and targets
    with torch.no_grad():

        if args.device == 'cuda':
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann) for ann in targets]

        # forward pass
        out = net(images)
        val_loc_loss, val_conf_loss = criterion(out, targets)
        val_loss_new = val_loc_loss + val_conf_loss

        # store val losses for later plotting
        val_conf_loss_list.append(val_conf_loss.item())
        val_loc_loss_list.append(val_loc_loss.item())
        print("Validation loss 1: \t " + str(val_loss_new))

        if args.early_stopping_condition[-1] == '2' and len(val_loc_loss_list) > 1:
            # do new validation pass
            images, targets = next(val_batch_iterator)

            if args.device == 'cuda':
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            # forward pass
            out = net(images)
            val_loc_loss, val_conf_loss = criterion(out, targets)
            val_loss_new2 = val_loc_loss + val_conf_loss

            # store val losses for later plotting
            val_conf_loss_list.append(val_conf_loss.item())
            val_loc_loss_list.append(val_loc_loss.item())
            print("Validation loss 2: \t " + str(val_loss_new2))

        if args.early_stopping_condition == 'val_loc_loss' and len(val_loc_loss_list) >= 2:
            if val_loc_loss_list[-1] > val_loc_loss_list[-2]:
                converged = True

        elif args.early_stopping_condition == 'val_conf_loss' and len(val_conf_loss_list) >= 2:
            if val_conf_loss_list[-1] > val_conf_loss_list[-2]:
                converged = True

        elif args.early_stopping_condition == 'val_loss' or len(val_conf_loss_list) < 3:
            if val_loss_new > val_loss:
                converged = True

        elif args.early_stopping_condition == 'val_loss2' and len(val_conf_loss_list) > 3:
            if (val_loss_new and val_loss_new2) > val_loss:
                converged = True

        elif args.early_stopping_condition == 'val_loc_loss2' and len(val_loc_loss_list) > 3:
            if (val_loc_loss_list[-1] and val_loc_loss_list[-2]) > val_loc_loss_list[-3]:
                converged = True

        elif args.early_stopping_condition == 'val_conf_loss2' and len(val_conf_loss_list) > 3:
            if (val_conf_loss_list[-1] and val_conf_loss_list[-2]) > val_conf_loss_list[-3]:
                converged = True

    val_loss = val_loss_new


    return converged, val_conf_loss_list, val_loc_loss_list, val_loss

class subset_sampler(Sampler):

    """
    inspired by: https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset

    for more information regarding samples see: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
    and: https://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html#Sampler

    """

    def __init__(self, mask, shuffled):
        if shuffled:
            random.shuffle(mask)
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))  # todo test if it works as expected

    def __len__(self):
        return len(self.mask)

class InfiniteDataLoader(torch.utils.data.DataLoader):
    """
    https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class Timer(object):
    """A simple timer.
     copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch

    """
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def eval_summary_writer(args,
                        timers,
                        epocht_test = False,
                        iters_trained = 'nan'):
    """
    Writes all evaluation results in one summary file
    """
    # create summary file if it doesn't exist yet
    if not os.path.isfile(args.save_dir_eval + 'eval_summary.json'):
        summary = {}
        # summary['idx'] = args.ensemble_idx


    else:
        # load current summary
        with open(args.save_dir_eval + 'eval_summary.json') as json_file:
            summary = json.load(json_file)

    # total time - overwrites the total time that has been written in train_model summary
    if str(args.al_iteration) not in summary:
        summary[str(args.al_iteration)] = {}
        summary[str(args.al_iteration)]['eval_model'] = {}
    if str(args.eval_ensemble_idx) not in summary[str(args.al_iteration)]['eval_model']:
        summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)] = {}
    try:
        train_iters = str(args.path_to_eval_weights.split('train-iter')[1].split('_')[1])
    except AttributeError:
        train_iters = 'None'
        args.path_to_eval_weights = 'None'

    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)] = {}
    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['train_iters'] = train_iters

    # eval model, timer, num_images, num_objects, map_scores
    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['timer'] = timers['eval_model'].total_time
    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['num_images_eval'] = args.summary['eval_model']['num_images_eval']
    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['num_objects_eval'] = args.summary['eval_model']['num_objects_eval']

    # # APs
    # if iters_trained == 'nan':

    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['APs'] = {}
    iou_thresholds = [0.3]
    iou_thresholds.extend(list(np.linspace(0.5,0.95,10)))
    for iou_threshold in iou_thresholds:
        summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['APs'][str(iou_threshold)] = args.summary['eval_model']['APs'][str(iou_threshold)]
    summary[str(args.al_iteration)]['eval_model'][str(args.eval_ensemble_idx)][str(args.path_to_eval_weights)]['APs']['mmAP'] = args.summary['eval_model']['APs']['mmAP']


    # Write to file
    with open(args.save_dir_eval+'eval_summary.json', "w+") as json_file:
        # Convert into JSON and write to outfile
        json.dump(summary, json_file)


def dataset_statistics(args, imageset, path, dataset):
    if args.dataset == 'VOC07':
        # load dataset

        # num classes
        num_classes = dataset.num_classes

        # objects per class
        objects_per_class = np.zeros(num_classes, dtype=int)

        # images per class
        images_per_class = np.zeros(num_classes, dtype=int)

        # image idxs
        real_image_idxs = []

        df_row_list = []
        for im_idx in range(len(dataset)):
            real_image_id, annos = dataset.pull_anno(im_idx)
            classes_on_image = np.zeros(num_classes)
            real_image_idxs.append(real_image_id)
            for anno in annos:
                class_id = anno[-1]
                objects_per_class[class_id] += 1
                classes_on_image[class_id] += 1

            df_row_list.append(classes_on_image)
            classes_on_image = classes_on_image > 0
            images_per_class += classes_on_image

        # image_idx_per_class
        df = pd.DataFrame(df_row_list,
                          columns=('aeroplane', 'bicycle', 'bird', 'boat',
                                   'bottle', 'bus', 'car', 'cat', 'chair',
                                   'cow', 'diningtable', 'dog', 'horse',
                                   'motorbike', 'person', 'pottedplant',
                                   'sheep', 'sofa', 'train', 'tvmonitor'))
        df = df.assign(real_image_idxs=pd.Series(real_image_idxs))


    else:
        raise NotImplementedError()



    stats = {}
    stats['dataframe'] = df
    stats['num_classes'] = num_classes
    stats['objects_per_class'] = objects_per_class
    stats['images_per_class'] = images_per_class


    # save statistics:
    with open(path, 'wb') as out:
        pickle.dump(stats, out)


    return stats


def unpickle(filename):
    with open(filename, "rb") as f:
        try:
            d = pickle.load(f)
        except RuntimeError:
            d = torch.load(f, map_location='cpu')
    return d

def delete_non_optimal_weights(args, mode = 'neural_net', ensemble_id = None, cross_val_fold = None):
    """
    Per active learning iteration, multiple weights get saved. However, these files are quite large.
    This function goes trough these files and deletes all but the one with the lowest validation loss.
    The default file format must be followed though, with in the brackets the values for the network:

    no enseble nor cross val: [Modelname]_al-iter_[al_iter]_train-loss_[train-loss]_val-loss_[val-loss]_[train-set]_train-iter_[train-iter]_.pth
    ensemble no cross val: [Modelname]_al-iter_[al_iter]_train-loss_[train-loss]_val-loss_[val-loss]_[train-set]_train-iter_[train-iter]_ensemble-id_[ensemble_id]_.pth
    ensemble and cross val: [Modelname]_al-iter_[al_iter]_train-loss_[train-loss]_val-loss_[val-loss]_[train-set]_train-iter_[train-iter]_cross-val-fold_[cross-val-fold]_ensemble-id_[ensemble_id]_.pth

    """

    # all weights
    if mode == 'neural_net':
        print('deleting bad weights of neural net')
        path = str(args.experiment_dir +
                    'weights/')
    # elif mode == 'optimizer':
    #     print('deleting the optimizers states of non-optimal neural nets')
    #     path = str(args.experiment_dir +
    #                 'optimizers/')

    saved_weights = os.listdir(path)
    al_iter = str(args.al_iteration)

    # if (ensemble_id != None and cross_val_fold != None):
    #     val_losses = [float(weights.split('_')[7]) for weights in saved_weights if
    #                   (weights.split('_')[2] == al_iter and weights.split('_')[12] == str(cross_val_fold) and
    #                    weights.split('_')[14] == str(ensemble_id))]
    #
    #     lowest_val_loss_this_ensemble = str(min(val_losses))
    #
    #     # names of weights to be deleted: this al_iteration and ensemble_id and not the lowest_val_loss
    #     delete_weights = [weights for weights in saved_weights if
    #                       (weights.split('_')[2] == al_iter and weights.split('_')[12] == str(cross_val_fold) and
    #                    weights.split('_')[14] == str(ensemble_id) and  weights.split('_')[7] != lowest_val_loss_this_ensemble)]
    #
    #     # delete weights from previous folds
    #     if cross_val_fold != 0:
    #         delete_all_from_previous_fold = [weights for weights in saved_weights if
    #                       (weights.split('_')[2] == al_iter and weights.split('_')[12] != str(cross_val_fold) and
    #                    weights.split('_')[14] == str(ensemble_id))]
    #
    #         delete_weights += delete_all_from_previous_fold
    #
    #
    #     keep_weight = [weights for weights in saved_weights if
    #                       (weights.split('_')[2] == al_iter and weights.split('_')[12] == str(cross_val_fold) and
    #                    weights.split('_')[14] == str(ensemble_id) and  weights.split('_')[7] == lowest_val_loss_this_ensemble)][0]
    #
    #     print('\n\n val_loss of best weights: ', lowest_val_loss_this_ensemble)

    if (ensemble_id != None and cross_val_fold == None):
        print('debug prints for deleting non optimal weights') # todo delete

        print([float(weights.split('_')[7]) for weights in saved_weights])
        print(al_iter)
        print([float(weights.split('_')[7]) for weights in saved_weights if weights.split('_')[2] == al_iter])
        print(ensemble_id)
        print(weights.split('_')[-2] for weights in saved_weights)
        print([float(weights.split('_')[7]) for weights in saved_weights if weights.split('_')[-2] == str(ensemble_id)])

        val_losses = [float(weights.split('_')[7]) for weights in saved_weights if
                      (weights.split('_')[2] == al_iter and weights.split('_')[-2] == str(ensemble_id))]

        lowest_val_loss_this_ensemble = str(min(val_losses))

        # names of weights to be deleted: this al_iteration and ensemble_id and not the lowest_val_loss
        delete_weights = [weights for weights in saved_weights if
                          (
                            weights.split('_')[2]  == al_iter
                           and weights.split('_')[-2] == str(ensemble_id)
                           and weights.split('_')[7] != lowest_val_loss_this_ensemble
                           )
                          or (
                              weights.split('_')[2]  < al_iter
                          )]

        keep_weight = [weights for weights in saved_weights if
                       (weights.split('_')[2] == al_iter
                        and weights.split('_')[-2] == str(ensemble_id)
                        and weights.split('_')[7] == lowest_val_loss_this_ensemble)][0]

        print('\n\n val_loss of best weights: ', lowest_val_loss_this_ensemble)
        print('keep weight: ', keep_weight)


    else:

        # val losses
        val_losses = [float(weights.split('_')[7]) for weights in saved_weights if weights.split('_')[2] == al_iter]
        lowest_val_loss = str(min(val_losses))

        # names of weights to be deleted: this al_iteration and not the lowest_val_loss
        delete_weights = [weights for weights in saved_weights if (weights.split('_')[2] == al_iter and weights.split('_')[7] != lowest_val_loss)
                          or (
                                  weights.split('_')[2] < al_iter
                          )
                          ]
        keep_weight = [weights for weights in saved_weights if (weights.split('_')[2] == al_iter and weights.split('_')[7] == lowest_val_loss)][0]
        print('\n\n val_loss of best weights: ', lowest_val_loss)

    # delete the weights
    for weights in delete_weights:
        os.remove(str(path+weights))
        print('weights deleted: ', weights)

    return keep_weight

def just_keep_lowest_val_loss_weights(args,
                                      net,
                                      criterion,
                                      val_data_loader,
                                      ensemble_id):
    ## below is partially copied from the delete_non_optimal_weights helper function
    # only save weights if current weight is better than previous weights
    # all weights
    path = str(args.experiment_dir +
               'weights/')
    saved_weights = os.listdir(path)
    al_iter = str(args.al_iteration)

    if not saved_weights: # if first model for this experiment (empty list)
        lowest_val_loss = np.inf
    else:
        val_losses = [float(weights.split('_')[7]) for weights in saved_weights if
                      (weights.split('_')[2] == al_iter and weights.split('_')[-2] == str(ensemble_id))]

        if not val_losses: # if first model for this ensemble_id, and al_iter
            lowest_val_loss = np.inf
        else:
            lowest_val_loss = min(val_losses)

    val_loc_losses = []
    val_conf_losses = []
    val_batch_iterator = iter(val_data_loader)

    ## validation pass
    for i in range(len(val_data_loader)):
        if i % 5 == 0:
            print('Val iter ', i, '/',len(val_data_loader))
        if args.debug and i > 0: # todo remove (all args.debug)
            break

        # calculate validation losses
        # load data
        images, targets = next(val_batch_iterator)

        # create variables for images and targets
        with torch.no_grad():

            if args.device == 'cuda':
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            # forward pass
            out = net(images)
            val_loc_loss, val_conf_loss = criterion(out, targets)
            val_loc_losses.append(val_loc_loss.item())
            val_conf_losses.append(val_conf_loss.item())

    val_loc_loss = np.mean(val_loc_losses)
    val_conf_loss = np.mean(val_conf_losses)

    val_loss_new_mean = val_loc_loss + val_conf_loss
    ## clean up weights - if better save new weights and throw away old ones, else: keep old 'best weights'
    if val_loss_new_mean < lowest_val_loss:
        # save weight
        save = True
        val_loss = str(val_loss_new_mean)

    else:
        # dont save weight
        save = False
        val_loss = str(lowest_val_loss)
        print('keeping old weights as best weight, with val_loss: ', str(lowest_val_loss))



    return val_loss, val_loc_loss, val_conf_loss, save

def set_all_seeds(args, seed_incrementer = 0):
    print('setting seeds to: ',args.seed+seed_incrementer)
    random.seed(args.seed+seed_incrementer)
    np.random.seed(args.seed+seed_incrementer)
    torch.manual_seed(args.seed+seed_incrementer)
    torch.cuda.manual_seed_all(args.seed+seed_incrementer)

def class_dist_in_imageset(args, image_idxs, dataset):
    if args.dataset in ['VOC07', 'VOC12','VOC07_1_class','VOC07_6_class']:

        path_to_imset = dataset.ids[0][0]

        selected_classes = [0 for cl in range(args.cfg['num_classes'] - 1)] # -1 for background

        for imset_idx in image_idxs:
            # im_annotations = dataset.pull_anno(imset_idx)[1]

            im_annotations = dataset.pull_anno_using_imageset_id((path_to_imset,imset_idx))[1]
            for anno in im_annotations:

                cl = anno[4]
                if args.dataset == 'VOC07_1_class': #todo: doesn't work as intended anymore because of changed interface
                    # print(cl)
                    if cl == 0:
                        selected_classes[0] += 1
                else:
                    selected_classes[cl] += 1

                # if args.dataset == 'VOC07_6_class':



    else:
        raise NotImplementedError()

    return selected_classes


def adjust_learning_rate(args, optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (args.gamma ** (step))
    print('adjusting lr to: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_image_features(args,
                        image_idxs,
                        path_to_image_feature_dir):
    #todo
    print('loading image features for: ', image_idxs)
    image_features = []

    for idx in image_idxs:
        path = path_to_image_feature_dir + idx + '.pickle'
        image_features.append(unpickle(path))

    print('image features loaded!')
    return image_features


def get_imageset_idx_from_imageset_files(args,
                                         indices,
                                         imageset,
                                         all_indices = None):
    """
    Returns a list of imageset idx corresponding to the indices in the imageset file.
    """

    # load dataset
    print('taking a subset of imageset: ', imageset)
    print('Subset indices: ', indices)
    dataset = VOCDetection(args.dataset_root, [imageset], BaseTransform(300, args.cfg['dataset_mean']))

    # take the imageset_idx
    if all_indices:
        imageset_idx = [dataset.ids[i][1] for i in range(len(dataset.ids))]
    else:
        imageset_idx = [dataset.ids[i][1] for i in indices]

    print('imageset idx of the subset: ', imageset_idx)
    return imageset_idx



def update_density_per_imageset_per_al_iter(dataset,
                                            load_dir_similarities,
                                            unlabeled_images
                                            ):
    """
    density is the mean similarity of one image to all other images in the dataset (see Settles 2008)
    """
    # todo: can be made faster, now doing redundant calculations (similarities of a->b and b->a), but is fast anyway

    # go trough dataset
    density = {}
    for i,idx in enumerate(unlabeled_images):
        # load similarity between all images in trainval and current image (idx)
        path = load_dir_similarities + idx + '.pickle'

        similarities_idx = unpickle(path)

        # go trough all OTHER images in the dataset (can be a subset of trainval, e.g. only the car images)
        # except the id where are currently
        other_images = [idj for idj in unlabeled_images if (idj != idx)]

        # placeholder
        density[idx] = 0
        for i, idj in enumerate(other_images):

            density[idx] += similarities_idx[idj]

        # divide by number of images to get mean
        density[idx] /= len(other_images)



    # save image density dir
    return density