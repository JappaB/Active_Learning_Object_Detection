import sys
import json
import time
import datetime
import os
import random
import pickle

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
from sklearn.cluster import AffinityPropagation


from layers.modules import MultiBoxLoss
from layers import box_utils
from . import helpers
# import helpers
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, detection_collate
from utils import augmentations
from . import uncertainty_helpers

import hdbscan


# import uncertainty_helpers


def detect_on_unlabeled_imgs(net,
                             args,
                             sample_select_dataset,
                             unlabeled_idx,
                             len_unlabeled_idx
                             ):

    # placeholder for all outputs of the image
    if args.merging_method == 'pre_nms_avg':
        # image_outputs = []
        loc_data_list = []
        conf_data_list = []

    else:
        # image_outputs = torch.zeros(len_unlabeled_idx, args.cfg['num_classes'], 200, args.num_classes + 4, device='cpu') # last element is full distribution + bounding box
        output_list = []
        num_boxes_per_class_list = []


    # transform needed for preprocessing
    transform = BaseTransform(net.size, (104, 117, 123))

    with torch.no_grad():
        # not using batches right now because of memory issues
        for i, idx in enumerate(unlabeled_idx): # enumerate, as we need the values, but need to loop over all elements

            if i == 0:
                start = time.time()

            # load image and transform (colors in different order)
            img = sample_select_dataset.pull_image_using_imageset_id((sample_select_dataset.ids[0][0],idx)) # todo: june 4 changed unlabeled_idx, does this still work?

            x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))


            if args.device == 'cuda':
                torch.cuda.empty_cache()
                x = x.to(args.device)


            # Forward pass
            if args.merging_method == 'pre_nms_avg':
                loc_data, conf_data, priors = net(x)

                # if batches used
                if loc_data.shape[0] > 1:
                    loc_data = loc_data.to('cpu').split(loc_data.shape[0], dim=0)
                    conf_data = loc_data.to('cpu').split(loc_data.shape[0], dim=0)
                    loc_data_list.extend(loc_data)
                    conf_data_list.extend(conf_data)

                else:
                    loc_data_list.append(loc_data.to('cpu'))
                    conf_data_list.append(conf_data.to('cpu'))
                del loc_data,conf_data

            else:
            # entropy_scores, real_predictions_indices, boxes, ssd_model_output = net.detect(loc_data, conf_data, priors)
                output, num_boxes_per_class, priors = net(x)

                #if batches used
                if output.shape[0] > 1:
                    output = output.to('cpu').split(output.shape[0], dim=0)
                    output_list.extend(output)
                    num_boxes_per_class.extend(num_boxes_per_class)

                else:
                    output_list.append(output.to('cpu'))
                    num_boxes_per_class_list.append(num_boxes_per_class.to('cpu'))

                del output, num_boxes_per_class


            if i % 10 == 0 and i != 0:
                stop = time.time()
                elapsed = stop - start
                if args.device == 'cuda':
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                else:
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                start = time.time()

    if args.merging_method == 'pre_nms_avg':

        image_outputs = (loc_data_list, conf_data_list,priors)# priors are same for every image and every ensemble (in my experiments)

    else:
        image_outputs = (output_list, num_boxes_per_class_list)

    return image_outputs, len_unlabeled_idx, priors.to('cpu'), unlabeled_idx



def cluster_detections_to_observations(args,
                                       merging_method,
                                       num_unlabeled_images,
                                       unlabeled_imgset,
                                       priors,
                                       IoU_tresh = 0.9
                                       ):
    """

    :param args:
    :param merging_method:
    :param num_unlabeled_images:
    :param unlabeled_imgset:
    :param priors:
    :param IoU_tresh:
    :return:
    """
    # load pickled detections
    all_detections = helpers.load_detections(args)

    if merging_method == 'pre_nms_avg':
        observations = pre_nms_avg(args,
                                   all_detections,
                                   num_unlabeled_images,
                                   unlabeled_imgset,
                                   priors)

    elif merging_method == 'bsas':
        # cluster
        observations = bsas(args,
                            all_detections,
                            num_unlabeled_images,
                            unlabeled_imgset,
                            priors)

    elif merging_method == 'hbdscan':
        # cluster
        observations = hbdscan(args,
                                all_detections,
                                num_unlabeled_images,
                                unlabeled_imgset,
                                priors)

    else:
        raise NotImplementedError()


    return observations

def pre_nms_avg(args,
                all_detections,
                unlabeled_imgset,
                num_unlabeled_images,
                priors,
                top_k = 200,
                nms_thresh = 0.5):
    """
    paper: Benchmarking Sampling-based Probabilistic Object Detectors - Dimity Miller, Niko Sunderhauf, Haoyang Zhang, David Hall, Feras Dayoub


    :param args:
    :param all_detections:
    :return:
    """

    # average the classification scores
    averaged_conf_data = torch.einsum('eibc->ibc',all_detections['conf_data'])/args.ensemble_size

    # decode bounding boxes from center offset-form to (x1x2y2y2)
    #  [max_boxes(ens), images, detections, decoded_coordinates)
    #  #todo: decoding boxes below should be doable without for loops, think it will result in a minimal reduction in time though
    decoded_boxes = torch.zeros(args.ensemble_size, num_unlabeled_images, 8732, 4, requires_grad=False, device='cpu')
    for ens in range(args.ensemble_size):
        for im in range(num_unlabeled_images):
            decoded_boxes[ens, im] = \
                box_utils.decode(all_detections['loc_data'][ens, im], priors,
                                 args.cfg['variance'])

    # if requires_grad = True, some operations in nms don't work
    decoded_boxes = decoded_boxes.detach()


    averaged_bounding_boxes, cov_0, cov_1 = uncertainty_helpers.means_covs_observation(decoded_boxes.permute(1,2,0,3))
    observations_means = {}
    observations_cov_0 = {}
    observations_cov_1 = {}
    observations_dist = {}

    total_num_observations = {}

    for i, img in enumerate(unlabeled_imgset):
        observations_means[img] = []
        observations_cov_0[img] = []
        observations_cov_1[img] = []
        observations_dist[img] = []
        total_num_observations[img] = {}

        conf_scores = averaged_conf_data[i].clone()
        conf_scores = conf_scores.permute(1,0)
        decoded_boxes = averaged_bounding_boxes[i,:]

        for cl in range(1, args.cfg['num_classes']):

            c_mask = conf_scores[cl].gt(args.sample_select_nms_conf_thresh)  # confidence mask, speeds up processing by not applying nms everywhere

            # to all bounding boxes
            scores = conf_scores[cl][c_mask]
            if scores.size(0) == 0:
                continue

            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)

            # apply nms
            # NOTE: ids are the ids in the 'boxes' variable (conf > conf_tresh), these are not the same as the bounding box ids
            # ids, count = box_utils.nms(boxes, scores.to(args.device), nms_thresh, top_k)
            ids, count = box_utils.nms(boxes, scores, args.sample_select_nms_conf_thresh, top_k)

            observations_means[img].append(boxes[ids[:count]].to('cpu'))  # x1y1x2y2
            observations_cov_0[img].append(cov_0[i][c_mask][ids[:count]].to('cpu'))
            observations_cov_1[img].append(cov_1[i][c_mask][ids[:count]].to('cpu'))
            observations_dist[img].append(averaged_conf_data[i][c_mask][ids[:count]].to('cpu'))
            total_num_observations[img][cl] = count

    # shape: [images, classes, mu1(x1y1) mu2(x2y2) covariance1 (xx xy yx yy) covariance2(xx xy yx yy) distribution]
    return observations_means,observations_cov_0,observations_cov_1,observations_dist, total_num_observations

def bsas(args,
         all_detections,
         unlabeled_imgset,
         num_unlabeled_images,
         priors,
         iou_tresh = 0.5):
    """
    I find the best explanation of the algorithm to be equation 2 in:
    Miller, Nicholson, Dayoub, Sunderhauf- Dropout Sampling for Robust Object Detection in Open-Set Conditions
    Another noteworthy paper is:
    Miller, Dayoub, Milford, Sunderhauf - Evaluating Merging Strategies for Sampling-based Uncertainty Techniques in Object Detection

    :param args:
    :param all_detections:
    :param unlabeled_imgset:
    :param num_unlabeled_images:
    :param priors:
    :param iou_tresh:
    :return:
    """

    print('Starting BSAS merging local time: ', datetime.datetime.now())

    if args.debug:
        iou_tresh = 0.5

    # placeholders
    clusters = {}
    observations_means = {}
    observations_cov_0 = {}
    observations_cov_1 = {}
    observations_dist = {}
    detections_per_observation = {}

    total_num_observations = {}
    # cluster for each image
    for i, img in enumerate(unlabeled_imgset):
        if i % 10 == 0 and i != 0:
            print('Merged image {:d}/{:d} trough {:d}/{:d}....'
                  .format(i - 9, num_unlabeled_images, i + 1, num_unlabeled_images))
            print('Local time: ', datetime.datetime.now())
        dets = [all_detections['detections'][ens, i, cl, :int(all_detections['num_boxes_per_class'][ens, i][cl].item()), :]
         for cl in range(1, args.num_classes) for ens in range(args.ensemble_size)]
        nonzero_dets = [det for det in dets if det.ge(0.0).sum() > 0]

        img_detections = torch.cat(nonzero_dets)
        observations_cov_0[img] = []
        observations_cov_1[img] = []
        observations_means[img] = []
        observations_dist[img] = []
        detections_per_observation[img] = []
        im_clusters = []
        first_cluster = True

        for d, detection in enumerate(img_detections):
            # first cluster of the image
            if first_cluster:
                im_clusters.append(detection.unsqueeze(dim = 0).unsqueeze(dim=0).unsqueeze(dim=0))  # list with tensors of shape: [batch, observations_per_cluster, num_classes+4] (this form is needed for later mean and cov calculations, they are written in batch form)
                # im_clusters_box_means = detection[args.cfg['num_classes']:].unsqueeze(dim=0)  # shape: [1,4] x1y1x2y2, the first dimension is needed to calculate the IoU overlap
                first_cluster = False
                continue

            for j, cluster in enumerate(im_clusters):
                # calculate IoUs of a detection and all the clusters
                iou_overlaps = box_utils.jaccard(detection[args.cfg['num_classes']:].unsqueeze(dim=0),
                                                 cluster[:,:,:,args.cfg['num_classes']:].squeeze(dim=0).squeeze(dim=0)) # needs to be of shape [num objects, 4]

                detection_fits_cluster = False
                if iou_overlaps.ge(iou_tresh).sum() == cluster.shape[2]:
                    detection_fits_cluster = True
                    break

            # if best IoU is larger than threshold, add detection to cluster with highes IoU overlap,go to next detection
            if detection_fits_cluster:
                # add box
                im_clusters[j] = torch.cat((im_clusters[j], detection.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)),dim=2)  # shape [batch, observations, num_classes+4]  # shape [batch, observations, num_classes+4]

            else:
                # if box wasn't added to any cluster, create new cluster
                # new cluster
                im_clusters.append(detection.unsqueeze(dim = 0).unsqueeze(dim=0).unsqueeze(dim=0)) # list of shape [observations, num_classes+4]

    # for the clusters with 2 or more detections
        num_obs = 0
        for cluster in im_clusters:
            if cluster.shape[2] > 1:
                num_obs += 1
                averaged_bounding_boxes, cov_0, cov_1 = uncertainty_helpers.means_covs_observation(cluster[:,:,:,-4:])
                observations_cov_0[img].append(cov_0)
                observations_cov_1[img].append(cov_1)
                observations_means[img].append(averaged_bounding_boxes)
                observations_dist[img].append(cluster[:,:,:,:args.cfg['num_classes']].mean(dim = 2))
                detections_per_observation[img].append(cluster.shape[2])

        total_num_observations[img] = num_obs

    print('Finished BSAS merging, local time: ', datetime.datetime.now())
    return observations_means,observations_cov_0,observations_cov_1,observations_dist, total_num_observations, detections_per_observation


def hbdscan(args,
         all_detections,
         unlabeled_imgset,
         num_unlabeled_images,
         priors):
    """
    # https: // github.com / scikit - learn - contrib / hdbscan

    :param args:
    :param all_detections:
    :param unlabeled_imgset:
    :param num_unlabeled_images:
    :param priors:
    :return:
    """

    # placeholders
    clusters = {}
    observations_means = {}
    observations_cov_0 = {}
    observations_cov_1 = {}
    observations_dist = {}
    detections_per_observation = {}

    total_num_observations = {}

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)

    for i, img in enumerate(unlabeled_imgset):
        observations_cov_0[img] = []
        observations_cov_1[img] = []
        observations_means[img] = []
        observations_dist[img] = []
        detections_per_observation[img] = []
        ## this crazy twoliner is needed because nms returns a variable number of bounding boxes and fills the rest with zeros.
        # here we extract the nonzero detections
        dets = [all_detections['detections'][ens, i, cl, :int(all_detections['num_boxes_per_class'][ens, i][cl].item()), :]
            for cl in range(1, args.num_classes) for ens in range(args.ensemble_size)]
        nonzero_dets = torch.cat([det for det in dets if det.ge(0.0).sum() > 0])
        upper_left_corners = nonzero_dets[:,args.cfg['num_classes']:-2].numpy()
        cluster_labels = clusterer.fit_predict(upper_left_corners)
        im_clusters = {}
        total_num_observations[img] = cluster_labels.max() + 1 #zero indexed

        # group clusters
        for id, det in enumerate(nonzero_dets):
            cluster_label = int(cluster_labels[id])
            if cluster_label in im_clusters:
                im_clusters[cluster_label].append(det.unsqueeze(dim = 0).unsqueeze(dim=0).unsqueeze(dim=0))  # list with tensors of shape: [batch, observations_per_cluster, num_classes+4] (this form is needed for later mean and cov calculations, they are written in batch form)))
            else:
                im_clusters[cluster_label] = []
                im_clusters[cluster_label].append(det.unsqueeze(dim = 0).unsqueeze(dim=0).unsqueeze(dim=0))  # list with tensors of shape: [batch, observations_per_cluster, num_classes+4] (this form is needed for later mean and cov calculations, they are written in batch form))

        # calculate cluster
        for cluster_id, cluster in im_clusters.items():
            if cluster_id == -1:
                continue

            cluster_tensor = torch.cat(cluster, dim = 2).to('cpu') # shape: [batch, observations, max(n_boxes) ,4]
            averaged_bounding_boxes, cov_0, cov_1 = uncertainty_helpers.means_covs_observation(cluster_tensor[:, :, :, args.cfg['num_classes']:])
            observations_cov_0[img].append(cov_0)
            observations_cov_1[img].append(cov_1)
            observations_means[img].append(averaged_bounding_boxes)
            observations_dist[img].append(cluster_tensor[:, :, :, :args.cfg['num_classes']].mean(dim=2))
            detections_per_observation[img].append(cluster_tensor.shape[2])

    return observations_means, observations_cov_0, observations_cov_1, observations_dist, total_num_observations, detections_per_observation


def neural_net_clusterer():
    raise NotImplementedError()
def train_neural_net_clusterer():
    raise NotImplementedError()
def bsas_exclusive():
    raise NotImplementedError()

def calculate_uncertainties(args,
                            observations,
                            unlabeled_imgset):


    # set torch default tensor to cpu with high precission for these uncertainties
    torch.set_default_tensor_type(torch.DoubleTensor)
    if args.merging_method == 'pre_nms_avg':
        observations_means, observations_cov_0, observations_cov_1, observations_dists, total_num_observations = observations
    else: # num observations is saved when not pre_nms_avg -> in pre_nms avg always the same: the number of networks in the ensemble
        observations_means, observations_cov_0, observations_cov_1, observations_dists, total_num_observations, detections_per_observation = observations

    classification_strategy = args.sampling_strategy.split('_')[0]
    localization_strategy = args.sampling_strategy.split('_')[1]

    if args.no_foreground_multiplication and localization_strategy != 'covariance-obj':
        raise NotImplementedError()

    # localization
    if localization_strategy == 'covariance-obj':
        # (hard to do in batches as each image has a different number of observations)
        # for all images
        localization_uncertainty = {}

        for img in unlabeled_imgset:
            # concat all observations and create batch dimension,
            # even though it is a batch of 1, it allows to keep the functions below to work with batches in the future
            if args.merging_method == 'pre_nms_avg':
                im_observations_cov_0 = torch.cat(observations_cov_0[img]).unsqueeze(0)
                im_observations_cov_1 = torch.cat(observations_cov_1[img]).unsqueeze(0)
                im_observation_means = torch.cat(observations_means[img]).unsqueeze(0)
                mu_0 = im_observation_means[:, :, :2]
                mu_1 = im_observation_means[:, :, 2:]
            else:
                im_observations_cov_0 = torch.cat(observations_cov_0[img], dim = 1)
                im_observations_cov_1 = torch.cat(observations_cov_1[img], dim = 1)
                im_observation_means = torch.cat(observations_means[img]).unsqueeze(0)
                mu_0 = im_observation_means.squeeze()[:, :2].unsqueeze(0)
                mu_1 = im_observation_means.squeeze()[:, 2:].unsqueeze(0)


            # calculate traces of covariances
            trace_0,trace_1 = uncertainty_helpers.trace_covariance(im_observations_cov_0,im_observations_cov_1)

            # calculate sizes of objects (normalized between 0 and 1, as all images are inputed as 300x300 anyways)
            object_sizes = uncertainty_helpers.dist_means_observation(mu_0, mu_1)

            # div the sum of the traces by the dist
            loc_uncertainty = (trace_0+trace_1) / object_sizes

            # multiply the localization uncertainty by (1-P(background)) -> we want objects of which the location is uncertain
            # but which are probably foreground objects
            if args.merging_method == 'pre_nms_avg':
                im_observations_dists = torch.cat(observations_dists[img]).unsqueeze(0)
            else:
                im_observations_dists = torch.cat(observations_dists[img]).squeeze().unsqueeze(0)
            background_probs = im_observations_dists[:,:,0]

            # todo: delete_top_n_percent_loc_uncertainty_hbdscan
            if args.merging_method == 'hbdscan' and args.delete_top_n_percent_loc_uncertainty_hbdscan:
                raise NotImplementedError
                # topk = round(args.delete_top_n_percent_loc_uncertainty_hbdscan * total_num_observations[img])
                # top_k_loc_uncertainty = torch.topk(loc_uncertainty, k = topk)

            if not args.no_foreground_multiplication:
                loc_uncertainty_enumerator = (loc_uncertainty * (1 - background_probs)).sum(dim=1)
                loc_uncertainty_denominator = background_probs.sum(dim=1)
            elif args.no_foreground_multiplication:
                loc_uncertainty_enumerator = loc_uncertainty.sum(dim=1)
                # div by num observations (Note: not when using batches during calculation of uncertainty below isn't possible)
                loc_uncertainty_denominator = loc_uncertainty.shape[1]

            localization_uncertainty[img] = (loc_uncertainty_enumerator / loc_uncertainty_denominator).to('cpu')

    elif localization_strategy == 'covariance':
        ## not dividing by object size
        # (hard to do in batches as each image has a different number of observations)
        # for all images
        localization_uncertainty = {}

        for img in unlabeled_imgset:
            # concat all observations and create batch dimension,
            # even though it is a batch of 1, it allows to keep the functions below to work with batches in the future
            if args.merging_method == 'pre_nms_avg':
                im_observations_cov_0 = torch.cat(observations_cov_0[img]).unsqueeze(0)
                im_observations_cov_1 = torch.cat(observations_cov_1[img]).unsqueeze(0)
            else:
                im_observations_cov_0 = torch.cat(observations_cov_0[img], dim = 1)
                im_observations_cov_1 = torch.cat(observations_cov_1[img], dim = 1)

            # todo: squeezing and unsqueezing shouldn't be necessary but PyCharm fucks it up otherwise

            # calculate traces of covariances
            trace_0, trace_1 = uncertainty_helpers.trace_covariance(im_observations_cov_0, im_observations_cov_1)

            # div the sum of the traces by the dist
            loc_uncertainty = (trace_0 + trace_1)

            # multiply the localization uncertainty by (1-P(background)) -> we want objects of which the location is uncertain
            # but which are probably foreground objects
            if args.merging_method == 'pre_nms_avg':
                im_observations_dists = torch.cat(observations_dists[img]).unsqueeze(0)
            else:
                im_observations_dists = torch.cat(observations_dists[img]).squeeze().unsqueeze(0)
            background_probs = im_observations_dists[:, :, 0]

            # todo: delete_top_n_percent_loc_uncertainty_hbdscan
            if args.merging_method == 'hbdscan' and args.delete_top_n_percent_loc_uncertainty_hbdscan:
                topk = round(args.delete_top_n_percent_loc_uncertainty_hbdscan * total_num_observations[img])
                top_k_loc_uncertainty = torch.topk(loc_uncertainty, k=topk)

            loc_uncertainty_enumerator = (loc_uncertainty * (1 - background_probs)).sum(dim=1)
            loc_uncertainty_denominator = background_probs.sum(dim=1)
            localization_uncertainty[img] = (loc_uncertainty_enumerator / loc_uncertainty_denominator).to('cpu')

    elif localization_strategy != 'none':
        raise NotImplementedError()


    # classification
    if classification_strategy == 'entropy':

        classification_uncertainty = {}

        for img in unlabeled_imgset:
            # concat all observations and create batch dimension,
            # even though it is a batch of 1, it allows to keep the functions below to work with batches in the future
            if args.merging_method == 'pre_nms_avg':
                im_observations_dists = torch.cat(observations_dists[img]).unsqueeze(0)
            else:
                im_observations_dists = torch.cat(observations_dists[img]).squeeze().unsqueeze(0)
            background_probs = im_observations_dists[:,:,0]
            # rescaled_foreground_probs = torch.nn.functional.softmax(im_observations_dists[:,:,1:], dim=2)
            if args.rescaled_foreground_probs:
                rescaled_foreground_probs = im_observations_dists[:, :, 1:] / im_observations_dists[:, :, 1:].sum(dim=2).unsqueeze(dim=2)
                entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            else:
                entropy_per_image = uncertainty_helpers.entropy(im_observations_dists, already_normalized=True)

            # entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            classification_uncertainty_enumerator = (entropy_per_image*(1-background_probs)).sum(dim=1)
            classification_uncertainty_denominator = background_probs.sum(dim=1)
            classification_uncertainty[img] = (classification_uncertainty_enumerator/classification_uncertainty_denominator).to('cpu')

    elif classification_strategy != 'none':
        raise NotImplementedError()




    if localization_strategy != 'none' and classification_strategy != 'none':
        print('calculating normalized loc+clss image uncertainty for: ',len(unlabeled_imgset),'images')
        # multiply localization and classification uncertainty per image
        # normalize loc and class uncertainties over all images to zero mean, unit variance
        class_uncertainty_tens = torch.cat([classification_uncertainty[img] for img in unlabeled_imgset])
        normalized_class_uncertainty_tens = (class_uncertainty_tens - class_uncertainty_tens.mean())/class_uncertainty_tens.std()


        loc_uncertainty_tens = torch.cat([localization_uncertainty[img] for img in unlabeled_imgset])
        normalized_loc_uncertainty_tens = (loc_uncertainty_tens - loc_uncertainty_tens.mean()) / loc_uncertainty_tens.std()

        uncertainty_per_image = (normalized_class_uncertainty_tens + normalized_loc_uncertainty_tens).to('cpu')

    elif localization_strategy == 'none':
        print('calculating class image uncertainty')
        uncertainty_per_image = torch.cat([classification_uncertainty[img] for img in unlabeled_imgset]).to('cpu')
        normalized_uncertainty_per_image = (uncertainty_per_image - uncertainty_per_image.mean()) / uncertainty_per_image.std()
        uncertainty_per_image = normalized_uncertainty_per_image

        localization_uncertainty = 'none'

    elif classification_strategy == 'none':
        print('calculating loc image uncertainty')
        uncertainty_per_image = torch.cat([localization_uncertainty[img] for img in unlabeled_imgset]).to('cpu')
        normalized_uncertainty_per_image = (uncertainty_per_image - uncertainty_per_image.mean()) / uncertainty_per_image.std()
        uncertainty_per_image = normalized_uncertainty_per_image
        classification_uncertainty = 'none'
    else:
        raise NotImplementedError()

    # change the torch default tensor back to floats
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    return uncertainty_per_image, classification_uncertainty, localization_uncertainty


def localization_stability_sample_selection(args,
                                            net,
                                            sample_select_dataset,
                                            unlabeled_idx,
                                            len_unlabeled_idx,
                                            active_learning_iteration):
    """
    This is a baseline sample selection method.
    Paper: Localization-Aware Active Learning for Object Detection" by Kao, Lee, Sen and Liu
    http://www.merl.com/publications/docs/TR2018-166.pdf

    In short, it evaluates how stable the predictions are
    under increasing gaussian noise per pixel

    This function calculates the LS+C method from their paper/results

    :param args:
    :param net:
    :param sample_select_dataset:
    :param unlabeled_idx:
    :param len_unlabeled_idx:
    :return:
    """


    if args.debug:
        args.samples_per_iter[0] = 3

    # transform needed for preprocessing
    # resizes to 300x300 and subtracts the (dataset) mean per channel from the images
    transform = BaseTransform(net.size, (104, 117, 123))

    std_devs = [8, 16, 24, 32, 40, 48]
    uncertainty_per_image = torch.zeros(len_unlabeled_idx)


    with torch.no_grad():
        for i, idx in enumerate(unlabeled_idx):  # enumerate, as we need the values, but need to loop over all elements
            if i == 0:
                start = time.time()

            im_preds = []

            # 1 normal and 6 incrasingly noisy forward passes
            for noise_level in range(7):

                # load image
                img = sample_select_dataset.pull_image_using_imageset_id(
                    (sample_select_dataset.ids[0][0], idx))

                # do the basetransform
                x = transform(img)[0]
                if noise_level != 0:
                    x = augmentations.GaussianRandomPixelNoise(x, std_devs[noise_level-1]) # -1 to get the correct index in the std_devs list (first itteration there is no noise level)
                # (cv2 reads in BGR and the network (and VGG16 base) is trained on RGB)
                    x = x.permute(2,0,1)
                else:
                    # (cv2 reads in BGR and the network (and VGG16 base) is trained on RGB)
                    x = torch.from_numpy(x).permute(2, 0, 1)
                x = Variable(x.unsqueeze(0))

                # todo: maybe faster when stacking the images and doing one forward pass.
                #   probably not much though, as the nms at the end of the network is still per image
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                    x = x.to(args.device)

                # Forward pass
                # todo changed on august 6, including the prior data in the forward
                output, num_boxes_per_class, prior_data = net(x)
                im_preds.append((output.to('cpu'),num_boxes_per_class.to('cpu')))

            ## After the increasingly noisy forward passes calculate the uncertainty for the image
            reference_output = im_preds[0]

            # use the nonzero boxes for each class
            reference_boxes = torch.cat([reference_output[0][:,cl,:int(reference_output[1][cl].item()),:].squeeze(dim=0) for cl in range(1,args.num_classes)])

            # Match noisy predictions to reference boxes for all M reference boxes
            matched_boxes = []

            # for box in reference_boxes:
            for noisy_preds_idx in range(1,7):
                noisy_boxes = torch.cat([im_preds[noisy_preds_idx][0][:, cl, :int(im_preds[noisy_preds_idx][1][cl].item()), :].squeeze(dim=0) for cl in range(1, args.num_classes)])

                # IoU overlaps is a matrix with the IoUs of all reference boxes and all noisy boxes for this noise level
                IoU_overlaps = box_utils.jaccard(reference_boxes[:,args.num_classes:], noisy_boxes[:,args.num_classes:])

                # max IoU overlap
                try:
                    matching_boxes = IoU_overlaps.max(dim =1)
                    matched_boxes.append(matching_boxes[0])
                except:
                    print('if IoU overlaps is empty, taking the maximum returns an error. Try except to skip the error.')
                    # todo: IoUs should all be zero then
                    continue


            # calculate S_b(B_0) (eq.2) (div by N noise levels for all M matched boxes there IoUs)
            Sb = torch.stack(matched_boxes).sum(dim=0)/len(matched_boxes)


            #todo: count background as class or dont? -> according to a mail from the authors, background class is excluded here.
            reference_boxes_p_max = reference_boxes[:,1:args.num_classes].max(dim=1)[0]

            # localixation stability
            # sum of (p_max of referencebox * max IoU overlap of matched box) / sum of (p_max of reference box)
            LS = torch.sum(reference_boxes_p_max * Sb) / torch.sum(reference_boxes_p_max)

            # LS+C, where C is the maximum uncertainty (1-p) out of all reference boxes
            C = 1-reference_boxes_p_max
            C = C.max(dim =0)[0]

            # we want to select images with high classification uncertainty and low localization stability:
            LS_C = C - LS
            uncertainty_per_image[i] = LS_C.item()



            if i % 10 == 0 and i != 0:
                stop = time.time()
                elapsed = stop - start
                if args.device == 'cuda':
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                    # os.system("free -m")
                else:
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                start = time.time()



    # select samples
    if not args.budget_measured_in_objects:
        if not args.user_relevance_feedback:
            top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration])
        else:
            top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration], largest= False)

        new_labeled_idx = [unlabeled_idx[idx] for idx in top_k_uncertain_images[1]]

    else:
        if not args.user_relevance_feedback:
            # make list of imgset ids ordered from most uncertain to least uncertain
            sorted_indices = uncertainty_per_image.cpu().sort(descending = True)[1].numpy()
            ordered_uncertain_images = [unlabeled_idx[i] for i in sorted_indices]
            new_labeled_idx = select_samples_with_object_budget(args,
                                                                ordered_uncertain_images,
                                                                object_budget = args.samples_per_iter[active_learning_iteration],
                                                                dataset = sample_select_dataset)

        else:
            sorted_indices = uncertainty_per_image.cpu().sort(descending = False)[1].numpy()
            ordered_certain_images = [unlabeled_idx[i] for i in sorted_indices]
            new_labeled_idx = select_samples_with_object_budget(args,
                                                                ordered_certain_images,
                                                                object_budget = args.samples_per_iter[active_learning_iteration],
                                                                dataset = sample_select_dataset)


    return new_labeled_idx


def sample_selection(args,
                     sample_select_dataset,
                     active_learning_iter):
    if args.sampling_strategy == 'random_none' and not args.budget_measured_in_objects:
        # Load image_idx of unlabeled pool as a set
        print('Selecting a fixed number of images...')
        label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                          args.annotate_all_objects,
                                          args.dataset,
                                          args)


        # randomly select samples
        unlabeled_idx = [s for s in args.sample_select_dataset_imageset_ids if
                         (s not in label_dict['train_set']
                          and s not in label_dict['val_set']['image_set_idx']
                          and s not in label_dict['seed_set']['image_set_idx'])] # should be redundant as train_set should contain seed_set as well, put in in, just to be sure

        print('Number of unlabeled images:', len(unlabeled_idx))

        # select samples
        new_labeled_idx = np.random.choice(unlabeled_idx,
                                           args.samples_per_iter[active_learning_iter],
                                           replace = False)

        new_labeled_idx = list(new_labeled_idx)

    elif args.sampling_strategy == 'random_none' and args.budget_measured_in_objects:
        # Load image_idx of unlabeled pool as a set
        label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                          args.annotate_all_objects,
                                          args.dataset,
                                          args)


        # randomly select samples
        unlabeled_idx = [s for s in args.sample_select_dataset_imageset_ids if
                         (s not in label_dict['train_set']
                          and s not in label_dict['val_set']['image_set_idx']
                          and s not in label_dict['seed_set']['image_set_idx'])] # should be redundant as train_set should contain seed_set as well, put in in, just to be sure

        print('Number of unlabeled images:', len(unlabeled_idx))

        # select samples
        random.shuffle(unlabeled_idx)
        new_labeled_idx = select_samples_with_object_budget(args,
                                                            unlabeled_idx,
                                                            args.samples_per_iter[active_learning_iter],
                                                            sample_select_dataset)


    else:
        raise NotImplementedError()

    return new_labeled_idx


def select_samples_with_object_budget(args,
                                      ordered_images,
                                      object_budget,
                                      dataset,
                                      ):
    """
    :param ordered_images: Unlabeled image names ordered from most uncertain to least uncertain
    :param object_budget: Number of objects that can be sampled
    :return: Top most uncertain images that fit within the budget. For the final image(s), if an image doesn't fit the budget anymore
    we keep going trough the images until there is an image that does.
    """
    print('Selecting a fixed number of objects')

    path_to_imset = dataset.ids[0][0]

    new_labeled_idx = []
    number_of_objects_sampled = 0
    selected_classes = [0 for cl in range(args.cfg['num_classes'] - 1)]  # -1 for background
    for imset_idx in ordered_images:
        # pull image
        budget_left = object_budget - number_of_objects_sampled

        # stop if labeling budget is depleted
        if budget_left <= 0:
            break
        im_annotations = dataset.pull_anno_using_imageset_id((path_to_imset, imset_idx))[1]

        # check whether this image fits the object budget
        objects_on_image = len(im_annotations)
        if budget_left < objects_on_image:
            continue

        new_labeled_idx.append(imset_idx)

        for anno in im_annotations:
            if args.relevant_class == None: # if all classes are used
                cl = anno[4]
                selected_classes[cl] += 1
                number_of_objects_sampled +=1


            if args.relevant_class != None: # if one class is used
                # cl= anno[4]
                # if cl == args.object_class_number:
                selected_classes[0] += 1
                number_of_objects_sampled += 1



    return new_labeled_idx



def entropy_only_baseline(args,
                            net,
                            sample_select_dataset,
                            unlabeled_idx,
                            len_unlabeled_idx,
                            active_learning_iteration):

    if args.debug:
        args.samples_per_iter[0] = 3

    # transform needed for preprocessing
    # resizes to 300x300 and subtracts the (dataset) mean per channel from the images
    transform = BaseTransform(net.size, (104, 117, 123))
    uncertainty_per_image = torch.zeros(len_unlabeled_idx)

    with torch.no_grad():
        for i, idx in enumerate(unlabeled_idx):  # enumerate, as we need the values, but need to loop over all elements
            if i == 0:
                start = time.time()

            # load image
            img = sample_select_dataset.pull_image_using_imageset_id((sample_select_dataset.ids[0][0], idx))

            # do the basetransform
            x = transform(img)[0]
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))

            # todo: maybe faster when stacking the images and doing one forward pass.
            #   probably not much though, as the nms at the end of the network is still per image
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                x = x.to(args.device)

            # Forward pass
            output, num_boxes_per_class, prior_data = net(x)
            output = output.to('cpu')
            num_boxes_per_class = num_boxes_per_class.to('cpu')
            prior_data = prior_data.to('cpu')
            dets = [output[0, cl, :int(num_boxes_per_class[cl].item()), :] for cl in range(0, args.num_classes-1)]
            nonzero_dets = torch.cat([det for det in dets if det.ge(0.0).sum() > 0]).unsqueeze(0)

            im_observations_dists = nonzero_dets[:, :,:args.cfg['num_classes']]
            background_probs = im_observations_dists[:, :, 0]
            # rescaled_foreground_probs = torch.nn.functional.softmax(im_observations_dists[:,:,1:], dim=2)
            if args.rescaled_foreground_probs:
                rescaled_foreground_probs = im_observations_dists[:, :, 1:] / im_observations_dists[:, :, 1:].sum(
                    dim=2).unsqueeze(dim=2)
                entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            else:
                entropy_per_image = uncertainty_helpers.entropy(im_observations_dists, already_normalized=True)


            # entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            classification_uncertainty_enumerator = (entropy_per_image * (1 - background_probs)).sum(dim=1)
            classification_uncertainty_denominator = background_probs.sum(dim=1)
            uncertainty_per_image[i] = (classification_uncertainty_enumerator / classification_uncertainty_denominator).to('cpu')

            if i % 10 == 0 and i != 0:
                stop = time.time()
                elapsed = stop - start
                if args.device == 'cuda':
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                    # os.system("free -m")
                else:
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                start = time.time()


    # normalize all image uncertainties to have zero mean and unit variance
    normalized_uncertainty_per_image = (uncertainty_per_image - uncertainty_per_image.mean())/uncertainty_per_image.std()

 # select samples
    if not args.density_diversity:
        if not args.budget_measured_in_objects:
            if not args.user_relevance_feedback:
                top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration])
            else:
                top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration], largest= False)

            new_labeled_idx = [unlabeled_idx[idx] for idx in top_k_uncertain_images[1]]

        else:
            if not args.user_relevance_feedback:
                # make list of imgset ids ordered from most uncertain to least uncertain
                sorted_indices = uncertainty_per_image.cpu().sort(descending = True)[1].numpy()
                ordered_uncertain_images = [unlabeled_idx[i] for i in sorted_indices]
                new_labeled_idx = select_samples_with_object_budget(args,
                                                                    ordered_uncertain_images,
                                                                    object_budget = args.samples_per_iter[active_learning_iteration],
                                                                    dataset = sample_select_dataset)

            else:
                sorted_indices = uncertainty_per_image.cpu().sort(descending = False)[1].numpy()
                ordered_certain_images = [unlabeled_idx[i] for i in sorted_indices]
                new_labeled_idx = select_samples_with_object_budget(args,
                                                                    ordered_certain_images,
                                                                    object_budget = args.samples_per_iter[active_learning_iteration],
                                                                    dataset = sample_select_dataset)



        return new_labeled_idx, normalized_uncertainty_per_image

    else:


        return None, normalized_uncertainty_per_image



def SSDKL_sample_selection(args,
                           net,
                           sample_select_dataset,
                           unlabeled_idx,
                           len_unlabeled_idx,
                           active_learning_iteration):

    if args.debug:
        args.samples_per_iter[0] = 3
        unlabeled_idx = unlabeled_idx[:10]


    classification_strategy = args.sampling_strategy.split('_')[0]
    localization_strategy = args.sampling_strategy.split('_')[1]

    # transform needed for preprocessing
    # resizes to 300x300 and subtracts the (dataset) mean per channel from the images
    transform = BaseTransform(net.size, (104, 117, 123))
    localization_uncertainty = {}
    classification_uncertainty = {}
    if args.save_variances_sample_selection:
        variances = {}
    uncertainty_per_image = torch.zeros(len_unlabeled_idx)

    with torch.no_grad():
        for i, idx in enumerate(unlabeled_idx):  # enumerate, as we need the values, but need to loop over all elements
            if i == 0:
                start = time.time()

            # load image
            img = sample_select_dataset.pull_image_using_imageset_id((sample_select_dataset.ids[0][0], idx))

            # do the basetransform
            x = transform(img)[0]
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))

            # todo: maybe faster when stacking the images and doing one forward pass.
            #   probably not much though, as the nms at the end of the network is still per image
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                x = x.to(args.device)

            # Forward pass
            output, num_boxes_per_class, prior_data = net(x)
            output = output.to('cpu')
            num_boxes_per_class = num_boxes_per_class.to('cpu')
            prior_data = prior_data.to('cpu')

            dets = [output[0, cl, :int(num_boxes_per_class[cl].item()), :] for cl in range(0, args.num_classes - 1)]
            nonzero_dets = torch.cat([det for det in dets if det.ge(0.0).sum() > 0]).unsqueeze(0)

            im_observations_dists = nonzero_dets[:, :, :args.cfg['num_classes']]
            background_probs = im_observations_dists[:, :, 0]
            # rescaled_foreground_probs = torch.nn.functional.softmax(im_observations_dists[:,:,1:], dim=2)
            if args.rescaled_foreground_probs:
                rescaled_foreground_probs = im_observations_dists[:, :, 1:] / im_observations_dists[:, :, 1:].sum(
                    dim=2).unsqueeze(dim=2)
                entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            else:
                entropy_per_image = uncertainty_helpers.entropy(im_observations_dists, already_normalized=True)


            # entropy_per_image = uncertainty_helpers.entropy(rescaled_foreground_probs, already_normalized=True)
            classification_uncertainty_enumerator = (entropy_per_image * (1 - background_probs)).sum(dim=1)
            classification_uncertainty_denominator = background_probs.sum(dim=1)
            classification_uncertainty[i] = (classification_uncertainty_enumerator / classification_uncertainty_denominator).to('cpu')

            # localization
            if localization_strategy in ['covariance-obj', 'covariance']:
                object_coords = nonzero_dets[:,:,args.num_classes:-4]
                var_coords = nonzero_dets[:,:,-4:]
                if args.save_variances_sample_selection:
                    variances[idx] = var_coords

                # summed variances per box
                summed_vars = var_coords.sum(dim = 2)

                if localization_strategy == 'covariance-obj':
                    # upper left(mu_0) and lower right corner (mu_1)
                    mu_0 = object_coords[:,:,:2]
                    mu_1 = object_coords[:,:,2:]

                    # size of object
                    object_sizes = uncertainty_helpers.dist_means_observation(mu_0, mu_1)
                    # uncertainty
                    loc_uncertainty = summed_vars / object_sizes
                else:
                    loc_uncertainty = summed_vars

                loc_uncertainty_enumerator = (loc_uncertainty * (1 - background_probs)).sum(dim=1)
                loc_uncertainty_denominator = background_probs.sum(dim=1)
                localization_uncertainty[i] = (loc_uncertainty_enumerator / loc_uncertainty_denominator).to('cpu')

            if i % 10 == 0 and i != 0:
                stop = time.time()
                elapsed = stop - start
                if args.device == 'cuda':
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                    # os.system("free -m")
                else:
                    print('Tested image {:d}/{:d} trough {:d}/{:d} in {:4f} seconds....'
                          .format(i - 9, len(unlabeled_idx), i + 1, len(unlabeled_idx), elapsed))
                start = time.time()


    # select samples
    if localization_strategy != 'none' and classification_strategy != 'none':
        print('calculating normalized loc+clss image uncertainty for: ',len(unlabeled_idx),'images using KL-Loss variances')

        class_uncertainty_tens = torch.cat([classification_uncertainty[i] for i, idx in enumerate(unlabeled_idx)])
        normalized_class_uncertainty_tens = (class_uncertainty_tens - class_uncertainty_tens.mean()) / class_uncertainty_tens.std()

        loc_uncertainty_tens = torch.cat([localization_uncertainty[i] for i, idx in enumerate(unlabeled_idx)])
        normalized_loc_uncertainty_tens = (loc_uncertainty_tens - loc_uncertainty_tens.mean()) / loc_uncertainty_tens.std()

        uncertainty_per_image = (normalized_class_uncertainty_tens + normalized_loc_uncertainty_tens).to('cpu')

    elif localization_strategy == 'none' and classification_strategy != 'none':
        print('calculated clss image uncertainty for: ',len(unlabeled_idx),'images using KL-Loss variances')
        class_uncertainty_tens = torch.cat([classification_uncertainty[i] for i, idx in enumerate(unlabeled_idx)])
        normalized_class_uncertainty_tens = (class_uncertainty_tens - class_uncertainty_tens.mean()) / class_uncertainty_tens.std()

        uncertainty_per_image = normalized_class_uncertainty_tens.to('cpu')

    if not args.budget_measured_in_objects:
        if not args.user_relevance_feedback:
            top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration])
        else:
            top_k_uncertain_images = uncertainty_per_image.topk(k=args.samples_per_iter[active_learning_iteration],
                                                                largest=False)

        new_labeled_idx = [unlabeled_idx[idx] for idx in top_k_uncertain_images[1]]

    else:
        if not args.user_relevance_feedback:
            # make list of imgset ids ordered from most uncertain to least uncertain
            sorted_indices = uncertainty_per_image.cpu().sort(descending=True)[1].numpy()
            ordered_uncertain_images = [unlabeled_idx[i] for i in sorted_indices]
            new_labeled_idx = select_samples_with_object_budget(args,
                                                                ordered_uncertain_images,
                                                                object_budget=args.samples_per_iter[active_learning_iteration],
                                                                dataset=sample_select_dataset)

        else:
            sorted_indices = uncertainty_per_image.cpu().sort(descending=False)[1].numpy()
            ordered_certain_images = [unlabeled_idx[i] for i in sorted_indices]
            new_labeled_idx = select_samples_with_object_budget(args,
                                                                ordered_certain_images,
                                                                object_budget=args.samples_per_iter[active_learning_iteration],
                                                                dataset=sample_select_dataset)


    if args.save_variances_sample_selection:
        folder = args.experiment_dir + 'sample_selection/'
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        path = folder + 'variances_SSDKL_al-iter_'+str(args.al_iteration)+'.pickle'
        with open(path, 'wb') as out:
            pickle.dump(variances, out)

        path = folder+ 'loc_uncertainties_SSDKL_al-iter_'+str(args.al_iteration)+'.pickle'
        with open(path, 'wb') as out:
            pickle.dump(localization_uncertainty, out)

        path = folder+ 'class_uncertainties_SSDKL_al-iter_'+str(args.al_iteration)+'.pickle'
        with open(path, 'wb') as out:
            pickle.dump(classification_uncertainty, out)

        path = folder+ 'chosen_samples_and_all_unlabeled_samples_SSDKL_al-iter_'+str(args.al_iteration)+'.pickle'
        with open(path, 'wb') as out:
            pickle.dump((new_labeled_idx,unlabeled_idx), out)

    return new_labeled_idx




def train_model(args,
                train_dataset,
                val_dataset,
                ensemble_idx):

    # set loss counters
    train_loc_loss_list = []
    train_conf_loss_list = []
    val_loc_loss_list = []
    val_conf_loss_list = []
    if args.modeltype == 'SSD300KL':
        args.summary['train_model']['predicted_alphas'] = []

    # load the labeled samples
    label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                      args.annotate_all_objects,
                                      args.dataset,
                                      args)
    labeled_idx = label_dict['train_set']

    # get the subset of the data that has been 'labeled'
    if not args.train_on_full_dataset:
        train_set = helpers.data_subset(train_dataset,
                                        labeled_idx)
    else:
        train_set = train_dataset


    # loss function
    criterion = MultiBoxLoss(num_classes = args.cfg['num_classes'],
                             overlap_thresh = 0.5,
                             prior_for_matching = True,
                             bkg_label = 0,
                             neg_mining = True,
                             neg_pos = 3,
                             neg_overlap = 0.5,
                             encode_target = False,
                             use_gpu = args.device == 'cuda',
                             modeltype=args.modeltype)


    # make net
    net = helpers.build_train_net(args,
                                  ensemble_idx)


    # Optimization algo
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    elif args.optimizer == 'ADAM':
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               amsgrad=True,
                               weight_decay=args.weight_decay)


    else:
        raise NotImplementedError()

    # make dataloaders
    train_data_loader = helpers.InfiniteDataLoader(train_set,
                                        # batch_size= min(len(labeled_idx), args.batch_size),
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True,
                                        collate_fn=detection_collate,
                                        pin_memory=True,
                                        drop_last = False
                                        )

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                        args.batch_size,
                                        num_workers=args.num_workers, # 4 seems fastest, no 'running out of batches' problems in eval either..
                                        shuffle=True,
                                        collate_fn=detection_collate,
                                        pin_memory=True,
                                        drop_last=False
                                        )


    # # validation loss frequency
    if len(labeled_idx) > 1:
        validation_loss_frequency = len(labeled_idx) - 1
    else:
        validation_loss_frequency = 100


    # create batch iterator
    train_batch_iterator = iter(train_data_loader)


    # args.train_iterations = int(len(train_split) * args.train_epochs / args.batch_size)

    print('Active learning iteration: ', str(args.al_iteration), '\t\t training for: ', str(args.train_epochs),
          ' epochs\t\t with batch size: ',str(args.batch_size),'\t\t on:', len(train_set.ids), ' images')
    print('\n\n')

    # counters
    lr_step = 0
    total_iterations = 0
    no_improvement_counter = 0
    decrease_lr = False
    stop_training = False

    # training loop
    if args.fixed_number_of_epochs == None: # early stopping
        for epoch in range(args.train_epochs):
            print("New epoch, Local time: ", datetime.datetime.now())

            # decrease learning rate after every 4 epochs
            if (decrease_lr):
                lr_step += 1
                helpers.adjust_learning_rate(args,
                                             optimizer,
                                             lr_step)
                decrease_lr = False


            for iteration in range(round(len(labeled_idx)/args.batch_size)): # todo: this was without /args.batch_size
                # validation pass every epoch
                if (iteration == 0 and epoch !=0) or args.debug: #todo delete args.debug
                    if args.early_stopping_condition == 'just_lowest_val':
                        # do full validation pass
                        print('Doing validation pass...')
                        val_loss, val_loc_loss,val_conf_loss, save = helpers.just_keep_lowest_val_loss_weights(args,
                                                                                                               net,
                                                                                                               criterion,
                                                                                                               val_data_loader,
                                                                                                               ensemble_idx)

                        print('Finished validation pass')

                        val_conf_loss_list.append(val_conf_loss)
                        val_loc_loss_list.append(val_loc_loss)
                        if args.debug == True: # todo remove
                            save = True
                        if save:
                            # save weights
                            path = str(args.experiment_dir +
                                       'weights/' +
                                       str(args.modeltype) + '_al-iter_' +
                                       str(args.al_iteration) + '_train-loss_' +
                                       str(loss.item()) + '_' + '_val-loss_' +
                                       str(val_loss) + '_' +
                                       str(args.dataset) + '_train-iter_' +
                                       str(total_iterations) + '_ensemble-id_' +
                                       str(ensemble_idx)+'_.pth')

                            # save nn weights
                            path_to_weights = helpers.save_weights(net,
                                                                   args,
                                                                   path,
                                                                   mode = 'neural_net')

                            # if it is the first path to be saved in paths to weights for this model in the ensemble, we need to append
                            try:
                                args.paths_to_weights[ensemble_idx] = path_to_weights
                            except:
                                args.paths_to_weights.append(path_to_weights)

                        else:
                            no_improvement_counter += 1
                            print('No improvement past ', no_improvement_counter ,' validation passes...')

                        if no_improvement_counter > args.epochs_no_improvement and lr_step < 2: # no improvement past epoch, decrease lr
                            decrease_lr = True
                            no_improvement_counter = 0

                        if no_improvement_counter > args.epochs_no_improvement and lr_step > 1: # no improvement after lr lowered twice, stop training
                            stop_training = True


                    else:
                        raise NotImplementedError()

                if stop_training:
                    print('No improvement\n\n Early stopping')
                    break

                total_iterations += 1

                if args.debug and iteration > 11:  # todo: remove
                    break

                images, targets = next(train_batch_iterator)

                # create variables for images and targets
                if args.device == 'cuda':
                    with torch.cuda.device(
                            0):  # for data parralel (multi-gpu), the data must be on the first cuda device
                        images = images.cuda()
                        with torch.no_grad():
                            targets = [ann.cuda() for ann in targets]

                    images = Variable(images)
                    with torch.no_grad():
                        targets = [Variable(target) for target in targets]
                else:
                    images = Variable(images)
                    with torch.no_grad():
                        targets = [Variable(ann) for ann in targets]

                # forward pass
                t0 = time.time()
                out = net(images)

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets, args)


                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                t1 = time.time()

                loc_loss = loss_l.item()  # FOR VISDOM: loc_loss += loss_l.item()
                conf_loss = loss_c.item()  # same

                # print losses
                if iteration % 10 == 0:
                    print('timer: %.4f sec.' % (t1 - t0))
                    print('Epoch ' + str(epoch) + '  iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))

                    # store train losses for later plotting
                    train_loc_loss_list.append(loc_loss)
                    train_conf_loss_list.append(conf_loss)
            if stop_training:
                break

        # after training, delete non optimal weights and save the relevant data
        best_nn_weights = helpers.delete_non_optimal_weights(args,
                                                         mode = 'neural_net',
                                                         ensemble_id = ensemble_idx)

        # save nn weights for sample selection
        args.paths_to_weights[ensemble_idx] = args.experiment_dir + 'weights/' + best_nn_weights

    # fixed number of epochs for training
    elif args.fixed_number_of_epochs:

        adj_lr = [round(args.fixed_number_of_epochs*(2/3)),round(args.fixed_number_of_epochs*(5/6))]

        for epoch in range(args.fixed_number_of_epochs):
            print("New epoch, Local time: ", datetime.datetime.now())

            # decrease learning rate after every 4 epochs
            if epoch in adj_lr:

                lr_step += 1
                helpers.adjust_learning_rate(args,
                                             optimizer,
                                             lr_step)


            for iteration in range(max(1, round(len(train_set.ids)/args.batch_size))): #TODO TRAINING ADJUSTED!!

                total_iterations += 1

                if args.debug and iteration > 11:  # todo: remove
                    break

                images, targets = next(train_batch_iterator)

                # create variables for images and targets
                if args.device == 'cuda':
                    with torch.cuda.device(
                            0):  # for data parralel (multi-gpu), the data must be on the first cuda device
                        images = images.to('cuda:0')
                        with torch.no_grad():
                            targets = [ann.to('cuda:0') for ann in targets]

                    images = Variable(images)
                    with torch.no_grad():
                        targets = [Variable(target) for target in targets]
                else:
                    images = Variable(images)
                    with torch.no_grad():
                        targets = [Variable(ann) for ann in targets]


                # forward pass
                t0 = time.time()
                out = net(images)

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                #todo: delete  later - only for debug purposes
                if args.modeltype == 'SSD300KL' and (epoch == 0 and iteration == 0):
                    print('loss_l: ',loss_l)
                    print('loss_c: ',loss_c)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                t1 = time.time()

                loc_loss = loss_l.item()
                conf_loss = loss_c.item()
                if args.debug:
                    # store train losses for later plotting
                    train_loc_loss_list.append(loc_loss)
                    train_conf_loss_list.append(conf_loss)

                # print losses
                if iteration % 10 == 0:
                    print('timer: %.4f sec.' % (t1 - t0))
                    print('Epoch ' + str(epoch) + '  iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item())
                          + ' || conf loss: %.4f ||' % (conf_loss) + ' || loc loss: %.4f ||' % (loc_loss))
                    if args.train_basenets:
                        print('loc_loss: ', loc_loss,'      val loss: ', val_loss)
                    # store train losses for later plotting
                    train_loc_loss_list.append(loc_loss)
                    train_conf_loss_list.append(conf_loss)

                # save weights
                val_loss = 999
                path = str(args.experiment_dir +
                           'weights/' +
                           str(args.modeltype) + '_al-iter_' +
                           str(args.al_iteration) + '_train-loss_' +
                           str(loss.item()) + '_' + '_val-loss_' +
                           str(val_loss) + '_' +
                           str(args.dataset) + '_train-iter_' +
                           str(total_iterations) + '_ensemble-id_' +
                           str(ensemble_idx) + '_.pth')

                # validation pass
                if (iteration == 0 and epoch > (args.fixed_number_of_epochs/2)) or (args.debug and iteration == 0):
                    if args.early_stopping_condition == 'just_lowest_val':
                        # do full validation pass
                        print('Doing validation pass...')
                        val_loss, val_loc_loss,val_conf_loss, save = helpers.just_keep_lowest_val_loss_weights(args,
                                                                                                               net,
                                                                                                               criterion,
                                                                                                               val_data_loader,
                                                                                                               ensemble_idx)

                        print('Finished validation pass')
                        val_loss = val_conf_loss+val_loc_loss
                        print('val los:  ', val_loss)
                        val_conf_loss_list.append(val_conf_loss)
                        val_loc_loss_list.append(val_loc_loss)

                        # save weights
                        path = str(args.experiment_dir +
                                   'weights/' +
                                   str(args.modeltype) + '_al-iter_' +
                                   str(args.al_iteration) + '_train-loss_' +
                                   str(loss.item()) + '_' + '_val-loss_' +
                                   str(val_loss) + '_' +
                                   str(args.dataset) + '_train-iter_' +
                                   str(total_iterations) + '_ensemble-id_' +
                                   str(ensemble_idx)+'_.pth')
                        if save:
                            # save nn weights
                            path_to_weights = helpers.save_weights(net,
                                                                   args,
                                                                   path,
                                                                   mode = 'neural_net')


            if args.save_every_epoch:
                # save nn weights
                path_to_weights = helpers.save_weights(net,
                                                       args,
                                                       path,
                                                       mode='neural_net')

        # after training, delete non optimal weights and save the relevant data
        best_nn_weights = helpers.delete_non_optimal_weights(args,
                                                             mode = 'neural_net',
                                                             ensemble_id = ensemble_idx)

        path_to_weights = args.experiment_dir + 'weights/' + best_nn_weights




        # if it is the first path to be saved in paths to weights for this model in the ensemble, we need to append
        try:
            args.paths_to_weights[ensemble_idx] = path_to_weights
        except:
            args.paths_to_weights.append(path_to_weights)


    elif args.fixed_number_of_train_iterations:
        for iteration in range(args.fixed_number_of_train_iterations):
            total_iterations += 1


            if iteration == 26666:
                print('Adjust learning rate: ')
                helpers.adjust_learning_rate(args,
                                             optimizer,
                                             lr_step)


            images, targets = next(train_batch_iterator)

            # create variables for images and targets
            if args.device == 'cuda':
                with torch.cuda.device(
                        0):  # for data parralel (multi-gpu), the data must be on the first cuda device
                    images = images.cuda()
                    with torch.no_grad():
                        targets = [ann.cuda() for ann in targets]

                images = Variable(images)
                with torch.no_grad():
                    targets = [Variable(target) for target in targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]

            # forward pass
            t0 = time.time()
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()

            loc_loss = loss_l.item()  # FOR VISDOM: loc_loss += loss_l.item()
            conf_loss = loss_c.item()  # same

            # print losses
            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('Iteration: ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item())
                      + ' || conf loss: %.4f ||' % (conf_loss) + ' || loc loss: %.4f ||' % (loc_loss))

                # store train losses for later plotting
                train_loc_loss_list.append(loc_loss)
                train_conf_loss_list.append(conf_loss)

            # validation pass
            if (iteration % validation_loss_frequency == 0 and iteration > 3000) or args.debug:  # todo delete args.debug
                if args.early_stopping_condition == 'just_lowest_val':
                    # do full validation pass
                    print('Doing validation pass...')
                    val_loss, val_loc_loss, val_conf_loss, save = helpers.just_keep_lowest_val_loss_weights(args,
                                                                                                            net,
                                                                                                            criterion,
                                                                                                            val_data_loader,
                                                                                                            ensemble_idx)

                    print('Finished validation pass')

                    val_conf_loss_list.append(val_conf_loss)
                    val_loc_loss_list.append(val_loc_loss)

                    if args.debug == True:  # todo remove
                        save = True

                    if save:
                        # save weights
                        path = str(args.experiment_dir +
                                   'weights/' +
                                   str(args.modeltype) + '_al-iter_' +
                                   str(args.al_iteration) + '_train-loss_' +
                                   str(loss.item()) + '_' + '_val-loss_' +
                                   str(val_loss) + '_' +
                                   str(args.dataset) + '_train-iter_' +
                                   str(total_iterations) + '_ensemble-id_' +
                                   str(ensemble_idx) + '_.pth')

                        # save nn weights
                        path_to_weights = helpers.save_weights(net,
                                                               args,
                                                               path,
                                                               mode='neural_net')

                        # if it is the first path to be saved in paths to weights for this model in the ensemble, we need to append
                        try:
                            args.paths_to_weights[ensemble_idx] = path_to_weights
                        except:
                            args.paths_to_weights.append(path_to_weights)

    args.summary['train_model']['losses'][ensemble_idx] = {}
    args.summary['train_model']['losses'][ensemble_idx]['val_loc_loss'] = val_loc_loss_list
    args.summary['train_model']['losses'][ensemble_idx]['val_conf_loss'] = val_conf_loss_list
    args.summary['train_model']['losses'][ensemble_idx]['train_loc_loss'] = train_loc_loss_list
    args.summary['train_model']['losses'][ensemble_idx]['train_conf_loss'] = train_conf_loss_list
    args.summary['train_model']['total_iterations'] = total_iterations


def density_sampling(args,
                     dataset,
                     unlabeled_idx):
    """
    Density sampling by mean similarity with other images in unlabeled pool. Similarities are pre-calculated

    :return:
    """

    # create density of the unlabeled pool
    density = helpers.update_density_per_imageset_per_al_iter(dataset,
                                                              load_dir_similarities=args.similarity_dir,
                                                              unlabeled_images=unlabeled_idx)

    # make tensor of densities in same order as the uncertainties, so they can be easily combined if needed
    density_per_image = torch.zeros(len(unlabeled_idx))
    for i, idx in enumerate(unlabeled_idx):
        density_per_image[i] = density[idx]

    # normalize
    # (class_uncertainty_tens - class_uncertainty_tens.mean()) / class_uncertainty_tens.std()
    density_per_image_normalized = (density_per_image -density_per_image.mean()) / density_per_image.std()

    return density_per_image_normalized.to('cpu')
