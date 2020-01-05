import os
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import time
import sys

import torch
from torch.autograd import Variable

import data
from . import helpers


def eval(test_dataset, args, net, al_iteration, eval_ensemble_idx = 99999, epochs_test = False, train_iters = None, use_dataset_image_ids = False):
    """
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    Slightly adjusted to fit in this active learning module
    """
    print('start VOC eval')

    num_images = len(test_dataset)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    if args.dataset in ['VOC07', 'VOC12']:
        labelmap = data.VOC_CLASSES
    elif args.dataset == 'VOC07_1_class':
        labelmap = [args.relevant_class]
    elif args.dataset == 'VOC07_6_class':
        labelmap = args.labelmap
    else:
        raise NotImplementedError()


    args.summary['eval_model']['num_images_eval'] = num_images
    args.summary['eval_model']['num_objects_eval'] = 'todo'
    args.summary['eval_model']['APs'] = {}

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': helpers.Timer(), 'misc': helpers.Timer()}

    output_dir = args.experiment_dir + 'eval/'
    print('output dir ', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if epochs_test:
        det_file = os.path.join(output_dir,'al-iter_'+str(al_iteration)+'_ensemble_'+str(args.eval_ensemble_idx)+'_'+str('todo')+'_detections.pkl')
    else:
        det_file = os.path.join(output_dir,'al-iter_'+str(al_iteration)+'_ensemble_'+str(args.eval_ensemble_idx)+str()+'_detections.pkl')

    # if already done the detection passes with this network.
    if os.path.isfile(det_file):
        with open(det_file, 'rb') as file:
            all_boxes = pickle.load(file)

    else:
        for i in range(num_images):
            im, gt, h, w = test_dataset.pull_item(i)

            x = Variable(im.unsqueeze(0))

            if args.cuda and torch.cuda.is_available():
                x = x.cuda()

            _t['im_detect'].tic()

            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)
            # set detections back to cpu
            if args.cuda and torch.cuda.is_available():
                detections = detections.to('cpu')

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :] # shape [200,5]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() # takes the detections that have confidence > 0. and expands to (5, 200) and then transposes
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s \t al iteration: {:d} \t ensemble_idx {:d}'.format(i,
                                                        num_images, detect_time, int(al_iteration), int(args.eval_ensemble_idx)))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


    print('Evaluating detections')
    evaluate_detections(all_boxes,
                        output_dir,
                        test_dataset,
                        args,
                        labelmap,
                        use_dataset_image_ids)


def evaluate_detections(box_list, output_dir, dataset, args, labelmap, use_dataset_image_ids):
    """
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    Slightly adjusted to fit in this active learning module
    """
    if args.dataset in ['VOC07','VOC07_1_class','VOC07_6_class']:

        YEAR = '2007'
        devkit_path = args.dataset_root + 'VOC' + YEAR

        write_voc_results_file(box_list,
                               dataset,
                               labelmap,
                               devkit_path,
                               args)

        do_python_eval(output_dir,
                       False,  # use VOC07 metrics
                       devkit_path,
                       labelmap,
                       args,
                       dataset,
                       use_dataset_image_ids)
    else:
        raise NotImplementedError()

def write_voc_results_file(all_boxes,
                           dataset,
                           labelmap,
                           devkit_path,
                           args):
    """
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    """

    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template('test',
                                                 cls,
                                                 devkit_path,
                                                 args)

        # if already made the results files with this network.
        if os.path.isfile(filename):
            continue

        else:
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dataset.ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir,
                   use_07,
                   devkit_path,
                   labelmap,
                   args,
                   dataset,
                   use_dataset_image_ids):
    """
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    Slightly adjusted to fit in this active learning module
    """
    annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml')
    if type(args.imageset_test) == list and len(args.imageset_test) == 1:
        imagesetfile = args.imageset_test[0][1]
    else:
        imagesetfile = args.imageset_test
    imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets',
                              'Main', '{:s}.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')

    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    iou_thresholds = [0.3]
    iou_thresholds.extend(list(np.linspace(0.5,0.95,10)))

    for iou_threshold in iou_thresholds:
        print('IoU threshold: ',str(iou_threshold),'\n_______________\n')
        args.summary['eval_model']['APs'][str(iou_threshold)] = {}

        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(labelmap):
            filename = get_voc_results_file_template('test', cls, devkit_path, args) # results file
            rec, prec, ap = voc_eval(
               filename, annopath, imgsetpath.format(imagesetfile), cls, cachedir,
               ovthresh=iou_threshold, use_07_metric=use_07_metric, dataset= dataset, use_dataset_image_ids=use_dataset_image_ids) # todo: imageset_file: '/home/jasper/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
            # rec,prec,ap = 0.1,0.2,0.3

            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

            #write summary average precissions
            args.summary['eval_model']['APs'][str(iou_threshold)][str(cls)] = ap

        # exclude classes without predictions
        aps = [ap for ap in aps if ap != -1.]
        args.summary['eval_model']['APs'][str(iou_threshold)]['mAP'] = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
        print('\n\n\n')

    # calculate mmAP (coco definition mAP)
    args.summary['eval_model']['APs']['mmAP'] = 0
    for key, value in args.summary['eval_model']['APs'].items():
        if key != 'mmAP':
            args.summary['eval_model']['APs']['mmAP'] += args.summary['eval_model']['APs'][key]['mAP']
    args.summary['eval_model']['APs']['mmAP'] /= 10


def get_voc_results_file_template(image_set, cls, devkit_path, args):
    """
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    Slightly adjusted to fit in this active learning module
    """

    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)

    filedir = args.experiment_dir + 'eval/results/al-iter_'+str(args.al_iteration)+'/ensemble_idx_'+args.eval_ensemble_idx
    # filedir = os.path.join(devkit_path, 'results') # old filedir from Max De Groot
    if not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok = True)
    path = os.path.join(filedir, filename)
    return path

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True,
             dataset = None,
             use_dataset_image_ids = False):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)


    NOTE: largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
    Slightly adjusted to fit in this active learning module
"""

# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl') # cachefile of correct annotations/truth values.
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    if use_dataset_image_ids:
        for imagename in dataset.ids:
            imagename = imagename[1]
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
    else:
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

    # read detections (see results folder in VOCDevkit)
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]] # can result in keyerror if: class recs doesn't have the image_id (class_rec is gt for all images in imagenames, where recs is taken from the cache file) todo
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        # note that below default values of -1 can cause negative mAPs.. Not sure why you would want this anyways..
        # rec = -1.
        # prec = -1.
        # ap = -1.
        rec = 0.
        prec = 0.
        ap = 0.
    return rec, prec, ap


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file
    largely copied from eval.py from the original pytorch SSD repository: https://github.com/amdegroot/ssd.pytorch
     """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


