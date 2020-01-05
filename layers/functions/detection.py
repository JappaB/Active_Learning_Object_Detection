import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg
from active_learning_package import uncertainty_helpers
import math


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh,
                 def_forward,merging_method,sampling_strategy,modeltype):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k

        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

        ## Active Learning Package variables
        self.def_forward = def_forward
        self.merging_method = merging_method
        self.sampling_strategy = sampling_strategy
        self.do_prob_dist_forward = False
        self.modeltype = modeltype

        if self.merging_method != 'pre_nms_avg' and \
                (self.sampling_strategy == 'p-max_localization-stability'
                 or self.sampling_strategy == 'no_ensemble_entropy-only'
                 or self.sampling_strategy in ['none_covariance', 'none_covariance-obj','entropy_covariance', 'entropy_covariance-obj'])\
                and self.modeltype != 'SSD300KL':
            self.do_prob_dist_forward = True




    def forward(self, loc_data, conf_data, prior_data, alphas = None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior (default) boxes and variances from priorbox layers
                Shape: [1,num_priors,4]

            Only when using KL-loss:
            alpha: α = log(σ^{2}) where σ^2 is the standard deviation per bounding box coordinate. The log is used during
                training to avoid gradient exploding
                shape: [1, num_priors, 4]
        """
        # the normal forward pass, as decribed in SSD paper
        if self.def_forward:

            # Merging method = None by default, if None has been passed trough
            output = self.default_forward(loc_data, conf_data, prior_data)
            return output

        else:
            # if not a regular forward -> ensemble of SSDs can be used to merge bounding boxes
            # into probabilistic object detections

            if self.merging_method == 'pre_nms_avg' and \
                not self.do_prob_dist_forward and\
                    self.modeltype != 'SSD300KL':
                # returns all locs and preds, without applying non maximum suppression to allow for pre-nms averaging
                # for more information, see paper: Miller et al - Benchmarking Sampling-based Probabilistic Object Detectors
                output_tup = (loc_data, conf_data, prior_data)
                return output_tup


            # elif self.merging_method in ['BSAS','Hungarian'] or 'p-max_localization-stability':
            elif self.do_prob_dist_forward:
                output, num_boxes_per_class = self.full_prob_dist_forward(loc_data,conf_data, prior_data)
                # output_tup = (output, prior_data) # todo: do I really need prior data for BSAS merging? -> only used for nms, which is already performed here or also for IoU calculation??
                return output, num_boxes_per_class, prior_data

            elif self.modeltype == 'SSD300KL':
                output, num_boxes_per_class = self.full_prob_KL_forward(loc_data, conf_data, prior_data, alphas)
                return output, num_boxes_per_class, prior_data
            else:
                raise NotImplementedError()


    def full_prob_KL_forward(self, loc_data, conf_data, prior_data, alphas):
        """
        Largely copief from the forward with the full probability distribution (full_prob_dist_forward). However,
        The bounding boxes are in point-form (x1,y1,x2,y2) instead of center-form (cx, cy, w, h) and for each corner also
        a standard deviation is returned.


        :param loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
        :param alphas: (tensor)  α = log(σ^{2}) where σ^2 is the standard deviation per bounding box coordinate.
                The log is used during training to avoid gradient exploding
                Shape: [batch,num_priors*4]
        :param conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
        :param prior_data:(tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        :return:

        the default forward returns the top-k (200) detections PER CLASS. The probability distribution over the classes
        is not returned, only the probability for a given detection for a given class.

        output in this functon is [image_ids, class_id ,detection_id,conf_dist + bb], where bb thus has 8 params (x1, std_x1, ...)
        where in the default forward it is  [image_ids, class_id ,detection_id,conf_score+bb]
        """

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, self.num_classes + 8)

        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # todo: why use the confidence mask? (not found in paper (found in paper...))
        # => makes it a lot faster, no nms for all boxes

        # very useful to filter out the nonzero boxes later
        num_boxes_per_class = torch.zeros(self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            # Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
            # These are the class agnostic bounding boxes!
            decoded_boxes = decode(loc_data[i], prior_data, self.variance, self.modeltype)
            conf_scores = conf_preds[i].clone()

            # For each class, perform nms
            for cl in range(1, self.num_classes):

                # self.conf_tresh is 0.01
                # gt: Computes input > other element-wise. source: https://pytorch.org/docs/stable/torch.html#torch.gt
                c_mask = conf_scores[cl].gt(
                    self.conf_thresh)  # confidence mask, speeds up processing by not applying nms

                # to all bounding boxes
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                alphas_cl = alphas[i][l_mask].view(-1,4)

                # idx of highest scoring and non-overlapping boxes per class (nms)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                # use c_mask to get the conf_scores per bounding box of the other classes for all bbs that exceed the conf treshold for this clas
                conf_scores2 = conf_scores[:, c_mask]
                assert math.isclose(conf_scores2.sum().item(), conf_scores2.shape[1], rel_tol=1e-03), \
                    "Sum of the probabilities over the classes for each detection must be (relatively close to) 1"

                distributions = conf_scores2[:, ids[:count]]
                num_boxes_per_class[cl] = count

                # idx of LOWEST scoring and non-overlapping boxes per class for boxes that don't belong
                # to the background class with a probability larger than the object treshold (IMPORTANT: Background = class 0)

                # [image_id,class_id,detection_id,conf_dist+bb]
                # [1,1,200,21+8]
                output[i, cl, :count, :self.num_classes] = distributions.permute(1,
                                                                                 0)  # permute reorders axes (here: 1 to 0 and 0 to 1)
                output[i, cl, :count, self.num_classes:-4] = boxes[ids[:count]]
                # transform alphas to variances:  α = log(σ^{2}) ->   σ = exp(.5 * α)
                output[i, cl, :count, -4:] = torch.exp(alphas_cl[ids[:count]]*.5)

                #todo [DONE]:
                # Example from original KL-Loss
                # def bbox_std_transform_xyxy(boxes, bbox_epsilon, describ=False):
                #     # bbox_std = np.exp(bbox_epsilon)
                #     if cfg.PRED_STD_LOG:
                #         bbox_std = np.exp(bbox_epsilon / 2.)


        # use cl 5 of image 1 to check: output[0,5,:5,:21]
        return output, num_boxes_per_class  # shape (pasval VOC) [1,21,200,25] = [1 = batch, classes+background class, top_k bounding boxes, 29(class_dist + bounding box_coords + coords_std))]

    def full_prob_dist_forward(self,loc_data,conf_data, prior_data):
        """
        This function is largely copied from the default forward. However, the default forward returns the top-k (200)
        detections PER CLASS. The probability distribution over the classes is not returned, only the probability for
        a given detection for a given class.

                Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]

        the default forward returns the top-k (200) detections PER CLASS. The probability distribution over the classes
        is not returned, only the probability for a given detection for a given class.

        output in this functon is [image_ids, class_id ,detection_id,conf_dist + bb]
        where in the default forward it is  [image_ids, class_id ,detection_id,conf_score+bb]

        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, self.num_classes + 4)

        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # todo: why use the confidence mask? (not found in paper (found in paper...))
        # => makes it a lot faster, no nms for all boxes

        # very useful to filter out the nonzero boxes later
        num_boxes_per_class = torch.zeros(self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            # Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
            # These are the class agnostic bounding boxes!
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)

            conf_scores = conf_preds[i].clone()

            # For each class, perform nms
            for cl in range(1, self.num_classes):

                # self.conf_tresh is 0.01
                # gt: Computes input > other element-wise. source: https://pytorch.org/docs/stable/torch.html#torch.gt
                c_mask = conf_scores[cl].gt(
                    self.conf_thresh)  # confidence mask, speeds up processing by not applying nms

                # to all bounding boxes
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                '''what is ids variable that is returned here in relation to the indices in the original conf_preds variable

                ids are the maximum ids in boxes (gt > 0.01). The ids that are not suppressed by nms
                count is how many boxes there are that are not nms'ed?
                count is hoeveel objecten er zijn van deze klasse die niet overlappen, op deze foto. 
                nms gaat namelijk vanaf de grootste confidence naar de kleinste en als ze genoeg overlappen, 
                dan wordt de een na grootste weg gegooid voor deze klasse

                '''

                # idx of highest scoring and non-overlapping boxes per class (nms)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                # use c_mask to get the conf_scores per bounding box of the other classes for all bbs that exceed the conf treshold for this clas
                conf_scores2 = conf_scores[:, c_mask]
                assert math.isclose(conf_scores2.sum().item(), conf_scores2.shape[1], rel_tol=1e-03), \
                    "Sum of the probabilities over the classes for each detection must be (relatively close to) 1"

                distributions = conf_scores2[:, ids[:count]]
                num_boxes_per_class[cl] = count


                # idx of LOWEST scoring and non-overlapping boxes per class for boxes that don't belong
                # to the background class with a probability larger than the object treshold (IMPORTANT: Background = class 0)

                # [image_id,class_id,detection_id,conf_dist+bb]
                # [1,1,200,21+4]
                output[i, cl, :count, :self.num_classes] = distributions.permute(1,
                                                                                 0)  # permute reorders axes (here: 1 to 0 and 0 to 1)
                output[i, cl, :count, self.num_classes:] = boxes[ids[:count]]


        # use cl 5 of image 1 to check: output[0,5,:5,:21]
        return output, num_boxes_per_class  # shape (pasval VOC) [1,21,200,25] = [1 = batch, classes+background class, top_k bounding boxes, 25(class_dist + bounding box))]

    def default_forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers todo prior box variances??
                Shape: [1,num_priors,4]

        the default forward returns the top-k (200) detections PER CLASS. The probability distribution over the classes
        is not returned, only the probability for a given detection for a given class.
        :returns:
            output:
                shape: [image_id,class_id,detection_id,conf_score+bb]

        """

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5) # 5 is for the bounding box => 4 corners and the class
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # why use the confidence mask?
        # => makes it a lot faster, no nms for all boxes => also used in paper
        for i in range(num):
            # Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
            # These are the class agnostic bounding boxes!
            #[8732,4]
            decoded_boxes = decode(loc_data[i], prior_data, self.variance, self.modeltype)
            #[21,8732]
            conf_scores = conf_preds[i].clone()

            # For each class, perform nms
            for cl in range(1, self.num_classes):

                # self.conf_tresh is 0.01
                # gt: Computes input > other element-wise. source: https://pytorch.org/docs/stable/torch.html#torch.gt
                c_mask = conf_scores[cl].gt(self.conf_thresh) #confidence mask, speeds up processing by not applying nms

                # to all bounding boxes
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue


                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # idx of highest scoring and non-overlapping boxes per class (nms)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                # [image_id,class_id,detection_id,conf+bb]
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)

        flt = output.contiguous().view(num, -1, 5) # [1,4200,5]
        _, idx = flt[:, :, 0].sort(1, descending=True) # sort over ALL confidences (not per class)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).   unsqueeze(-1).expand_as(flt)].fill_(0) # take top_k

        # use cl 5 of image 1 to check: output[0,5,:5,:21]
        return output  # shape (pasval VOC) [1,21,200,5] = [1 = batch, classes+background class, top_k bounding boxes, 5(bounding box + class))]
