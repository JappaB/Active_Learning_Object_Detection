import torch.nn.functional as F
import utils.augmentations as augmentations
import torch

def entropy(confs, already_normalized = True):
    """
    https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
    softmax proof: https://math.stackexchange.com/questions/331275/softmax-function-and-modelling-probability-distributions

    :param confs: (tensor)
                shape: (batch, observations, class_probabilities) where class probabilities are real probabilities (already normalized)
    :return: H: (tensor) entropy
                shape: (batch, observations)
    """
    # tested with a uniform and a peak distribution in a tensor

    if already_normalized == False:
        H = F.softmax(confs, dim=2) * F.log_softmax(confs, dim=2)
        H = H.sum(dim=2) * -1.0
    else:
        H = confs * torch.log(confs)
        H = H.sum(dim=2) * -1.0

    return H

def trace_covariance(cov_0, cov_1):
    """
    https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
    see trace calculation, however, now we keep the first two dimensions (batches and observations) as free variables


        TODO: below is just
    args:
        cov_0: (tensor)
            shape: [batch, observations, 2 ,2] #last two dimensions are xx,xy and xy,yy
        cov_1: tensor)
            shape: [batch, observations, 2 ,2] #last two dimensions are xx,xy and xy,yy
    :return:
        traces_0: (tensor)
            shape: [batch, observation]
        traces_1: (tensor)
            shape: [batch, observation]
    """

    # todo: assert that the trace must be positive
    traces_0 = torch.einsum('boxx->bo',cov_0)
    traces_1 = torch.einsum('boxx->bo',cov_1)

    return traces_0, traces_1




def dist_means_observation(mu_0,mu_1):
    """
    calculate the (euclidean) distance between the mean of the upper left corner (mu_0) and lower right corner (mu_1) of the bounding box

    args:
        mu_0:
            shape: [batch, observations, 2] where the last dim is x1y1
        mu_1:
            shape: [batch, observations, 2] where the last dim is x2y2
    :return:
        distances:
                shape:
    """



    mu_1_minus_0 = mu_1-mu_0
    squared = torch.pow(mu_1_minus_0,2)
    summed = squared.sum(dim=2)
    distances = torch.pow(summed,0.5)

    return distances

def means_observation(observations):
    """
    This function is exactly the same as the means_covs_observation below, without the cov part.
    """
    max_boxes = observations.shape[2]
    num_observations = observations.shape[1]
    num_batches = observations.shape[0]

    # per bounding box, sum each individual coordinate
    summed_coordinates = observations.sum(dim=2)
    zeros = observations.le(0.)
    zeros_per_box = zeros.sum(dim=3)
    N = zeros_per_box.le(3).sum(dim=2).float()
    mean = torch.div(summed_coordinates, N.unsqueeze(-1))
    return mean

def means_covs_observation(observations):
    """
    For a guide on np.einsum (vs using for loops, a lot faster)
    (which is really similar to torch.einsum, which is used below to keep gpu speed-ups)
    check:
        - (short) http://ajcr.net/Basic-guide-to-einsum/
        - or (eleborate, but VERY good) https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        - or (example also involving a covariance calculation) https://medium.com/the-quarks/an-einsum-use-case-8dafcb933c66

    args:
        observations: (tensor) combined bounding boxes, only spatial information
            one bounding box shnumber ofould have the coordinates like this:
            [x0,y0,x1,y1], the coordinates of the upper left and lower right corners
            respectively. As each observation can have a variable number of bounding boxes,
            the observations that have less than the maximum number of bounding are assumed to be padded
            with zeros.

            shape: [batch, observations, max(n_boxes_of_all_obs) ,4]

    :return:
        means_covs_observation: last dim is (mu0,mu1,cov0,cov1)
            shape: [batch, observation, 4]
    """
    max_boxes = observations.shape[2]
    num_observations = observations.shape[1]
    num_batches = observations.shape[0]

    # per bounding box, sum each individual coordinate
    summed_coordinates = observations.sum(dim=2)
    zeros = observations.le(0.)
    zeros_per_box = zeros.sum(dim=3)
    N = zeros_per_box.le(3).sum(dim=2).float()
    mean = torch.div(summed_coordinates, N.unsqueeze(-1))
    # mean = torch.div(summed_coordinates, torch.transpose(N, 0, 1))
    #### covariances
    # must be done seperately for upperleft corner (0) and lower right corner (1) of bounding box
    mean_0 = mean[:, :, 0:2]
    mean_1 = mean[:, :, 2:4]
    observations_0 = observations[:, :, :, 0:2]
    observations_1 = observations[:, :, :, 2:4]

    # Batch Observation boXes coordinatesTransposed and Batch Observation boXes Coordinates
    cov_first_part_summed_0 = torch.einsum('boxt,boxc -> botc', observations_0, observations_0)
    cov_first_part_summed_1 = torch.einsum('boxt,boxc -> botc', observations_1, observations_1)

    # double unsqueeze to allow for batches
    stacked_N = N.unsqueeze(-1).unsqueeze(-1)

    cov_first_part_0 = torch.div(cov_first_part_summed_0, stacked_N)
    cov_first_part_1 = torch.div(cov_first_part_summed_1, stacked_N)

    cov_second_part_0 = torch.einsum('bik,bij-> bijk',mean_0, mean_0)
    cov_second_part_1 = torch.einsum('bik,bij-> bijk',mean_1, mean_1)

    cov_0 = cov_first_part_0 - cov_second_part_0
    cov_1 = cov_first_part_1 - cov_second_part_1


    return mean ,cov_0, cov_1


def means_observations(observations):
    """
    For a guide on np.einsum (vs using for loops, a lot faster)
    (which is really similar to torch.einsum, which is used below to keep gpu speed-ups)
    check:
        - (short) http://ajcr.net/Basic-guide-to-einsum/
        - or (eleborate, but VERY good) https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        - or (example also involving a covariance calculation) https://medium.com/the-quarks/an-einsum-use-case-8dafcb933c66

    args:
        observations: (tensor) combined bounding boxes, only spatial information
            one bounding box should have the coordinates like this:
            [x0,y0,x1,y1], the coordinates of the upper left and lower right corners
            respectively. As each observation can have a variable number of bounding boxes,
            the observations that have less than the maximum number of bounding are assumed to be padded
            with zeros.

            shape: [max(n_boxes), batch, observations,4]

    :return:
        means_observation: last dim is (mu0,mu1)
            shape: [batch, observation, 2]
    """

    # per bounding box, sum each individual coordinate
    summed_coordinates = observations.sum(dim=2)
    zeros = observations.le(0.)
    zeros_per_box = zeros.sum(dim=0)
    N = zeros_per_box.le(3).sum(dim=2).float()
    mean = torch.div(summed_coordinates, torch.transpose(N, 0, 1))

    return mean
