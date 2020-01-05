"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import shutil
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


# # 'aeroplane', 'bus', 'car', 'motorbike', 'person'   #Easy classes
# # 'pottedplant', 'bottle', 'cow', 'chair', 'bird',  #Hard classes
# VOC_SUBSET_CLASSES = (
#     'aeroplane', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
#     'motorbike', 'person','pottedplant'
# )

# note: if you used our download scripts, this should be right
VOC_ROOT_LOCAL = osp.join(HOME, "data/VOCdevkit/")
VOC_ROOT = "data/VOCdevkit/"





"""
@Maarten: for a custom dataset, this looks promising; https://github.com/amdegroot/ssd.pytorch/issues/72
"""

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        if class_to_ind != None:
            self.skip_non_relevant_classes = True
        else:
            self.skip_non_relevant_classes = False
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            # if class_to_ind
            name = obj.find('name').text.lower().strip()
            # if class_to_ind is passed, we want to skip classes that are set to background (-1)
            if self.skip_non_relevant_classes:
                if self.class_to_ind[name] == -1:
                    continue

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



class VOCAnnotationTransform2(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        if class_to_ind != None:
            self.skip_non_relevant_classes = True
        else:
            self.skip_non_relevant_classes = False
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            # if class_to_ind
            name = obj.find('name').text.lower().strip()
            # if class_to_ind is passed, we want to skip classes that are set to background (-1)
            if self.skip_non_relevant_classes:
                if self.class_to_ind[name] == -1:
                    continue

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
                if difficult:
                    bndbox.append(1)
                else:
                    bndbox.append(0)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]


        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'),
                             ('2007', 'train'),
                             ('2007', 'val'),
                             ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(keep_difficult=True),
                 dataset_name='VOC0712', idx = None, num_classes = 20, object_class_number = None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.num_classes = num_classes
        self.object_class_number = object_class_number

        if not idx:
            self.ids = list()

            for (year, name) in image_sets:
                rootpath = osp.join(self.root, 'VOC' + year)
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((rootpath, line.strip()))
        else:
            self.ids=list()
            for year, name in image_sets:
                rootpath = osp.join(self.root,'VOC' + year)
                for id in idx:
                    self.ids.append((rootpath, id))

        self.size = len(self.ids)

    def __getitem__(self, index):
        # if self.num_classes == 20:
        im, gt, h, w = self.pull_item(index)
        # elif self.num_classes == 1:
        #     im, gt, h, w = self.pull_item(index)
        #     if gt[4] != self.object_class_number:
        #         gt[4] = 0 # turn other classes into background classes if we only use one
        #     else:
        #         gt[4] = 1 # turn class of interest into foreground class (always 1)

        # else:
        #     raise NotImplementedError()
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None and target != []: # target is an empty list if it contains no relevant class
            target = np.array(target)
            try:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            except IndexError:
                print()
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_image_using_imageset_id(self,imageset_id):
        return cv2.imread(self._imgpath % imageset_id, cv2.IMREAD_COLOR)



    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_anno_using_imageset_id(self,img_id):
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def VOC_file_classification_to_detection_file(input_file_name):
    # Open
    in_path = VOC_ROOT_LOCAL+'/VOC2007/ImageSets/Main/'+input_file_name+'.txt'
    # Read file

    with open(in_path,'r') as f:
        data = f.readlines()

    # strip \n and ' '
    stripped_data = [l[:6] for l in data if l[-3:-1] != '-1']
    stripped_data = [l+'\n' for l in stripped_data]
    # save file
    out_path = VOC_ROOT_LOCAL+'/VOC2007/ImageSets/Main/'+input_file_name+'_detect.txt'
    with open(out_path,'w') as f:
        f.writelines(stripped_data)


def coco_to_voc_weights(source_path, destination_path):
    """
    # inspired by: https://github.com/pierluigiferrari/ssd_keras/blob/master/weight_sampling_tutorial.ipynb

    """
    # 0 -> 0 (background)
    # 5 -> 1 (airplane)
    # 2 -> 2 (bicycle)
    # 15 -> 3 (bird)
    # 9 -> 4 (boat)
    # 40 -> 5 (bottle)
    # 6-> 6(bus)
    # 3-> 7(car)
    # 16-> 8(cat)
    # 57-> 9(chair)
    # 20-> 10 (cow)
    # 61-> 11 (dining table)
    # 17-> 12 (dog)
    # 18-> 13 (horse)
    # 4-> 14 (motorbike)
    # 1-> 15(person)
    # 59 -> 16 (pottedplant)
    # 19-> 17 (sheep)
    # 58-> 18 (couch->sofa)
    # 7-> 19 (train)
    # 63-> 20 (tvmomitor)


    classes_of_interest = [0, 5, 2, 15, 9, 40, 6, 3, 16, 57, 20, 61, 17, 18, 4, 1, 59, 19, 58, 7, 63]
    # classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

    # torch.load_state_dict('../active_learning_dir/debug/weights/SSD300_train-loss_7.00734196465446__val-loss_7.189980634894848_COCO_train-iter_3000_trained_COCO.pth')

    n_classes_source = 81

    classifier_names = ['conf.0',
                        'conf.1',
                        'conf.2',
                        'conf.3',
                        'conf.4',
                        'conf.5']

    if not osp.isfile(destination_path):
        # load weights
        trained_weights = torch.load(source_path, map_location='cpu')
        # Make a copy of the weights file.
        shutil.copy(source_path, destination_path)

    else:
        # load weights
        trained_weights = torch.load(source_path, map_location='cpu')

    weights_destination_file = torch.load(destination_path, map_location='cpu')

    for name in classifier_names:
        # get the trained weights for this layer
        kernel = trained_weights[name + '.weight']
        bias = trained_weights[name + '.bias']

        # get the shape of the kernel.
        # height, width, in_channels, out_channels = kernel.shape #3 3 512 324

        out_channels, in_channels, height, width = kernel.shape
        # print(kernel.shape)

        # Compute the indices of the elements we want to sub-sample.
        # Keep in mind that each classification predictor layer predicts multiple
        # bounding boxes for every spatial location, so we want to sub-sample
        # the relevant classes for each of these boxes.
        if isinstance(classes_of_interest, (list, tuple)):
            subsampling_indices = []
            for i in range(int(out_channels / n_classes_source)):
                indices = np.array(classes_of_interest) + i * n_classes_source
                subsampling_indices.append(indices)
            subsampling_indices = list(np.concatenate(subsampling_indices))

        elif isinstance(classes_of_interest, int):
            subsampling_indices = int(classes_of_interest * (out_channels / n_classes_source))
        else:
            raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")

        # Sub-sample the kernel and bias.
        new_kernel, new_bias = sample_tensors(weights_list=[kernel.numpy(), bias.numpy()],
                                              sampling_instructions= [subsampling_indices,in_channels, height, width],
                                              axes=[[0]],
                                              init=['gaussian', 'zeros'],
                                              mean=0.0,
                                              stddev=0.005)

        # Delete the old weights from the destination file.
        del weights_destination_file[name+'.weight']
        del weights_destination_file[name+'.bias']
        # Create new datasets for the sub-sampled weights.
        weights_destination_file[name+'.weight'] = torch.FloatTensor(new_kernel)
        weights_destination_file[name+'.bias'] = torch.FloatTensor(new_bias)
    # save state-dict with voc output nodes
    torch.save(weights_destination_file, destination_path)


def sample_tensors(weights_list, sampling_instructions, axes=None, init=None, mean=0.0, stddev=0.005):
    '''
    Adjusted from: https://github.com/pierluigiferrari/ssd_keras/blob/master/misc_utils/tensor_sampling_utils.py
    
    Can sub-sample and/or up-sample individual dimensions of the tensors in the given list
    of input tensors.
    It is possible to sub-sample some dimensions and up-sample other dimensions at the same time.
    The tensors in the list will be sampled consistently, i.e. for any given dimension that
    corresponds among all tensors in the list, the same elements will be picked for every tensor
    along that dimension.
    For dimensions that are being sub-sampled, you can either provide a list of the indices
    that should be picked, or you can provide the number of elements to be sub-sampled, in which
    case the elements will be chosen at random.
    For dimensions that are being up-sampled, "filler" elements will be insterted at random
    positions along the respective dimension. These filler elements will be initialized either
    with zero or from a normal distribution with selectable mean and standard deviation.
    Arguments:
        weights_list (list): A list of Numpy arrays. Each array represents one of the tensors
            to be sampled. The tensor with the greatest number of dimensions must be the first
            element in the list. For example, in the case of the weights of a 2D convolutional
            layer, the kernel must be the first element in the list and the bias the second,
            not the other way around. For all tensors in the list after the first tensor, the
            lengths of each of their axes must identical to the length of some axis of the
            first tensor.
        sampling_instructions (list): A list that contains the sampling instructions for each
            dimension of the first tensor. If the first tensor has `n` dimensions, then this
            must be a list of length `n`. That means, sampling instructions for every dimension
            of the first tensor must still be given even if not all dimensions should be changed.
            The elements of this list can be either lists of integers or integers. If the sampling
            instruction for a given dimension is a list of integers, then these integers represent
            the indices of the elements of that dimension that will be sub-sampled. If the sampling
            instruction for a given dimension is an integer, then that number of elements will be
            sampled along said dimension. If the integer is greater than the number of elements
            of the input tensors in that dimension, that dimension will be up-sampled. If the integer
            is smaller than the number of elements of the input tensors in that dimension, that
            dimension will be sub-sampled. If the integer is equal to the number of elements
            of the input tensors in that dimension, that dimension will remain the same.
        axes (list, optional): Only relevant if `weights_list` contains more than one tensor.
            This list contains a list for each additional tensor in `weights_list` beyond the first.
            Each of these lists contains integers that determine to which axes of the first tensor
            the axes of the respective tensor correspond. For example, let the first tensor be a
            4D tensor and the second tensor in the list be a 2D tensor. If the first element of
            `axis` is the list `[2,3]`, then that means that the two axes of the second tensor
            correspond to the last two axes of the first tensor, in the same order. The point of
            this list is for the program to know, if a given dimension of the first tensor is to
            be sub- or up-sampled, which dimensions of the other tensors in the list must be
            sub- or up-sampled accordingly.
        init (list, optional): Only relevant for up-sampling. Must be `None` or a list of strings
            that determines for each tensor in `weights_list` how the newly inserted values should
            be initialized. The possible values are 'gaussian' for initialization from a normal
            distribution with the selected mean and standard deviation (see the following two arguments),
            or 'zeros' for zero-initialization. If `None`, all initializations default to
            'gaussian'.
        mean (float, optional): Only relevant for up-sampling. The mean of the values that will
            be inserted into the tensors at random in the case of up-sampling.
        stddev (float, optional): Only relevant for up-sampling. The standard deviation of the
            values that will be inserted into the tensors at random in the case of up-sampling.
    Returns:
        A list containing the sampled tensors in the same order in which they were given.




''''''
Utilities that are useful to sub- or up-sample weights tensors.
Copyright (C) 2018 Pierluigi Ferrari
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

    first_tensor = weights_list[0]

    # if (not isinstance(sampling_instructions, (list, tuple))) or (len(sampling_instructions) != len(first_tensor.shape)):
    if (not isinstance(sampling_instructions, (list, tuple))) or (len(sampling_instructions) != first_tensor.ndim):
        raise ValueError(
            "The sampling instructions must be a list whose length is the number of dimensions of the first tensor in `weights_list`.")

    if (not init is None) and len(init) != len(weights_list):
        raise ValueError(
            "`init` must either be `None` or a list of strings that has the same length as `weights_list`.")

    up_sample = []  # Store the dimensions along which we need to up-sample.
    out_shape = []  # Store the shape of the output tensor here.
    # Store two stages of the new (sub-sampled and/or up-sampled) weights tensors in the following two lists.
    subsampled_weights_list = []  # Tensors after sub-sampling, but before up-sampling (if any).
    upsampled_weights_list = []  # Sub-sampled tensors after up-sampling (if any), i.e. final output tensors.

    # Create the slicing arrays from the sampling instructions.
    sampling_slices = []
    for i, sampling_inst in enumerate(sampling_instructions):
        if isinstance(sampling_inst, (list, tuple)):
            amax = np.amax(np.array(sampling_inst))
            if amax > first_tensor.shape[i]:
                raise ValueError(
                    "The sample instructions for dimension {} contain index {}, which is greater than the length of that dimension, which is {}.".format(
                        i, amax, first_tensor.shape[i]))
            sampling_slices.append(np.array(sampling_inst))
            out_shape.append(len(sampling_inst))
        elif isinstance(sampling_inst, int):
            out_shape.append(sampling_inst)
            if sampling_inst == first_tensor.shape[i]:
                # Nothing to sample here, we're keeping the original number of elements along this axis.
                sampling_slice = np.arange(sampling_inst)
                sampling_slices.append(sampling_slice)
            elif sampling_inst < first_tensor.shape[i]:
                # We want to SUB-sample this dimension. Randomly pick `sample_inst` many elements from it.
                sampling_slice1 = np.array([0])  # We will always sample class 0, the background class.
                # Sample the rest of the classes.
                sampling_slice2 = np.sort(
                    np.random.choice(np.arange(1, first_tensor.shape[i]), sampling_inst - 1, replace=False))
                sampling_slice = np.concatenate([sampling_slice1, sampling_slice2])
                sampling_slices.append(sampling_slice)
            else:
                # We want to UP-sample. Pick all elements from this dimension.
                sampling_slice = np.arange(first_tensor.shape[i])
                sampling_slices.append(sampling_slice)
                up_sample.append(i)
        else:
            raise ValueError(
                "Each element of the sampling instructions must be either an integer or a list/tuple of integers, but received `{}`".format(
                    type(sampling_inst)))

    # Process the first tensor.
    subsampled_first_tensor = np.copy(first_tensor[np.ix_(*sampling_slices)])
    subsampled_weights_list.append(subsampled_first_tensor)

    # Process the other tensors.
    if len(weights_list) > 1:
        for j in range(1, len(weights_list)):
            this_sampling_slices = [sampling_slices[i] for i in axes[j - 1]]  # Get the sampling slices for this tensor.
            subsampled_weights_list.append(np.copy(weights_list[j][np.ix_(*this_sampling_slices)]))

    if up_sample:
        # Take care of the dimensions that are to be up-sampled.

        out_shape = np.array(out_shape)

        # Process the first tensor.
        if init is None or init[0] == 'gaussian':
            upsampled_first_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape)
        elif init[0] == 'zeros':
            upsampled_first_tensor = np.zeros(out_shape)
        else:
            raise ValueError("Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[0]))
        # Pick the indices of the elements in `upsampled_first_tensor` that should be occupied by `subsampled_first_tensor`.
        up_sample_slices = [np.arange(k) for k in subsampled_first_tensor.shape]
        for i in up_sample:
            # Randomly select across which indices of this dimension to scatter the elements of `new_weights_tensor` in this dimension.
            up_sample_slice1 = np.array([0])
            up_sample_slice2 = np.sort(
                np.random.choice(np.arange(1, upsampled_first_tensor.shape[i]), subsampled_first_tensor.shape[i] - 1,
                                 replace=False))
            up_sample_slices[i] = np.concatenate([up_sample_slice1, up_sample_slice2])
        upsampled_first_tensor[np.ix_(*up_sample_slices)] = subsampled_first_tensor
        upsampled_weights_list.append(upsampled_first_tensor)

        # Process the other tensors
        if len(weights_list) > 1:
            for j in range(1, len(weights_list)):
                if init is None or init[j] == 'gaussian':
                    upsampled_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape[axes[j - 1]])
                elif init[j] == 'zeros':
                    upsampled_tensor = np.zeros(out_shape[axes[j - 1]])
                else:
                    raise ValueError(
                        "Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[j]))
                this_up_sample_slices = [up_sample_slices[i] for i in
                                         axes[j - 1]]  # Get the up-sampling slices for this tensor.
                upsampled_tensor[np.ix_(*this_up_sample_slices)] = subsampled_weights_list[j]
                upsampled_weights_list.append(upsampled_tensor)

        return upsampled_weights_list
    else:
        return subsampled_weights_list

