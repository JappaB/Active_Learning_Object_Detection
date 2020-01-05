import os
import argparse

import torch



from data import VOC_ROOT_LOCAL, VOC_CLASSES as labelmap
from data import config

from active_learning_package.simulated_active_learning import sim_active_learning

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# parser arguments from train.py
parser = argparse.ArgumentParser(description='Active Learning With Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--dataset_root', default=VOC_ROOT_LOCAL,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading') # probably faster with 4 (on a 4 gpu machine), but only worked with 0 (locally)
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dataset', default='VOC07', choices=['VOC07', 'VOC12','VOC07_1_class','VOC07_1_class','VOC07_6_class'],
                    type=str, help='VOC07_1_class is with one class of interest and the background class')
parser.add_argument('--optimizer', default='ADAM',choices=['SGD', 'ADAM'],
                    help='Optimizer to use, check which are supported in this code first') # todo, check if this is used
parser.add_argument('--active_learning_dir', default='active_learning_dir/',
                    help='The dir which the active learning system should use for any I/O operation, '
                         'including the samples that should be selected and potentially the uncertainty '
                         'over multiple stochastic forward passes')
parser.add_argument('--simulated_al', default=True, type=str2bool,
                    help='Wether the active learning is simulated (on a datasets with labels actually already available)'
                         'or not.')
parser.add_argument('--labeled_idx_file', default='labels', type=str,
                    help='The name of the file containing the idx of the pool of labeled images')
parser.add_argument('--train_on_all_labeled_data', default=True, type=str2bool,
                    help="In each Active Learning iteration, you can either learn on all the labeled data or just on the new data")
parser.add_argument('--eval_every_iter', default = False, type=str2bool,
                    help= 'evaluate mAPs every al iteration')
parser.add_argument('--short_gpu', default = True, type=str2bool,
                    help = 'On the short GPU nodes of the Lisa cluster the maximum runtime is 60 min. Flag ensures to save weightsin time.' ) #critical
parser.add_argument('--debug', default=True, type=str2bool,
                    help='A flag that makes sure just a limited number of samples are used'
                         'in test, train and eval')
parser.add_argument('--resume_al', default=False, type=str2bool,
                    help='Wether the active learning system should resume from a certain point or start from scratch.'
                         'DONT FORGET: if true loads the last weights of the model trained in this active learning experiment, '
                         'change the --resume flag, if a seed net is used, always set resume_al to true')
parser.add_argument('--train_from_basenet_every_iter', default=True, type= str2bool,
                    help='If true, the system will be trained with only the weights of the'
                         'specified base (e.g. vgg16) for every AL-iteration')
parser.add_argument('--seed', default = 42, type=int,
                    help='seed for pseudo random number generators')
parser.add_argument('--save_all_uncertainties', default = False , type= str2bool,
                    help= 'Save the class uncertainties for all bounding boxes for all images in the summary. Becomes a large summary file, AND'
                          'also takes a lot of RAM but helps to verify whether the values of the uncertainty metric makes sense (compare trained vs untrained model)' )
parser.add_argument('--save_all_bounding_boxes', default = False, type=str2bool,
                    help='Save all bounding boxes')
parser.add_argument('--modeltype', default='SSD300',choices=['SSD300', 'SSD300KL'],
                    help='Which model to use: standard SSD or the SSD with  uncertainty in the bounding box regression and KL loss.'
                         'Note that SSD300KL does not work well!') #SSD300KL doesn't work well
parser.add_argument('--sampling_strategy', default='no_ensemble_entropy-only',
                    choices=['random_none', 'none_covariance', 'none_covariance-obj',
                             'entropy_none', 'entropy_covariance', 'entropy_covariance-obj',
                             'var-ratio_none', 'var-ratio_covariance', 'var-ratio_covariance-obj',
                             'p-max_localization-stability','no_ensemble_entropy-only'],
                    help='The sampling strategy the active learning system should use in the form of '
                         '[classification]_[localization]. If either one is not used then fill in [none]')
parser.add_argument('--merging_method', default=None,
                    choices=['pre_nms_avg','bsas','hbdscan'],
                    help = 'strategy used to merge bounding boxes from different ensembles')
parser.add_argument('--sample_select_nms_conf_thresh', default = 0.01, type = float,
                    help = 'The conf threshold used in before non maximum suppression. Only detections with a confidence above '
                           'this threshold for a certain class will go trough nms')
parser.add_argument('--delete_top_n_percent_loc_uncertainty_hbdscan', default = None, type = float,
                    help = 'as hbdscan can merge bounding boxes that are very far apart, resulting in a high localization uncertainty,'
                           'this parameter allows to delete those during the image uncertainty calculation')
parser.add_argument('--experiment_dir',default='debug2/', type=str,
                    help = 'directory in which to save experimental results. MUST end with a /')
parser.add_argument('--annotate_all_objects', default=True, type=str2bool,
                    help='Whether to annotate all objects in the image, or only the object that scores the highest '
                         'on e.g. uncertainty')
parser.add_argument('--validation_loss_frequency', default = 1, type=int,
                    help='how often the validation loss will be calculated in simulated active learning for early stopping')
parser.add_argument('--samples_per_iter', default='',type=str, nargs='+',
                     help='How many of the samples should be selected to be annotated per active learning iteration (input a list)')
# parser.add_argument('--early_stopping_condition', default = None,
#                     choices=['val_loss', 'val_loc_loss', 'val_conf_loss','val_loss2', 'val_loc_loss2', 'val_conf_loss2','just_lowest_val'],
                    # help= 'Early stopping condition. The 2 in the options means that the validation has to be lower 2 times in a row.')

parser.add_argument('--train_epochs', default=10, type=int,
                    help= 'Number of epochs')
parser.add_argument('--fixed_number_of_epochs', default=None, type=int,
                    help= 'Fill a fixed number of train iterations to use that instead of early stopping based on a validation set')
parser.add_argument('--fixed_number_of_train_iterations', default=None, type=int,
                    help= 'Train for a fixed number of iterations, like in the localization stability paper')
parser.add_argument('--paths_to_weights', default='',type=str, nargs='+',
                    help='These are the weights that ere used the initial evaluation of the unlabeled dataset') # if no trained model is given, this will return an error when loading the model.
parser.add_argument('--trained_models', default='',type=str, nargs='+',
                    help='TODO')#TODO) # if no trained model is given, this will return an error when loading the model.
parser.add_argument('--ensemble_size', default=5, type=int,
                    help='the number of models in the ensemble, must correspond to the number of models given at paths_to_weights and trained_models')
parser.add_argument('--imageset_train', default='sheep_trainval_detect', type=str, nargs='+',
                    help = 'VOC imageset that should be used for sample selection and training')
parser.add_argument('--subset_train', default=None, type=int,
                    help = 'Should be integer between 0 and the total length of the smallest train set.'
                           'During development, smaller test set for faster eval.'
                           'The first x idx of the test_set imageids will be used per imageset (in case there multiple imagesets are combined)')
parser.add_argument('--imageset_test', default='sheep_test_detect', type=str, nargs='+',
                    help = 'VOC imageset that should be used for evaluating performance')
parser.add_argument('--subset_test', default=None, type=int,
                    help = 'Should be integer between 0 and the total length of the smallest imageset_test.'
                           'During development, smaller test set for faster eval.'
                           'The first x idx of the test_set imageids will be used per imageset (in case there multiple imagesets are combined)')
parser.add_argument('--seed_set_file',default ='data/sheep_seed_set.json',type=str)
parser.add_argument('--scratch_tmpdir', default = None, type=str,
                    help='if run on LISA, data files can be transfered to scrath file system'
                         'if not None, this argument gives the location of the $TMPDIR')
parser.add_argument('--skip_sample_selection_first_iter', default = False, type=str2bool,
                    help = 'Useful when an experiment didnt run long enough to finish')
parser.add_argument('--skip_detection_part_sample_selection', default=False, type=str2bool,
                    help='Useful when an experiment crashed after a (long) detection part of the sample selection,'
                         'e.g. when it crashes in the calculation of the uncertainties')
parser.add_argument('--start_first_iter_from_ensemble_id', default=0, type=int,
                    help = 'Useful when an experiment didnt run long enough to finish')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Useful when an experiment didnt run long enough to finish')
parser.add_argument('--save_every_epoch', default=False, type=str2bool,
                    help='Useful when an experiment didnt run long enough to finish')
parser.add_argument('--rescaled_foreground_probs', default = True,
                    help = 'in the classification uncertainty, wether to rescale the probabilities, excluding the bacground prob')
parser.add_argument('--eval_n_per_al_iter', default = None)
parser.add_argument('--save_dir_eval', default = 'eval/')
parser.add_argument('--train_basenets', default= False)
parser.add_argument('--user_relevance_feedback', default=False,
                    help ='by default, the most uncertain images are selected. if user_relevance_feedback'
                          'is used, the least uncertain images are selected. ')
parser.add_argument('--budget_measured_in_objects', default = True, type = str2bool,
                    help = 'Budget either measured in number of images sampled or in number of objects sampled')
parser.add_argument('--loc_loss_weight',default=1.0,type=float,
                    help='To alter the weight between loc_loss weight and classification weight. Normally 1, but with '
                         'KL-Loss, might need adjustment')
parser.add_argument('--epochs_no_improvement', default=2, type=int,
                    help='To find a learning rate scheme, we do experiments with different schemes')
parser.add_argument('--save_variances_sample_selection', default = True, type = str2bool,
                    help='in SSD300 with KL Loss you can choose to save the variances for later evaluation or not')
parser.add_argument('--relevant_class', default=None, type = str,
                    choices= ['car','bottle','pottedplant','horse','sheep','boat'],
                    help='The class that is the relevant class when only one object class is used')
parser.add_argument('--density_diversity', default = None, type = str,
                    choices = ['density','density_diversity'])
parser.add_argument('--similarity_dir', default='data/image_similaritiesVOC07/')
parser.add_argument('--train_on_full_dataset', default=False, type=str2bool,
                    help='only do training with full dataset')
parser.add_argument('--no_foreground_multiplication', default=None, type=str2bool,
                    help= 'For ablation study -- not multiplying the uncertainty of the observations by the foreground probability')

def active_learning(args):

    if not args.simulated_al: # Active learning, with human oracles
        raise NotImplementedError()
        human_active_learning()
    else: # Simulated active learning -> research purposes
        sim_active_learning(args)

if __name__ == '__main__':

    args = parser.parse_args()

    if not args.annotate_all_objects:
        raise NotImplementedError
    # making sure the file system works
    if args.experiment_dir[-1] != '/':
        args.experiment_dir = args.experiment_dir +'/'

    args.experiment_dir = str(args.active_learning_dir + args.experiment_dir)


    if args.dataset in ['VOC12','VOC07']:
        args.cfg = config.voc # adapted from pytorch SSD code

    elif args.dataset == 'VOC07_1_class':
        args.cfg = config.voc_1_class

    elif args.dataset == 'VOC07_6_class':
        args.cfg = config.voc_6_class

    if not os.path.exists(args.active_learning_dir):
        os.mkdir(args.active_learning_dir)

    if not os.path.exists(args.experiment_dir):
        os.mkdir(args.experiment_dir)

    if not os.path.exists(args.experiment_dir+'weights/'):
        os.mkdir(args.experiment_dir+'weights/')
    if not os.path.exists(args.experiment_dir+'sample_selection/'):
        os.mkdir(args.experiment_dir+'sample_selection/')

    if args.cuda and torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'
        torch.set_default_tensor_type('torch.FloatTensor')

    args.device = device

    if args.dataset not in ["VOC07","VOC07_1_class",'VOC07_6_class']:
        raise NotImplementedError()

    args.dataset_root = os.path.join(config.HOME, args.dataset_root)
    args.save_dir_eval = str(args.experiment_dir + args.save_dir_eval)

    args.path_to_labeled_idx_file = args.experiment_dir+args.labeled_idx_file + '.json'

    # convert the samples per iter from str to ints
    args.samples_per_iter = [int(num_samples) for num_samples in args.samples_per_iter]

    # must be changed to list in that case
    if args.ensemble_size > 1 and isinstance(args.paths_to_weights, str):
        args.paths_to_weights = []


    # If only a single class is used
    if args.relevant_class == 'car':
        args.object_class_number = 6
    elif args.relevant_class == 'horse':
        args.object_class_number = 12
    elif args.relevant_class == 'sheep':
        args.object_class_number = 16
    elif args.relevant_class == 'boat':
        args.object_class_number = 3
    elif args.relevant_class == 'bottle':
        args.object_class_number = 4
    elif args.relevant_class == 'pottedplant':
        args.object_class_number = 15


    if args.relevant_class:
        # used to set all classes to background except the single relevant class
        args.class_to_ind = {'aeroplane': -1,
                            'bicycle': -1,
                            'bird': -1,
                            'boat': -1,
                            'bottle': -1,
                            'bus': -1,
                            'car': -1,
                            'cat': -1,
                            'chair': -1,
                            'cow': -1,
                            'diningtable': -1,
                            'dog': -1,
                            'horse': -1,
                            'motorbike': -1,
                            'person': -1,
                            'pottedplant': -1,
                            'sheep': -1,
                            'sofa': -1,
                            'train': -1,
                            'tvmonitor': -1}

        args.class_to_ind[args.relevant_class] = 0


    if args.dataset =='VOC07_6_class':
        args.class_to_ind = {'aeroplane': -1,
                            'bicycle': -1,
                            'bird': -1,
                            'boat': 0,
                            'bottle': 1,
                            'bus': -1,
                            'car': 2,
                            'cat': -1,
                            'chair': -1,
                            'cow': -1,
                            'diningtable': -1,
                            'dog': -1,
                            'horse': 3,
                            'motorbike': -1,
                            'person': -1,
                            'pottedplant': 4,
                            'sheep': 5,
                            'sofa': -1,
                            'train': -1,
                            'tvmonitor': -1}

        # original class numbers in dataset
        args.object_class_numbers = {}
        args.object_class_numbers['car'] = 6
        args.object_class_numbers['horse'] = 12
        args.object_class_numbers['sheep'] = 16
        args.object_class_numbers['pottedplant'] = 3
        args.object_class_numbers['bottle'] = 4
        args.object_class_numbers['boat'] = 15
        args.labelmap = ['boat','bottle','car','horse','pottedplant','sheep']

    # needed for paths to imagesets of VOC
    if type(args.imageset_train) == list and args.dataset in ["VOC07","VOC07_1_class",'VOC07_6_class']:
        imageset_train = []
        for i in args.imageset_train:
            imageset_train.append(('2007',i))
        args.imageset_train = imageset_train
    elif type(args.imageset_train) == str:
        args.imageset_train = [('2007',args.imageset_train)]
    else:
        raise NotImplementedError()

    if type(args.imageset_test) == list and args.dataset in ["VOC07","VOC07_1_class",'VOC07_6_class']:
        imageset_test = []
        for i in args.imageset_test:
            imageset_test.append(('2007',i))
        args.imageset_test = imageset_test
    elif type(args.imageset_test) == str:
        args.imageset_test = [('2007',args.imageset_test)]
    else:
        raise NotImplementedError()



active_learning(args)
