import sys
import time
import os
import pickle
import cProfile
import re
import datetime

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Sampler, DataLoader

from . import helpers
from . import voc_eval_helpers
from . import active_learning_helpers

from layers import box_utils

import data

def sim_active_learning(args):
    """
    Simulated active learning loop for experimentation purposes, labels are already known.
    """

    global unlabeled_idx
    if args.labeled_idx_file is None:
        raise FileNotFoundError("No path specified for the labeled idx file'")
        sys.exit()
    # if not args.resume_al: # if a seed net is used, always set resume_al to true
    #     args.path_to_weights = None # used for sample_selection

    if args.debug and not os.path.exists(args.experiment_dir+'sample_selection/'):
        os.mkdir(args.experiment_dir+'sample_selection/')


    # set seeds:
    helpers.set_all_seeds(args)


    sample_select_dataset = helpers.load_sample_select_set(args,
                                                           args.imageset_train)

    train_dataset = helpers.load_trainset(args,
                                          args.imageset_train)

    label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                      args.annotate_all_objects,
                                      args.dataset,
                                      args)

    # np.random.choice([str(ids[1]) for ids in train_dataset.ids], size=10) # get val set todo: delete line

    val_dataset = helpers.load_evalset(args,
                                       args.imageset_train,
                                       idx=label_dict['val_set']['image_set_idx'])

    print(args)

    args.num_classes = train_dataset.num_classes + 1 # +1 for background
    # args.sample_select_dataset = [im for im in range(sample_select_dataset.size)]
    args.sample_select_dataset_imageset_ids = [id[1] for id in sample_select_dataset.ids] #imageset ids


    args.summary = {}
    args.summary['sample_selection'] = {}
    args.summary['train_model'] = {}
    args.summary['eval_model'] = {}

    # print('args: ', args)


    for i in range(args.start_iter,len(args.samples_per_iter)):

        print("Starting Active Learning iteration: {:d}/{:d}".format(i, len(args.samples_per_iter)))
        print("Local time: ", datetime.datetime.now())

        # args.al_iter_dir = args.experiment_dir + 'al_iter_' + str(i)+'/'
        # # make sure the path for this al_iteration exists
        # if not os.path.exists(args.al_iter_dir):
        #     os.mkdir(args.al_iter_dir)

        # timers
        timers = {}
        timers['full_al_iteration'] = helpers.Timer()
        timers['full_al_iteration'].tic()

        timers['sample_selection'] = helpers.Timer()
        timers['train_model'] = helpers.Timer()
        timers['eval_model'] = helpers.Timer()

        args.al_iteration = i # used for several filenames


        # selects samples to be labeled and writes them to the labeled_idx file
        print('Selecting samples...')
        timers['sample_selection'].tic()
        if not(args.skip_sample_selection_first_iter and i == args.start_iter) and not args.train_basenets:

            # no need for uncertainty calculations if we sample random
            if args.sampling_strategy == 'random_none' and not args.density_diversity:

                # samples are selected and labels are written to the labeled pool
                new_labeled_idx = active_learning_helpers.sample_selection(args,
                                                                           sample_select_dataset,
                                                                           active_learning_iter = i,
                                                                           )

            ## ENSEMBLE METHODS ##
            elif args.sampling_strategy not in ['p-max_localization-stability', 'no_ensemble_entropy-only','random_none']\
                    and args.modeltype != 'SSD300KL':
                ## sample selection

                # Load image_idx of unlabeled pool as a set
                label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                                  args.annotate_all_objects,
                                                  args.dataset,
                                                  args)

                # images to be labeled
                unlabeled_idx = [s for s in args.sample_select_dataset_imageset_ids if
                                (s not in label_dict['train_set'] and s not in label_dict['val_set']['image_set_idx'])]
                print('Number of unlabeled images:', len(unlabeled_idx))




                len_unlabeled_idx = len(unlabeled_idx)
                if not (args.skip_detection_part_sample_selection and args.start_iter == i):  # useful when the detection part went well, but an error occured in the uncertainty calculation

                    if len_unlabeled_idx > 2000:
                        num_splits = 20
                    else:
                        num_splits = 10

                    split_indices = np.ceil(np.linspace(0,len_unlabeled_idx,num_splits+1, dtype=int))
                    split_indices = [int(i) for i in split_indices]

                    # split the unlabeled imageset to cope with memory issues
                    for split_num in range(1,num_splits+1):
                        if args.debug and split_num > 1:
                            continue
                        print('split num: ', split_num)
                        print(split_indices[split_num - 1],split_indices[split_num])
                        print("Local time: ", datetime.datetime.now())
                        print('\n\n\n')

                        unlabeled_imageset_split = unlabeled_idx[split_indices[split_num - 1]:split_indices[split_num]]
                        len_unlabeled_idx = len(unlabeled_imageset_split)

                        if args.debug:
                            if split_num == 0:
                                unlabeled_imageset_split = unlabeled_imageset_split[:4]
                            else:
                                unlabeled_imageset_split = unlabeled_imageset_split[:5]

                            len_unlabeled_idx = len(unlabeled_imageset_split)
                            unlabeled_idx = unlabeled_imageset_split # todo only if i skip over the other imgsetsplits during debug

                        for j in range(args.ensemble_size):
                            print('Ensemble: ', j)
                            print("Local time: ", datetime.datetime.now())
                            if args.device == 'cuda':
                                torch.cuda.empty_cache()
                            # load one of the models
                            net = helpers.build_sample_selection_net(args,
                                                                     ensemble_idx = j,
                                                                     merging_method = args.merging_method,
                                                                     sampling_strategy = args.sampling_strategy,
                                                                     default_forward=False) # here, we want to merge the predictions of all networks in the ensemble -> can't use default forward output for that


                            # do the detection
                            output, num_unlabeled_images, priors, unlabeled_imgset = active_learning_helpers.detect_on_unlabeled_imgs(net,
                                                                                                                                      args,
                                                                                                                                      sample_select_dataset,
                                                                                                                                      unlabeled_idx=unlabeled_imageset_split,
                                                                                                                                      len_unlabeled_idx=len_unlabeled_idx
                                                                                                                                      )
                            # save the detections
                            detections_path = helpers.save_detections(args,
                                                    output,
                                                    j,
                                                    num_unlabeled_images)


                        print('\nStarting clustering observations...')
                        print("Local time: ", datetime.datetime.now())
                        print('')

                        # cluster detections to observations
                        observations = active_learning_helpers.cluster_detections_to_observations(args,
                                                                                                   args.merging_method,
                                                                                                   unlabeled_imgset,
                                                                                                   num_unlabeled_images,#todo remove
                                                                                                   priors = priors) #priors

                        # delete 'all_detections', is a fairly large file, especially for pre_nms_avg
                        if args.merging_method in ['pre_nms_avg', 'bsas','hbdscan']:
                            os.remove(detections_path)

                        # save observations
                        # todo: append observations
                        all_observations, observations_path = helpers.save_observations(args,
                                                                                          i,
                                                                                          observations)
                    # # todo delete t

                else:
                    # when skipping the observation part
                    # load detections
                    print('Loading observations from memory...')
                    path = args.experiment_dir + 'sample_selection/observations-iter_' + str(i) + '_.pickle'
                    all_observations = helpers.unpickle(path)


                print('\nStarting calculating uncertainties...')
                print("Local time: ", datetime.datetime.now())
                print('')

                # calculate the uncertainties
                uncertainty_per_image, classification_uncertainty, localization_uncertainty = active_learning_helpers.calculate_uncertainties(args,
                                                                                                all_observations,
                                                                                                unlabeled_idx) #todo remove
                print('Uncertainties calculated')
                print("Local time: ", datetime.datetime.now())
                print('')

                # save the uncertainties per iter
                with open(args.experiment_dir+'sample_selection/uncertainties-iter_'+str(i)+'_.pickle', 'wb') as f:
                    pickle.dump((uncertainty_per_image, classification_uncertainty, localization_uncertainty), f)


                # select the samples and add to labeled pool
                    #todo delete
                if not args.density_diversity:
                    if args.debug:
                        args.samples_per_iter[i] = 3
                    if not args.budget_measured_in_objects:
                        if not args.user_relevance_feedback:

                            top_k_uncertain_images = uncertainty_per_image.cpu().topk(k=args.samples_per_iter[i])
                        else:
                            top_k_uncertain_images = uncertainty_per_image.cpu().topk(k=args.samples_per_iter[i], largest = False)
                        new_labeled_idx = [unlabeled_idx[idx] for idx in top_k_uncertain_images[1]]

                    else:
                        if not args.user_relevance_feedback:
                            # make list of imgset ids ordered from most uncertain to least uncertain
                            sorted_indices = uncertainty_per_image.cpu().sort(descending=True)[1].numpy()
                            ordered_uncertain_images = [unlabeled_idx[i] for i in sorted_indices]
                            new_labeled_idx = active_learning_helpers.select_samples_with_object_budget(args,
                                                                                                        ordered_uncertain_images,
                                                                                                        object_budget=args.samples_per_iter[i],
                                                                                                        dataset=sample_select_dataset)

                        else:
                            sorted_indices = uncertainty_per_image.cpu().sort(descending=False)[1].numpy()
                            ordered_certain_images = [unlabeled_idx[i] for i in sorted_indices]
                            new_labeled_idx = active_learning_helpers.select_samples_with_object_budget(args,
                                                                                                        ordered_certain_images,
                                                                                                        object_budget=args.samples_per_iter[i],
                                                                                                        dataset=sample_select_dataset)

                    # clear memory
                    del all_observations, uncertainty_per_image, classification_uncertainty, localization_uncertainty

            ## NON ENSEMBLE METHODS
            else:
                # Load image_idx of unlabeled pool
                label_dict = helpers.read_labeled(args.path_to_labeled_idx_file,
                                                  args.annotate_all_objects,
                                                  args.dataset,
                                                  args)

                # list comprehension to create a list with images to be labeled
                unlabeled_idx = [s for s in args.sample_select_dataset_imageset_ids if
                                (s not in label_dict['train_set'] and s not in label_dict['val_set']['image_set_idx'])]
                print('Number of unlabeled images:', len(unlabeled_idx))

                if args.debug:
                    unlabeled_idx = [idx[1] for idx in sample_select_dataset.ids[:10]]
                    args.samples_per_iter[i] = 3


                len_unlabeled_idx = len(unlabeled_idx)

                # load one model
                if args.sampling_strategy == 'p-max_localization-stability' and args.modeltype != 'SSD300KL':

                    net = helpers.build_sample_selection_net(args,
                                                             ensemble_idx=0,
                                                             merging_method=args.merging_method,
                                                             default_forward=False,
                                                             sampling_strategy = args.sampling_strategy)  # need full class distribution per bounding box


                    new_labeled_idx = active_learning_helpers.localization_stability_sample_selection(args,
                                                                                                            net,
                                                                                                            sample_select_dataset,
                                                                                                            unlabeled_idx,
                                                                                                            len_unlabeled_idx,
                                                                                                            i)

                elif args.sampling_strategy == 'no_ensemble_entropy-only' and args.modeltype != 'SSD300KL':
                    net = helpers.build_sample_selection_net(args,
                                                             ensemble_idx=0,
                                                             merging_method=args.merging_method,
                                                             default_forward=False,
                                                             sampling_strategy=args.sampling_strategy)  # need full class distribution per bounding box

                    new_labeled_idx,uncertainty_per_image = active_learning_helpers.entropy_only_baseline(args,
                                                                                  net,
                                                                                  sample_select_dataset,
                                                                                  unlabeled_idx,
                                                                                  len_unlabeled_idx,
                                                                                  i)


                elif args.modeltype == 'SSD300KL':
                    net = helpers.build_sample_selection_net(args,
                                                             ensemble_idx=0,
                                                             merging_method=args.merging_method,
                                                             default_forward=False,
                                                             sampling_strategy=args.sampling_strategy)  # need full class distribution per bounding box

                    new_labeled_idx = active_learning_helpers.SSDKL_sample_selection(args,
                                                                                       net,
                                                                                       sample_select_dataset,
                                                                                       unlabeled_idx,
                                                                                       len_unlabeled_idx,
                                                                                       i)

                elif args.density_diversity == 'density':
                    pass

                else:
                    raise NotImplementedError()

            if args.density_diversity == 'density':
                density_per_image = active_learning_helpers.density_sampling(args,
                                                                             sample_select_dataset,
                                                                             unlabeled_idx
                                                                             )

                # no uncertainty added, only density in this case
                if args.sampling_strategy == 'random_none':
                    print('sampling strategy: density only')
                    informativeness_per_image = density_per_image



                else:
                    # combine density and uncertainty
                    print('sampling strategy: density + uncertainty, equal weighing')
                    informativeness_per_image = density_per_image+uncertainty_per_image

                if not args.budget_measured_in_objects:
                    if not args.user_relevance_feedback:
                        top_k_uncertain_images = informativeness_per_image.topk(
                            k=args.samples_per_iter[args.al_iteration])
                    else:
                        top_k_uncertain_images = informativeness_per_image.topk(
                            k=args.samples_per_iter[args.al_iteration], largest=False)

                    new_labeled_idx = [unlabeled_idx[idx] for idx in top_k_uncertain_images[1]]

                else:
                    if not args.user_relevance_feedback:
                        # make list of imgset ids ordered from most uncertain to least uncertain
                        sorted_indices = informativeness_per_image.cpu().sort(descending=True)[1].numpy()
                        ordered_uncertain_images = [unlabeled_idx[i] for i in sorted_indices]
                        new_labeled_idx = active_learning_helpers.select_samples_with_object_budget(args,
                                                                            ordered_uncertain_images,
                                                                            object_budget=args.samples_per_iter[
                                                                                args.al_iteration],
                                                                            dataset=sample_select_dataset)

                    else:
                        sorted_indices = informativeness_per_image.cpu().sort(descending=False)[1].numpy()
                        ordered_certain_images = [unlabeled_idx[i] for i in sorted_indices]
                        new_labeled_idx = active_learning_helpers.select_samples_with_object_budget(args,
                                                                            ordered_certain_images,
                                                                            object_budget=args.samples_per_iter[
                                                                                args.al_iteration],
                                                                            dataset=sample_select_dataset)

            # write to labeled
            helpers.write_labeled(args,
                                  new_labeled_idx
                                  )

            # write to summary for this active learning iteration
            selected_classes = helpers.class_dist_in_imageset(args,
                                                              new_labeled_idx,
                                                              sample_select_dataset)

            args.summary['sample_selection']['object_classes_selected'] = selected_classes
            args.summary['sample_selection']['images_selected'] = new_labeled_idx

            # write summary of select_samples
            helpers.write_summary(args,
                                  timers,
                                  write='sample_selection')

            print("Samples are selected for Active Learning iteration: {:d}/{:d}".format(i, len(args.samples_per_iter)))
            print("Samples are written to:  {}".format(args.path_to_labeled_idx_file))



        else:
            print('skip sample selection for first al-iter')



        ## train network(s)##
        print('Starting training of Active Learning iteration: {:d}/{:d}'.format(i, len(args.samples_per_iter)))
        timers['train_model'].tic()

        # placeholder
        args.summary['train_model']['losses'] = {}
        ## Train ensemble
        for j in range(args.ensemble_size):
            print("train model {:d}/{:d}".format(j+1,args.ensemble_size))
            print("Local time: ", datetime.datetime.now())
            if i == args.start_iter and j < args.start_first_iter_from_ensemble_id:
                print('Skip training this model as it has already been trained')
                continue

            # change seeds to ensure different order in which it sees the training samples and in which the model is initialized
            helpers.set_all_seeds(args,seed_incrementer=j)

            # train model, best model of final cross val split is saved in args.path_to_weights[ensemble_id]
            active_learning_helpers.train_model(args,        ## Training
                                                train_dataset,
                                                val_dataset,
                                                ensemble_idx = j)

        timers['train_model'].toc()

        # write summary of train_model this al_iteration
        helpers.write_summary(args,
                              timers,
                              write='train_model'
                              )


        ## EVAL ##
        if args.eval_every_iter:
            print('Start eval')
            print("Local time: ", datetime.datetime.now())
            eval_dataset = helpers.load_evalset(args, imageset=args.imageset_test)

            # check if eval folder exists, if not, make it
            # todo
            # path
            path_to_weights = os.path.join(args.experiment_dir, 'weights/')

            # todo: eval_n_per_al_iter doesn't work
            if args.eval_n_per_al_iter:
                al_iters_avaluated = {}

            # network this iter
            network_list = [net for net in os.listdir(path_to_weights) if int(net.split('_')[2]) == i]
            if args.eval_n_per_al_iter:
                if i in al_iters_avaluated:
                    if al_iters_avaluated[i] >= args.eval_n_per_al_iter:
                        print('already evaluated ', str(args.eval_n_per_al_iter),
                              'networks, continuing to next network ')
                        print()
                        continue
                    else:
                        al_iters_avaluated[i] += 1
                al_iters_avaluated[i] = 1
            for weights in network_list:
                #     al_iter = net.split('_')[2]

                # load the network
                args.path_to_eval_weights = path_to_weights + weights
                eval_net = helpers.build_eval_net(args)

                timers['eval_model'].tic()
                args.eval_ensemble_idx = weights.split('_')[-2]
                args.al_iteration = i


                # run evalutation
                if args.dataset in ['VOC07','VOC07_1_class','VOC07_6_class']:
                    voc_eval_helpers.eval(eval_dataset, args, eval_net, i, args.eval_ensemble_idx)
                else:
                    raise NotImplementedError()
                timers['eval_model'].toc()

                # save evaluation results
                # write eval (eval uses args..)
                helpers.eval_summary_writer(args,
                                            timers)