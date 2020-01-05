import os
import pickle
import json

from sklearn.decomposition import PCA
from torch.autograd import Variable
import torchvision.models as models

from data import *
import active_learning_package.helpers as helpers



def get_feature_maps(dataset,
                     net,
                     imageset_name,
                     save_dir):

    path_to_image_feature_dir = os.path.join(save_dir, imageset_name + '586_conv5_3_features_before_relu/')
    # path_to_image_feature_dir = save_dir+'2012trainval586_conv5_3_features/'

    if not os.path.exists(path_to_image_feature_dir):
        os.mkdir(path_to_image_feature_dir)

    # go trough all images in imageset
    already_saved = os.listdir(path_to_image_feature_dir)

    transform = BaseTransform(586, (104, 117, 123))

    for i, idx in enumerate(dataset.ids):
        image_feature_path = path_to_image_feature_dir + str(idx[1]) + '.pickle'
        if str(idx[1]) + '.pickle' in already_saved:
            print(i, '/', len(dataset.ids), ' was already saved')

            # load feature and append it
            # features = helpers.unpickle(image_feature_path)

            # conv_feature_list.append(features)

            continue

        print(i, '/', len(dataset.ids))

        # load image and transform (colors in different order)
        img = dataset.pull_image_using_imageset_id(idx)

        # if features already saved, load them
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)  # We use pre-trained model from pytorch model zoo, which is trained with RGB, cv2.imread loads in BGR

        x = Variable(x.unsqueeze(0))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.to('cuda')

        # directly calculate sum over channels
        features = net(x)

        # # take the sum of the 512 channels as features (NOTE: 512 is specific to VGG16 conv5_3)
        features = features.reshape(1, 512, -1).sum(dim=-1)

        # set detections back to cpu
        if torch.cuda.is_available():
            features = features.to('cpu')

        # append to conv_feature list
        # conv_feature_list.append(features)

        with open(image_feature_path, 'wb') as f:
            pickle.dump(features, f)

    return
def calculate_PCA_and_whitening_parameters(dataset,
                                           imageset_name,
                                           save_dir):


    conv_feature_list = []
    if '2007' in imageset_name:
        print('This is to get PCA, should be done with 2012 dataset, this is a failsafe to not overwrite the 2012 PCA with 2007 PCA')
        raise NotImplementedError

    # path_to_image_feature_dir = os.path.join(save_dir,imageset_name+'586_conv5_3_features_before_relu/')
    path_to_image_feature_dir = save_dir+'2012trainval586_conv5_3_features/'
    #

    # load features
    pca_save_path = path_to_image_feature_dir + imageset_name +'PCA.pickle'

    if os.path.exists(pca_save_path):
        print('already did this PCA')
        return
    print('load features:')
    for i, idx in enumerate(dataset.ids):
        print('load feature', i, '/', len(dataset.ids),' and L2 normalize features before PCA')
        image_feature_path = path_to_image_feature_dir + str(idx[1]) + '.pickle'

        # load feature and append it
        features = helpers.unpickle(image_feature_path)

        # L2 normalize
        features = features / features.norm(2)


        conv_feature_list.append(features)

    np_features = torch.cat(conv_feature_list).detach().numpy()
    print('loaded all features and transformed them into a numpy array')

    ## calculate PCA parameters (which dimensions should be kept)
    # numpy array
    print('Do PCA')
    pca = PCA(n_components = 256, svd_solver = 'full', random_state = 42,whiten=True)
    pca.fit(np_features)
    print('did PCA')

    # save PCA
    pca_save_path = path_to_image_feature_dir + imageset_name +'PCA.pickle'

    with open(pca_save_path, 'wb') as f:
        pickle.dump(pca, f)

    print('Saved PCA')

    return

def create_spoc_features(dataset,
                         image_features_path,
                         PCA_param_path,
                         imageset_name,
                         save_dir):
    """
    See Babenko 2014

    """

    # load pca and whitening parameters
    pca = helpers.unpickle(PCA_param_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load image features
    for i, idx in enumerate(dataset.ids):
        image_feature_path = image_features_path + str(idx[1]) + '.pickle'

        # load feature and append it
        features = helpers.unpickle(image_feature_path)

        print(i, '/', len(dataset.ids))

        # l2 normalization
        features = features/features.norm(2)

        # apply pca transform + whitening to features
        features = pca.transform(features.detach().numpy())
        features = torch.tensor(features)

        # l2-normalization
        features = features/features.norm(2)
        spoc_feature_path = save_dir + str(idx[1]) + '.pickle'

        # save SPoC representation

        with open(spoc_feature_path,'wb') as f:
            pickle.dump(features, f)

    print('Created and Saved all SpoC representations of images')

    return


def calculate_scalar_product_image_similarity(tensor_a,tensor_b):
    """
    https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics

    calculates image similarity between two images using a simple scalar product matching kernel
    L. Bo and C. Sminchisescu. Efficient match kernel between
    sets of features for visual recognition. In Advances in Neural Information Processing Systems (NIPS)., pages 135–143,
    2009.

    :return: similarity
    """

    return torch.dot(tensor_a.squeeze(),tensor_b.squeeze())


def calculate_all_images_similarities(dataset, load_dir_spoc_features):
    """

    :return:
    """

    # todo: can be made faster, now doing redundant calculations (similarities of a->b and b->a)

    save_dir = load_dir_spoc_features + 'image_similarities/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    already_saved = os.listdir(save_dir)

    # go trough dataset
    for i,idx in enumerate(dataset.ids):
        if str(idx[1]) + '.pickle' in already_saved:
            print(i, '/', len(dataset.ids), ' was already saved')

        print(i,'/',len(dataset.ids))
        # placeholder to store similarities between all images
        image_similarity_dir = {}

        # load image description
        image_path_a = load_dir_spoc_features+ str(idx[1]) + '.pickle'
        image_a = helpers.unpickle(image_path_a)


        # go trough all OTHER images except the idx whe are currently at
        other_images = [idj for idj in dataset.ids if idj != idx]

        for j, idj in enumerate(other_images):

            # load image description
            image_path_b = load_dir_spoc_features + str(idj[1]) + '.pickle'
            image_b = helpers.unpickle(image_path_b)


            # calculate similarity
            similarity = calculate_scalar_product_image_similarity(image_a,image_b)

            if similarity.shape == torch.Size([0]):
                print(similarity)
                print('similarity should bed a scalar')
                raise NotImplementedError

            # store similarity
            image_similarity_dir[idj[1]] = similarity.item()

        # save image similarity dir
        path = save_dir + str(idx[1]) + '.pickle'

        with open(path,'wb') as f:
            pickle.dump(image_similarity_dir, f)

    return save_dir


def calculate_density_per_imageset(dataset,load_dir_similarities):
    """
    density is the mean similarity of one image to all other images in the dataset (see Settles 2008)
    """

    # todo: can be made faster, now doing redundant calculations (similarities of a->b and b->a)
    # go trough dataset
    density = {}
    for i,idx in enumerate(dataset.ids):
        print(i,'/',len(dataset.ids))
        # load similarity between all images in trainval and current image (idx)
        path = load_dir_similarities + str(idx[1]) + '.pickle'

        similarities_idx = helpers.unpickle(path)

        # go trough all OTHER images in the dataset (can be a subset of trainval, e.g. only the car images)
        # except the id where are currently
        other_images = [idj for idj in dataset.ids if idj != idx]

        # placeholder
        density[idx[1]] = 0
        for i, idj in enumerate(other_images):

            density[idx[1]] += similarities_idx[idj[1]]

        # divide by number of images to get mean
        density[idx[1]] /= len(other_images)



    # save image density dir
    path = load_dir_similarities + dataset.image_set[0][1] + '.pickle'



    with open(path,'wb') as f:
        pickle.dump(density, f)




# def create_image_affinity_propagation_clusters(features,
#                                                dataset,
#                                                imageset_name):
#
#     return

if __name__ == '__main__':

    save_dir = 'data/'

    """get feature maps"""
    # imagesets = [[('2012', 'trainval')],
    #              [('2007', 'trainval')]
    #              [('2012', 'bottle_trainval_detect')],
    #               [('2012', 'car_trainval_detect')],
    #               [('2012', 'horse_trainval_detect')],
    #               [('2012', 'sheep_trainval_detect')],
    #               [('2012', 'pottedplant_trainval_detect')]
    #               ]


    # load network
    # vgg16 = models.vgg16(pretrained=True) #NOTE:  I adjusted the source code of the vgg16 such that it only goes up to the conv5_3 layer in forward passes
    # vgg16.eval()
    #
    # for imageset in imagesets:
    #     # load dataset
    #     dataset = VOCDetection(VOC_ROOT_LOCAL, imageset, BaseTransform(300, config.voc['dataset_mean']),
    #                            VOCAnnotationTransform())
    #
    #     get_feature_maps(dataset = dataset,
    #                      net = vgg16,
    #                      imageset_name=imageset[0][0] + imageset[0][1],
    #                      save_dir= save_dir)

    """ Get PCA and whitening params on hold-out dataset (VOC2012)"""

    #
    #
    # imagesets = [[('2012', 'trainval')],
    #              [('2012', 'bottle_trainval_detect')],
    #               [('2012', 'car_trainval_detect')],
    #               [('2012', 'horse_trainval_detect')],
    #               [('2012', 'sheep_trainval_detect')],
    #               [('2012', 'pottedplant_trainval_detect')]
    #               ]
    #
    # for imageset in imagesets:
    #
    #     # load dataset
    #     dataset = VOCDetection(VOC_ROOT_LOCAL, imageset, BaseTransform(300, config.voc['dataset_mean']), VOCAnnotationTransform())
    #
    #     calculate_PCA_and_whitening_parameters(dataset=dataset,
    #                                            imageset_name=imageset[0][0]+imageset[0][1],
    #                                            save_dir=save_dir)
    #
    #

    """ Make spoc features """
    # Imagesets
    # imagesets = [[('2007', 'trainval')]]
    #
    # for imageset in imagesets:
    #     # load dataset
    #     dataset = VOCDetection(VOC_ROOT_LOCAL, imageset, BaseTransform(586, config.voc['dataset_mean']),
    #                            VOCAnnotationTransform())
    #     #
    #     # calculate_PCA_and_whitening_parameters(dataset=dataset,
    #     #                                        imageset_name=imageset[0][0]+imageset[0][1],
    #     #                                        save_dir=save_dir,
    #     #                                        net=vgg16)
    #     pca_dir = save_dir+'2012trainval586_conv5_3_features_before_relu/'
    #     PCA_param_path = pca_dir + '2012trainvalPCA.pickle' # for now only using the 2012 full trainval PCA
    #     image_features_path = os.path.join(os.getcwd(), save_dir, '2007trainval586_conv5_3_features_before_relu/')
    #     # path_to_image_feature_dir = os.path.join(save_dir,imageset_name+'586_conv5_3_features/')
    #
    #     create_spoc_features(dataset,
    #                          image_features_path,
    #                          PCA_param_path,
    #                          imageset_name=imageset[0][0] + imageset[0][1],
    #                          save_dir=image_features_path + '2012trainvalPCA/')


    """ Calculate complete similarities from each image in trainval 2007 to all other images"""

    # dataset = VOCDetection(VOC_ROOT_LOCAL, [('2007', 'trainval')], BaseTransform(586, config.voc['dataset_mean']),
    #                        VOCAnnotationTransform())
    # image_features_path = os.path.join(os.getcwd(), save_dir, '2007trainval586_conv5_3_features_before_relu/')
    # load_dir_spoc_features = image_features_path + '2012trainvalPCA/'
    # similarity_dir = calculate_all_images_similarities(dataset=dataset,
    #                                                      load_dir_spoc_features = load_dir_spoc_features)
    #

    """ Create density per imageset """
    image_sim_dir = save_dir+'2007trainval586_conv5_3_features_before_relu/2012trainvalPCA/image_similarities/'


    imagesets = [[('2007', 'trainval')],
                 [('2007', 'bottle_trainval_detect')],
                  [('2007', 'car_trainval_detect')],
                  [('2007', 'horse_trainval_detect')],
                  [('2007', 'sheep_trainval_detect')],
                  [('2007', 'pottedplant_trainval_detect')]
                  ]

    for imageset in imagesets:
        print(imageset)
        # load dataset
        dataset = VOCDetection(VOC_ROOT_LOCAL, imageset, BaseTransform(300, config.voc['dataset_mean']), VOCAnnotationTransform())
        calculate_density_per_imageset(dataset=dataset,
                                       load_dir_similarities = image_sim_dir)