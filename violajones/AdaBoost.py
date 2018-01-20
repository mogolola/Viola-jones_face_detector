#!/usr/bin/env python
import numpy as np
import progressbar
from functools import partial
from multiprocessing import Pool
from violajones.HaarLikeFeature import HaarLikeFeature
from violajones.HaarLikeFeature import FeatureTypes

LOADING_BAR_LENGTH = 40


def learn(positive_iis, negative_iis, features, num_classifiers=-1):
    """
    Selects a set of classifiers. Iteratively takes the best classifiers based
    on a weighted error. Implementation of table 1 of the paper.
    :param positive_iis: List of positive integral image examples
    :type positive_iis: list[numpy.ndarray]
    :param negative_iis: List of negative integral image examples
    :type negative_iis: list[numpy.ndarray]
    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int

    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]
    """
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg

    images = positive_iis + negative_iis
    print("adaboost - num of images: " + str(len(images)))
    
    # positive and negative examples labels
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

    # Create initial weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))

    num_features = len(features)
    print(num_features)

    num_classifiers = num_features if num_classifiers == -1 else num_classifiers
    print('Num of classifiers of current layer\'s adaboost: ' + str(num_classifiers) + '\n')

    # select classifiers
    classifiers = []
    alpha = []

    print('Selecting classifiers..')
    bar = progressbar.ProgressBar()
    for _ in bar(range(num_classifiers)):

        classification_errors = np.zeros(len(features))

        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        for idx, feature in enumerate(features):
            # classifier error is the sum of image weights where the classifier is right
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != _get_feature_vote(feature, images[img_idx]) else 0, range(num_imgs)))
            classification_errors[idx] = error

        # get best feature, i.e. with smallest error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature = features[min_error_idx]
        # add weak classifiers to strong classifier
        classifiers.append(best_feature)
        
        beta = best_error / (1 - best_error)  # parameter using to update weights and calculate strong classifier
        alpha.append(np.log(1/beta)) 
        # print(alpha)

        # update image weights
        # weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs))))
        weights = np.array(list(map(lambda img_idx: weights[img_idx] if labels[img_idx] != _get_feature_vote(best_feature, images[img_idx]) else weights[img_idx] * beta, range(num_imgs))))

        # remove feature (a feature can't be selected twice)
        # feature_indexes.remove(best_feature_idx)
        del features[min_error_idx]
        # votes = np.delete(votes, best_feature_idx, 1)

        print(len(features))
        print(min_error_idx)
        # print(votes.shape)

    return classifiers, alpha


def ensemble_vote(int_img, classifiers, alpha):
    """
    Strong classifier: classifies given integral image (numpy array) using given classifiers, 
    i.e. if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    
    return 1 if sum([alpha[i] * c.get_vote(int_img) for i, c in enumerate(classifiers)]) > 0.5 * sum(alpha) else 0

def ensemble_vote_all(int_imgs, classifiers, alpha):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else 0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers, alpha=alpha)
    return list(map(vote_partial, int_imgs))


def _get_feature_vote(feature, image):
    return feature.get_vote(image)

def _create_features(img_height, img_width, min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
    """
    Create Haar like features.
    """
    print('Creating haar-like features..')
    
    # Maximum feature width and height default to image width and height
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    features = []
    for feature in FeatureTypes:   # FeatureTypes are just tuples
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
    
    print('..done. ' + str(len(features)) + ' features created.\n')
    return features