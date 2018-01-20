#!/usr/bin/env python
import numpy as np
import violajones.AdaBoost as ab

LOADING_BAR_LENGTH = 40


"""
Implementation of attentional cascade architecture with 
"""
def cascade_fix(num_classifier_levels, positive_iis, negative_iis, features):
    cascade = []
    alpha_set = []

    images = positive_iis + negative_iis 
    labels = np.hstack((np.ones(len(positive_iis)), np.zeros(len(negative_iis))))

    print("\nNum of images entering to cascade: " + str(len(images)) + "\n")

    # select cascade 
    for idx, classifier in enumerate(num_classifier_levels):
        print("Calculating the " + str(idx+1) + "th layer classifier...\n")
        classifiers, alpha = ab.learn(positive_iis, negative_iis, features, classifier)
        cascade.append(classifiers)
        alpha_set.append(alpha)

        # reject negative example of each strong classifier
        idx_img_reject = [idx for idx, img in enumerate(images) if ab.ensemble_vote(img, classifiers, alpha) != labels[idx]]
        for idx in range(len(idx_img_reject)-1, -1, -1):   
            if idx_img_reject[idx] < len(positive_iis): del positive_iis[idx_img_reject[idx]]
            elif idx_img_reject[idx]  >= len(positive_iis): del negative_iis[idx_img_reject[idx] - len(positive_iis)]
        
        images = positive_iis + negative_iis        
        print("\nAfter the " + str(idx+1) + "th layer classifier, num of rest images: " + str(len(images)))

    return cascade, alpha_set

def cascade(num_classifier_levels, positive_iis, negative_iis, F_target=0.3, f=0.4, d=0.7, min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
    cascade = []
    j = 1
    images = positive_iis + negative_iis
    print ("num of images: " + str(len(images)))
    F = 1
    D = 1

    for i in num_classifier_levels:
        print("Calculating the " + str(j) + "th layer classifier...\n")
        F_previous = F
        classifiers = ab.learn(positive_iis, negative_iis, i, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
        cascade.append(classifiers)
        # reject negative example of each strong classifier
        for idx, img in enumerate(images):
            if ab.ensemble_vote(img, classifiers) != 1:
                if idx < len(positive_iis): del positive_iis[idx]
                elif idx > len(positive_iis): del negative_iis[idx-len(positive_iis)]
        
        # false negative : real negative, predict positive 
        f = ab.ensemble_vote_all(negative_iis, classifiers) / len(negative_iis)
        # recall : real positive, predict positive
        d = ab.ensemble_vote_all(negative_iis, classifiers) / len(negative_iis)
        print("rest num of images: " + str(len(images)))
        if F <= F_target : break
        print("After " + str(j) + " layers classifiers, the false negative rate is " + str(F))
        j += 1

    return cascade




