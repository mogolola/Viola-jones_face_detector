#!/usr/bin/env python
import os
import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.AttentionalCascade as ac

if __name__ == "__main__":
    pos_training_path = 'trainset/faces'
    neg_training_path = 'trainset/non-faces'
    # pos_training_path = 'trainset/faces_short'
    # neg_training_path = 'trainset/non-faces_short'
    pos_testing_path = 'testset/faces'
    neg_testing_path = 'testset/non-faces'

    num_classifier_levels = [1, 2, 2, 5, 5]
    # For train speed restricting feature size
    min_feature_height = 6
    max_feature_height = 12
    min_feature_width = 6
    max_feature_width = 12

    # target false negative rate
    F_target = 0.3
    # the maximum acceptable false positive rate per layer
    f = 0.4
    # the mimnimum acceptable detection rate per layer 
    d = 0.7

    # load trainset images
    print('Training classifiers..\nLoading faces..')
    faces_training = ii.load_images(pos_training_path)
    faces_ii_training = list(map(ii.to_integral_image, faces_training))
    print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_training = ii.load_images(neg_training_path)
    non_faces_ii_training = list(map(ii.to_integral_image, non_faces_training))
    print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

    img_height, img_width = faces_ii_training[0].shape

    # classifiers are haar like features
    # classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers,, min_feature_width, max_feature_width min_feature_height, max_feature_height)
    # cascade = ac.cascade(num_classifier_levels, faces_ii_training, non_faces_ii_training, F_target, f, d, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    features = ab._create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    cascade, alpha_set = ac.cascade_fix(num_classifier_levels, faces_ii_training, non_faces_ii_training, features)
    
    # load testset images
    print('\nLoading test faces..')
    faces_testing = ii.load_images(pos_testing_path)
    faces_ii_testing = list(map(ii.to_integral_image, faces_testing))
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = ii.load_images(neg_testing_path)
    non_faces_ii_testing = list(map(ii.to_integral_image, non_faces_testing))
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0

    for idx, classifier in enumerate(cascade):
        correct_faces = sum(ab.ensemble_vote_all(faces_ii_testing, classifier, alpha_set[idx]))
        correct_non_faces = len(non_faces_testing) - sum(ab.ensemble_vote_all(non_faces_ii_testing, classifier, alpha_set[idx]))

        print('Result after ' + str(idx+1) + ' layer(s):\n     Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
            + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
            + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
            + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')
    
    # when the program finishes, remind user
    os.system('say "your program has finished"')  
