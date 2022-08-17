# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

"""
Idea:
    - Model is confused when the best n softmax scores have similar values.
    - Measure the confusion score for every unlabelled image.
    - Request the labels of images that confuses the model the most

"""



softmax_scores_unlabelled_images = np.load("softmax_scores.npy")

softmax_average_by_entry = np.mean(softmax_scores_unlabelled_images, axis = 1)

softmax_max_by_entry = np.max(softmax_scores_unlabelled_images, axis = 1)

potential_number_of_confusions = np.round(1/softmax_max_by_entry)
potential_number_of_confusions = potential_number_of_confusions.astype(np.int)


softmax_confusion = []
softmax_confusion_scores = []



for i in tqdm(range(len(softmax_scores_unlabelled_images))):
    entry = softmax_scores_unlabelled_images[i]
    confusion_item_count = potential_number_of_confusions[i]
    
    confused_class_id = np.argsort(entry)[-confusion_item_count:]
    confused_class_scores = entry[confused_class_id]
    
    confusion_score = np.std(confused_class_scores)
    
    softmax_confusion.append(confused_class_id)
    softmax_confusion_scores.append(confusion_score)
    
    
    
softmax_confusion_scores = np.array(softmax_confusion_scores)    
softmax_confusion = np.array(softmax_confusion)    


# starts from most confused image, ends with least confused image
most_confused_image_indices = np.argsort(softmax_confusion_scores)[np.where(np.sort(softmax_confusion_scores) 
                                                                            > 0)[0][0:]]



















































