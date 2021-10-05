import numpy as np
from sklearn.metrics import *


def evaluate_supervised_external(labels_true, labels_predicted):
    """
    Compare the predicted results with the real ones (supervised methods) with some different metrics
    """
    supervised_scores = dict(
        AMI=adjusted_mutual_info_score(labels_true, labels_predicted),
        ARI=adjusted_rand_score(labels_true, labels_predicted),
        compl=completeness_score(labels_true, labels_predicted),
        homo=homogeneity_score(labels_true, labels_predicted),
        vmeas=v_measure_score(labels_true, labels_predicted))
    return supervised_scores


def evaluate_unsupervised_internal(x: np.ndarray, labels_predicted):
    """
    Evaluate the predicted results using unsupervised metrics
    """
    unsupervised_scores = dict(db=davies_bouldin_score(x, labels_predicted),
                               silhouette=silhouette_score(x, labels_predicted))
    return unsupervised_scores
