from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils.multiclass import unique_labels


def class_cluster_match(y_true, y_pred):
    """Translate prediction labels to maximize the accuracy.

    Translate the prediction labels of a clustering output to enable calc
    of external metrics (eg. accuracy, f1_score, ...). Translation is done by
    maximization of the confusion matrix :math:`C` main diagonal sum
    :math:`\sum{i=0}^{K}C_{i, i}`. Notice the number of cluster has to be equal
     or smaller than the number of true classes.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a clustering algorithm.

    Returns
    -------
    trans : array, shape = [n_classes, n_classes]
        Mapping of y_pred clusters, such that :math:`trans\subseteq y_true`

    References
    ----------
    """

    classes = unique_labels(y_true).tolist()
    n_classes = len(classes)
    clusters = unique_labels(y_pred).tolist()
    n_clusters = len(clusters)

    if n_clusters > n_classes:
        classes += ['DEF_CLASS' + str(i) for i in range(n_clusters - n_classes)]
    elif n_classes > n_clusters:
        clusters += ['DEF_CLUSTER' + str(i) for i in range(n_classes - n_clusters)]

    C = contingency_matrix(y_true, y_pred)
    true_idx, pred_idx = linear_assignment(-C).T

    true_idx = true_idx.tolist()
    pred_idx = pred_idx.tolist()

    true_idx = [classes[idx] for idx in true_idx]
    true_idx = true_idx + sorted(set(classes) - set(true_idx))
    pred_idx = [clusters[idx] for idx in pred_idx]
    pred_idx = pred_idx + sorted(set(clusters) - set(pred_idx))

    return_list = [true_idx[pred_idx.index(y)] for y in y_pred]

    return return_list

