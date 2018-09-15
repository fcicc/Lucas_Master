import numpy as np
import rpy2
from scipy.spatial import distance
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, f1_score, adjusted_rand_score

from rpy2.robjects import r

from package.utils import class_cluster_match

R_ALLOWED_FITNESSES = [('C_index', -1), ('Calinski_Harabasz', 1), ('Davies_Bouldin', -1),
                       ('Dunn', 1), ('Gamma', 1), ('G_plus',
                                                   1), ('GDI11', 1), ('GDI12', 1),
                       ('GDI13', 1), ('GDI21', 1), ('GDI22',
                                                    1), ('GDI23', 1), ('GDI31', 1),
                       ('GDI32', 1), ('GDI33', 1), ('GDI41',
                                                    1), ('GDI42', 1), ('GDI43', 1),
                       ('GDI51', 1), ('GDI52', 1), ('GDI53',
                                                    1), ('McClain_Rao', -1), ('PBM', 1),
                       ('Point_Biserial', 1), ('Ray_Turi', -
                                               1), ('Ratkowsky_Lance', 1),
                       ('SD_Scat', -1), ('SD_Dis', -
                                         1), ('Silhouette', 1), ('Tau', 1),
                       ('Wemmert_Gancarski', 1)]
ALLOWED_FITNESSES = R_ALLOWED_FITNESSES + \
    [('silhouette_sklearn', 1), ('min_silhouette_sklearn', 1)]


r('''
    library('clusterCrit')
    unique_criteria <- function(X, labels, criteria) {
        intIdx <- intCriteria(X, as.integer(labels), criteria)
        intIdx
    }
    ''')


def eval_features(X, ac, metric, samples_dist_matrix, individual):
    """Evaluate individual according to silhouette score."""
    prediction = ac.fit(X * individual).labels_
    if metric == 'min_silhouette_sklearn':
        index1 = np.min(silhouette_samples(X, prediction))
    elif metric == 'silhouette_sklearn':
        index1 = silhouette_score(
            samples_dist_matrix, prediction, metric='precomputed')
    else:
        index1 = r['unique_criteria'](X, prediction, metric)
        index1 = np.asarray(index1)[0][0]

    if 'silhouette' in metric:
        index1 += 1

    return index1,


def non_zero_and_dist(I, J):
    non_zero_index = (np.nonzero(I) and np.nonzero(J))
    return distance.cityblock(I[non_zero_index], J[non_zero_index])


def non_zero_or_dist(I, J):
    non_zero_index = (np.nonzero(I) or np.nonzero(J))
    return distance.cityblock(I[non_zero_index], J[non_zero_index])


def count_and_dist_similar(I, J):
    non_zero_index = (np.nonzero(I) and np.nonzero(J))
    return 1/(np.sum(non_zero_index) + 0.00001)


def count_or_dist_similar(I, J):
    non_zero_index = (np.nonzero(I) or np.nonzero(J))
    return 1/(np.sum(non_zero_index) + 0.00001)


def custom_distance(X):
    """

    :type X: np.ndarray
    """
    dataset_shape = X.shape

    dist_matrix = np.zeros((dataset_shape[0], dataset_shape[0]))
    for i in range(dataset_shape[0]):
        for j in range(dataset_shape[0]):
            dist_matrix[i, j] = non_zero_and_dist(X[i, :], X[j, :])

    return dist_matrix


def perfect_eval_features(X, y, ac, individual):
    """Evaluate individual according to accuracy and f1-score."""
    prediction = ac.fit(X * individual).labels_

    y_prediction = class_cluster_match(y, prediction)

    y_num = class_cluster_match(prediction, y)

    return accuracy_score(y, y_prediction), f1_score(y_num, prediction, average='weighted')


def evaluate_rate_metrics(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according multiple metrics and scores.
    :type X: numpy.ndarray
    """
    prediction = ac.fit(X * individual).labels_

    y_prediction = class_cluster_match(y, prediction)

    fitness_names = list([fit[0] for fit in R_ALLOWED_FITNESSES])
    X_R = rpy2.robjects.r.matrix(
        rpy2.robjects.FloatVector(X.flatten()), nrow=X.shape[0])

    int_idx = r['unique_criteria'](
        X_R, rpy2.robjects.FloatVector(prediction.tolist()), fitness_names)
    int_idx = [val[0] for val in list(int_idx)]

    silhouette = silhouette_score(
        samples_dist_matrix, prediction, metric='precomputed')
    min_silhouette = np.min(silhouette_samples(X, prediction))
    adj_rand = adjusted_rand_score(y, prediction)
    f1 = f1_score(y, y_prediction, average='weighted')
    acc = accuracy_score(y, y_prediction)
    complexity = int(np.sum(individual))

    return list(int_idx) + [acc, f1, adj_rand, silhouette, min_silhouette, complexity]


def evaluate(toolbox, offspring):
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
