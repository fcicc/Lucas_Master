import math
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from random import randint, choice, sample
from typing import List, Any, Union

import numpy as np
import pandas as pd
import rpy2
import sklearn.base
from deap import creator, tools, base, algorithms
from rpy2.robjects import r
from scipy.spatial import distance
from sklearn.metrics import silhouette_samples, silhouette_score, f1_score, accuracy_score, adjusted_rand_score
from tqdm import tqdm

from analysis_utils import class_cluster_match

# rpy2.robjects.numpy2ri.activate()


R_ALLOWED_FITNESSES = [('C_index', -1), ('Calinski_Harabasz', 1), ('Davies_Bouldin', -1),
                       ('Dunn', 1), ('Gamma', 1), ('G_plus', 1), ('GDI11', 1), ('GDI12', 1),
                       ('GDI13', 1), ('GDI21', 1), ('GDI22', 1), ('GDI23', 1), ('GDI31', 1),
                       ('GDI32', 1), ('GDI33', 1), ('GDI41', 1), ('GDI42', 1), ('GDI43', 1),
                       ('GDI51', 1), ('GDI52', 1), ('GDI53', 1), ('McClain_Rao', -1), ('PBM', 1),
                       ('Point_Biserial', 1), ('Ray_Turi', -1), ('Ratkowsky_Lance', 1),
                       ('SD_Scat', -1), ('SD_Dis', -1), ('Silhouette', 1), ('Tau', 1),
                       ('Wemmert_Gancarski', 1)]
ALLOWED_FITNESSES = R_ALLOWED_FITNESSES + [('silhouette_sklearn', 1), ('min_silhouette_sklearn', 1)]


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
        index1 = silhouette_score(samples_dist_matrix, prediction, metric='precomputed')
    else:
        index1 = r['unique_criteria'](X, prediction, metric)
        index1 = np.asarray(index1)[0][0]

    if 'silhouette' in metric:
        index1 += 1

    return index1,


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
    X_R = rpy2.robjects.r.matrix(rpy2.robjects.FloatVector(X.flatten()), nrow=X.shape[0])

    int_idx = r['unique_criteria'](X_R, rpy2.robjects.FloatVector(prediction.tolist()), fitness_names)
    int_idx = [val[0] for val in list(int_idx)]

    silhouette = silhouette_score(samples_dist_matrix, prediction, metric='precomputed')
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


def force_bounds(minimum, maximum, individual):
    """Force complexity bound constraints.

    :param int minimum: minimum allowed number of features
    :param int maximum: maximum allowed number of features
    :param individual: iterable individual
    :return: iterable individual with the desired number of 1 alleles
    """
    used_features = int(np.sum(individual))
    if used_features > maximum:
        extra_features = used_features - maximum
        used_features_idx = np.flatnonzero(individual == 1).tolist()
        turn_off_idx = sample(used_features_idx, extra_features)
        individual[turn_off_idx] = 0
    elif used_features < minimum:
        missing_features = minimum - used_features
        unused_features_idx = np.flatnonzero(individual == 0).tolist()
        turn_on_idx = sample(unused_features_idx, missing_features)
        individual[turn_on_idx] = 1
    return individual


class GAClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):

    metrics_: Union[List[Any], Any]

    def __init__(self, algorithm=None, n_generations=100, perfect=False,
                 min_features=5, max_features=50, fitness_metric='silhouette_sklearn',
                 pop_size=128, pop_eval_rate=0):
        self.algorithm = algorithm
        self.n_generations = n_generations
        self.perfect = perfect
        self.min_features = min_features
        self.max_features = max_features
        self.fitness_metric = fitness_metric
        self.pop_size = pop_size
        self.pop_eval_rate = pop_eval_rate

    def fit(self, X, y=None):
        population_rate = math.ceil(self.pop_eval_rate * self.pop_size)

        weight = [fit[1] for fit in ALLOWED_FITNESSES if fit[0] == self.fitness_metric][0]
        creator.create("FitnessMax", base.Fitness,
                       weights=(weight,))

        samples_dist_matrix = distance.squareform(distance.pdist(X))

        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        pool = Pool(cpu_count() - 1)
        toolbox.register("map", pool.map)
        toolbox.register("attr_bool", randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            n=X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        if self.perfect:
            toolbox.register("evaluate", perfect_eval_features, X, y, self.algorithm, samples_dist_matrix)
        else:
            toolbox.register("evaluate", eval_features, X, self.algorithm, self.fitness_metric, samples_dist_matrix)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.decorate("mate", check_bounds(args.min_features, args.max_features))
        # toolbox.decorate("mutate", check_bounds(args.min_features, args.max_features))

        population = toolbox.population(n=self.pop_size)
        ind = choice(range(len(population)))
        for i, _ in enumerate(population[ind]):
            population[ind][i] = 1
        population = list(toolbox.map(partial(force_bounds, self.min_features, self.max_features), population))
        evaluate(toolbox, population)

        metrics = []
        if population_rate:
            sample_population = sample(population, population_rate)
            metrics = list(
                toolbox.map(partial(evaluate_rate_metrics, X, y, self.algorithm, samples_dist_matrix), sample_population)
            )
            metrics = [metric + [0] for metric in metrics]

        top = []
        feature_selection_rate = []
        for gen in tqdm(range(self.n_generations)):
            offspring = algorithms.varOr(population, toolbox, self.pop_size, cxpb=0.2, mutpb=0.8)
            # noinspection PyTypeChecker
            evaluate(toolbox, offspring)

            if population_rate:
                sample_offspring = sample(offspring, population_rate)
                sample_fits = toolbox.map(partial(evaluate_rate_metrics, X, y, self.algorithm, samples_dist_matrix),
                                          sample_offspring)
                sample_fits = [metric + [gen] for metric in sample_fits]
                metrics += sample_fits

            if not top:
                top = tools.selBest(offspring + population, k=1)
            else:
                top = tools.selBest(offspring + top, k=1)

            population = toolbox.select(offspring + population, k=len(population))

            feature_selection_rate.append(list(map(lambda x: x / len(population), np.sum(population, axis=0))))

        metrics_names = ['C_index', 'Calinski_Harabasz',
                          'Davies_Bouldin', 'Dunn', 'Gamma', 'G_plus', 'GDI11', 'GDI12', 'GDI13', 'GDI21',
                          'GDI22', 'GDI23', 'GDI31', 'GDI32', 'GDI33', 'GDI41', 'GDI42', 'GDI43', 'GDI51',
                          'GDI52', 'GDI53', 'McClain_Rao', 'PBM', 'Point_Biserial', 'Ray_Turi',
                          'Ratkowsky_Lance', 'SD_Scat', 'SD_Dis', 'Silhouette',  'Tau', 'Wemmert_Gancarski']
        metrics_names += ['accuracy', 'f1_score', 'adjusted_rand_score', 'silhouette_sklearn', 'min_silhouette_sklearn',
                          'complexity']
        metrics = pd.DataFrame(metrics, columns=metrics_names+['generation'])

        self.top_ = top[0]
        self.metrics_ = metrics
        self.feature_selection_rate_ = feature_selection_rate
