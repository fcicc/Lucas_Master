import math
import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import random

import numpy as np
import pandas as pd
import sklearn.base
from deap import creator, tools, base, algorithms
from scipy.spatial import distance
from tqdm import tqdm

from package.evaluation_functions import ALLOWED_FITNESSES, eval_features, evaluate, \
    DICT_ALLOWED_FITNESSES, eval_multiple


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
        turn_off_idx = random.sample(used_features_idx, extra_features)
        individual[turn_off_idx] = 0
    elif used_features < minimum:
        missing_features = minimum - used_features
        unused_features_idx = np.flatnonzero(individual == 0).tolist()
        turn_on_idx = random.sample(unused_features_idx, missing_features)
        individual[turn_on_idx] = 1
    return individual


def check_bounds(minimum, maximum):
    """Enforce complexity bounds to offspring.
    :param int minimum: minimum allowed number of features
    :param int maximum: maximum allowed number of features
    :return: wrapped function
    """
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            offspring = map(partial(force_bounds, minimum, maximum), offspring)

            return offspring

        return wrapper

    return decorator


def setup_creator(fitness_metric):
    weight = [fit[1] for fit in ALLOWED_FITNESSES if fit[0] == fitness_metric][0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        creator.create("FitnessMax", base.Fitness, weights=(weight,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def setup_toolbox(data_shape, min_features, max_features, algorithm, n_clusters):
    toolbox = base.Toolbox()

    if algorithm is None:
        toolbox.register("attr_bool", random.randint, 0, n_clusters-1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=data_shape[0])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_clusters-1, indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=3)

    else:
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=data_shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=3)

        toolbox.decorate("mate", check_bounds(min_features, max_features))
        toolbox.decorate("mutate", check_bounds(min_features, max_features))

    return toolbox


class GAClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):

    def __init__(self, algorithm=None, n_generations=100, perfect=False,
                 min_features=5, max_features=1000, fitness_metric='silhouette_sklearn',
                 pop_size=128, pop_eval_rate=0, n_clusters=2):
        self.algorithm = algorithm
        self.n_generations = n_generations
        self.perfect = perfect
        self.min_features = min_features
        self.max_features = max_features
        self.fitness_metric = fitness_metric
        self.pop_size = pop_size
        self.pop_eval_rate = pop_eval_rate
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        population_rate = math.ceil(self.pop_eval_rate * self.pop_size)

        samples_dist_matrix = distance.squareform(distance.pdist(X))

        setup_creator(self.fitness_metric)
        toolbox = setup_toolbox(X.shape, self.min_features, self.max_features, self.algorithm, self.n_clusters)
        toolbox.register("evaluate", eval_features, X, self.algorithm, self.fitness_metric, samples_dist_matrix, y)

        n_threads = math.floor(cpu_count() / 2)
        pool = Pool(processes=n_threads, initializer=setup_creator, initargs=[self.fitness_metric])
        toolbox.register("map", pool.map)

        population = toolbox.population(n=self.pop_size)
        ind = random.choice(range(len(population)))
        if self.algorithm is not None:
            population[ind][:] = [0]*X.shape[1]
            population = list(map(partial(force_bounds, self.min_features, self.max_features), population))

        evaluate_rate_methods = ['accuracy']
        evaluate_rate_function = partial(eval_multiple, X, self.algorithm, evaluate_rate_methods, samples_dist_matrix, y)

        global_best = None
        feature_selection_rate = []
        evaluate(toolbox, population)
        metrics = []
        gens_since_last_improvment = 0
        current_silhouette = 0
        with tqdm(total=self.n_generations, desc=f'{n_threads} threads', postfix={'gens_without_improv': gens_since_last_improvment}) as pbar:
            for gen in range(self.n_generations):
                pbar.update(1)
                if population_rate > 0:
                    sample_offspring = random.sample(population, population_rate)
                    sample_fits = toolbox.map(evaluate_rate_function, sample_offspring)
                    sample_fits = [metric + [gen] for metric in sample_fits]
                    metrics += sample_fits

                offspring = algorithms.varOr(population, toolbox, self.pop_size, cxpb=0.5, mutpb=0.5)
                offspring = evaluate(toolbox, offspring)

                if global_best is None:
                    global_best = deepcopy(tools.selBest(offspring + population, k=1)[0])
                else:
                    old_global_best = deepcopy(global_best.fitness.values)
                    global_best = deepcopy(tools.selBest(offspring + [global_best], k=1)[0])
                    pbar.set_postfix({'gens_without_improv': gens_since_last_improvment, 'current_fitness': global_best.fitness.values})
                    if old_global_best == global_best.fitness.values:
                        gens_since_last_improvment += 1
                    else:
                        gens_since_last_improvment = 0

                population = toolbox.select(offspring + population, k=len(population))

                feature_selection_rate.append(list(map(lambda x: x / len(population), np.sum(population, axis=0))))

                if gens_since_last_improvment >= 200:
                    break

        metrics = pd.DataFrame(metrics, columns=evaluate_rate_methods + ['generation'])

        pool.close()

        self.global_best_ = global_best
        self.population_ = population
        self.metrics_ = metrics
        self.feature_selection_rate_ = feature_selection_rate
