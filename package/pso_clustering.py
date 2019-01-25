import math
import operator
import random
import warnings
from functools import partial
from multiprocessing.pool import Pool

import pandas as pd
import sklearn
from deap import base
from deap import creator
from deap import tools
from scipy.spatial import distance
from tqdm import tqdm

from package.evaluation_functions import ALLOWED_FITNESSES, eval_features, evaluate_rate_metrics, evaluate


def generate(size, creator, position_minimum, position_maximum, speed_minimum, speed_maximum):
    part = creator.Individual(random.uniform(position_minimum, position_maximum) for _ in range(size))
    part.speed = [random.uniform(speed_minimum, speed_maximum) for _ in range(size)]
    part.speed_minimum = speed_minimum
    part.speed_maximum = speed_maximum
    return part


def update_particle(part, global_best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.local_best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, global_best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.speed_minimum:
            part.speed[i] = part.speed_minimum
        elif speed > part.speed_maximum:
            part.speed[i] = part.speed_maximum
    part[:] = list(map(operator.add, part, part.speed))


def setup_creator(fitness_metric):
    weight = [fit[1] for fit in ALLOWED_FITNESSES if fit[0] == fitness_metric][0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        creator.create("FitnessMax", base.Fitness, weights=(weight,))
        creator.create("Individual", list, fitness=creator.FitnessMax, speed=list,
                       speed_minimum=None, speed_maximum=None, local_best=None)


def setup_toolbox(data_shape):
    toolbox = base.Toolbox()
    mod_speed = 0.1
    toolbox.register("individual", generate, creator=creator, size=data_shape[1], position_minimum=-10,
                     position_maximum=10, speed_minimum=-mod_speed, speed_maximum=mod_speed)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("update", update_particle, phi1=1.0, phi2=1.0)

    return toolbox


class PSOClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):

    def __init__(self, algorithm=None, n_generations=100, perfect=False,
                 fitness_metric='silhouette_sklearn',
                 pop_size=128, pop_eval_rate=0):
        self.algorithm = algorithm
        self.n_generations = n_generations
        self.perfect = perfect
        self.fitness_metric = fitness_metric
        self.pop_size = pop_size
        self.pop_eval_rate = pop_eval_rate

    def fit(self, X, y=None):
        population_rate = math.ceil(self.pop_eval_rate * self.pop_size)

        samples_dist_matrix = distance.squareform(distance.pdist(X))

        setup_creator(self.fitness_metric)
        toolbox = setup_toolbox(X.shape)
        toolbox.register("evaluate", eval_features, X, self.algorithm, self.fitness_metric, samples_dist_matrix, y)

        pool = Pool(initializer=setup_creator, initargs=[self.fitness_metric])
        toolbox.register("map", pool.map)

        population = toolbox.population(n=self.pop_size)
        ind = random.choice(range(len(population)))
        population[ind][:] = [0] * X.shape[1]

        evaluate_rate_function = partial(evaluate_rate_metrics, X, y, self.algorithm, samples_dist_matrix)

        metrics = []
        global_best = None
        feature_selection_rate = []
        for gen in tqdm(range(self.n_generations)):
            evaluate(toolbox, population)

            if population_rate:
                sample_offspring = random.sample(population, population_rate)
                sample_fits = toolbox.map(evaluate_rate_function, sample_offspring)
                sample_fits = [metric + [gen] for metric in sample_fits]
                metrics += sample_fits

            for particle in population:
                if not particle.local_best or particle.local_best.fitness < particle.fitness:
                    particle.local_best = creator.Individual(particle)
                    particle.local_best.fitness.values = particle.fitness.values

                if not global_best or global_best.fitness < particle.fitness:
                    global_best = creator.Individual(particle)
                    global_best.fitness.values = particle.fitness.values

            for particle in population:
                toolbox.update(particle, global_best)

        metrics_names = ['C_index', 'Calinski_Harabasz', 'Davies_Bouldin', 'Dunn', 'Gamma', 'G_plus', 'GDI11', 'GDI12',
                         'GDI13', 'GDI21', 'GDI22', 'GDI23', 'GDI31', 'GDI32', 'GDI33', 'GDI41', 'GDI42', 'GDI43',
                         'GDI51', 'GDI52', 'GDI53', 'McClain_Rao', 'PBM', 'Point_Biserial', 'Ray_Turi',
                         'Ratkowsky_Lance', 'SD_Scat', 'SD_Dis', 'Silhouette', 'Tau', 'Wemmert_Gancarski']
        metrics_names += ['accuracy', 'f1_score', 'adjusted_rand_score', 'silhouette_sklearn', 'min_silhouette_sklearn',
                          'complexity']
        metrics = pd.DataFrame(metrics, columns=metrics_names + ['generation'])

        pool.close()

        self.global_best_ = global_best
        self.population_ = population
        self.metrics_ = metrics
        self.feature_selection_rate_ = feature_selection_rate
