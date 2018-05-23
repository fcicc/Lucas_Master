"""Genetic algorithm clustering by hard subspace"""

import argparse
import csv
import glob
import logging
import multiprocessing
import os
import random
import re
import sys
import time
import math
import subprocess
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabaz_score, confusion_matrix,
                             f1_score, silhouette_score, silhouette_samples)
from analysis_utils import class_cluster_match
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

import rpy2.robjects.numpy2ri
from rpy2.robjects import r

R_ALLOWED_FITNESSES = [('C_index', -1), ('Calinski_Harabasz', 1), ('Davies_Bouldin', -1),
                       ('Dunn', 1), ('Gamma', 1), ('G_plus', 1), ('GDI11', 1), ('GDI12', 1),
                       ('GDI13', 1), ('GDI21', 1), ('GDI22', 1), ('GDI23', 1), ('GDI31', 1),
                       ('GDI32', 1), ('GDI33', 1), ('GDI41', 1), ('GDI42', 1), ('GDI43', 1),
                       ('GDI51', 1), ('GDI52', 1), ('GDI53', 1), ('McClain_Rao', -1), ('PBM', 1),
                       ('Point_Biserial', 1), ('Ray_Turi', -1), ('Ratkowsky_Lance', 1),
                       ('SD_Scat', -1), ('SD_Dis', -1), ('Silhouette', 1), ('Tau', 1),
                       ('Wemmert_Gancarski', 1)]
ALLOWED_FITNESSES = R_ALLOWED_FITNESSES + [('silhouette_sklearn', 1), ('min_silhouette_sklearn', 1)]

rpy2.robjects.numpy2ri.activate()

logging.getLogger().setLevel(logging.INFO)

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


def perfect_eval_features(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according to accuracy and f1-score."""
    prediction = ac.fit(X * individual).labels_

    y_pred = class_cluster_match(y, prediction)

    y_num = class_cluster_match(prediction, y)

    return accuracy_score(y, y_pred), f1_score(y_num, prediction, average='weighted')


def evaluate_rate_metrics(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according multiple metrics and scores."""
    prediction = ac.fit(X * individual).labels_

    y_prediction = class_cluster_match(y, prediction)

    int_idx = r['unique_criteria'](X, prediction, [fit[0] for fit in R_ALLOWED_FITNESSES])
    int_idx = [val[0] for val in list(int_idx)]

    silhouette = silhouette_score(samples_dist_matrix, prediction, metric='precomputed')
    min_silhouette = np.min(silhouette_samples(X, prediction))
    adj_rand = adjusted_rand_score(y, prediction)
    f1 = f1_score(y, y_prediction, average='weighted')
    acc = accuracy_score(y, y_prediction)
    complexity = int(np.sum(individual))

    return tuple(int_idx) + (acc, f1, adj_rand, silhouette, min_silhouette, complexity)


def feature_relevance(X, y):
    """Calculate feature relevance according to the internal and external
       feature relevance."""
    clusters = unique_labels(y)
    features = X.columns.values
    C = 1

    mr_s = {feature: [] for feature in features}
    for cluster_i in clusters:
        cluster_instances = X.loc[[i == cluster_i for i in y]]
        not_cluster_instances = X.loc[[i != cluster_i for i in y]]
        for feature in features:
            vi = np.std(cluster_instances[feature])
            ve = np.std(not_cluster_instances[feature])

            mr = ve / (vi + C)
            mr_s[feature].append(mr)

    mr_s = pd.DataFrame.from_dict(mr_s, orient='index')
    mr_s.columns = clusters

    return mr_s


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')
    parser.add_argument('input_file', type=str, help='''input CSV file''')
    parser.add_argument('experiment_name', type=str, help='''name to be used in output files''')
    parser.add_argument('--num-gen', type=int, default=500,
                        help='number of generations')
    parser.add_argument('--pop-size', type=int, default=600,
                        help='number of individuals in the population')
    parser.add_argument('-c', '--use-categorical', action='store_true',
                        help='wether to use features attributes as categorical individual data')
    parser.add_argument('-p', '--perfect', action='store_true',
                        help='wether to use the perfect evaluation function')
    parser.add_argument('-e', '--evall-rate', type=float, default=0,
                        help='rate of best individuals to calculate all metrics')
    parser.add_argument('--min-features', type=int, default=4,
                        help='minimum number of features to be considered')
    parser.add_argument('--max-features', type=int, default=50,
                        help='maximum number of features to be considered')
    parser.add_argument('--fitness-metric', type=str, default='silhouette_sklearn',
                        help='fitness function to be used from the clusterCrit R package')

    args = parser.parse_args()

    if args.fitness_metric not in [fit[0] for fit in ALLOWED_FITNESSES]:
        raise ValueError(args.fitness_metric + ' is not an acceptable fitness metric')

    return args


def extract_subtotals(dataset):
    """Extract subtotals from compositional feature's attributes."""
    compositional_features = [feature for feature in dataset.columns if ' - ' in feature]

    attributes = {}
    for feature in compositional_features:
        big_group = re.search('\[(.*)\]', feature).group(1)
        feature_attributes = re.sub('\[(.*)\]', '', feature).split(' - ')

        if big_group not in attributes:
            attributes[big_group] = [{} for _ in feature_attributes]

        for i, attribute in enumerate(feature_attributes):
            if attribute not in attributes[big_group][i]:
                attributes[big_group][i][attribute] = [0 for _ in range(dataset.shape[0])]

    for i, row in enumerate(dataset.iterrows()):
        for feature in compositional_features:
            if row[1][feature] > 0:
                big_group = re.search("\[(.*)\]", feature).group(1)
                feature_attributes = re.sub("\[(.*)\]", '', feature).split(' - ')

                for j, attribute in enumerate(feature_attributes):
                    attributes[big_group][j][attribute][i] += row[1][feature]

    df = {}
    for big_group in attributes:
        for position, features in enumerate(attributes[big_group]):
            for attribute in features:
                df['[' + big_group + ']' + str(position) + '-' + attribute] = features[attribute]

    df = pd.DataFrame.from_dict(df)

    return df


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


def weighted_flip_bit(individual, negative_w):
    """FlipBit from deap with negative_w more chances of turning 1 to 0, than the reverse"""
    for i, _ in enumerate(individual):
        if individual[i]:
            if random.random() < negative_w:
                individual[i] = 0
        else:
            if random.random() > negative_w:
                individual[i] = 1
    return individual,


def main():
    """Main function."""
    args = argument_parser()

    input_dir = os.path.dirname(args.input_file)

    experiment_name = args.experiment_name
    start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    exec_label = ','.join([experiment_name, start_time])
    output_summary = open(os.path.join(input_dir, 'dataset_analysis{0}.txt'.format(exec_label)), 'w')

    population_rate = math.ceil(args.evall_rate * args.pop_size)

    output_summary.write(str(args) + '\n')

    own_script = open(sys.argv[0])
    own_script_text = own_script.read()
    own_script.close()

    df = pd.read_csv(args.input_file, index_col=0, header=0)

    y = df['petrofacie'].as_matrix()
    del df['petrofacie']
    dataset = df

    if args.use_categorical:
        index = dataset.index
        dataset = dataset.reset_index(drop=True)
        dataset = pd.concat([dataset, extract_subtotals(dataset)], axis=1)
        dataset.index = index
    index = dataset.index
    dataset = dataset.reset_index(drop=True)
    dataset_matrix = dataset.as_matrix()

    logging.info(args)

    samples_dist_matrix = distance.squareform(distance.pdist(dataset_matrix))

    ac = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(y)),
                                         affinity='manhattan',
                                         linkage='complete')

    weight = [fit[1] for fit in ALLOWED_FITNESSES if fit[0] == args.fitness_metric][0]
    creator.create("FitnessMax", base.Fitness,
                   weights=(weight,))

    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    pool = Pool(multiprocessing.cpu_count()-1)

    toolbox.register("map", pool.map)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=dataset_matrix.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    if args.perfect:
        toolbox.register("evaluate", perfect_eval_features, dataset_matrix, y, ac, samples_dist_matrix)
    else:
        toolbox.register("evaluate", eval_features, dataset_matrix, ac, args.fitness_metric, samples_dist_matrix)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    if len(unique_labels(y)) > args.min_features:
        args.min_features = len(unique_labels(y))
        output_summary.write('setting minimum number of features to ' + str(args.min_features) + '\n\n')
    # toolbox.decorate("mate", check_bounds(args.min_features, args.max_features))
    # toolbox.decorate("mutate", check_bounds(args.min_features, args.max_features))

    population = toolbox.population(n=args.pop_size)
    ind = random.choice(range(len(population)))
    for i, _ in enumerate(population[ind]):
        population[ind][i] = 1
    population = list(toolbox.map(partial(force_bounds, args.min_features, args.max_features), population))
    evaluate(toolbox, population)

    if args.evall_rate:
        sample_population = random.sample(population, population_rate)
        correlation = list(
            toolbox.map(partial(evaluate_rate_metrics, dataset_matrix, y, ac, samples_dist_matrix), sample_population))

    NGEN = args.num_gen
    top = []
    feature_selection_rate = []
    for gen in tqdm(range(NGEN)):
        offspring = algorithms.varOr(population, toolbox, args.pop_size, cxpb=0.2, mutpb=0.8)
        evaluate(toolbox, offspring)

        if args.evall_rate:
            sample_offspring = random.sample(offspring, population_rate)
            sample_fits = toolbox.map(partial(evaluate_rate_metrics, dataset_matrix, y, ac, samples_dist_matrix),
                                      sample_offspring)
            correlation += sample_fits

        old_top = top
        if top == []:
            top = tools.selBest(offspring + population, k=1)
        else:
            top = tools.selBest(offspring + top, k=1)

        population = toolbox.select(offspring + population, k=len(population))

        feature_selection_rate.append(list(map(lambda x: x / len(population), np.sum(population, axis=0))))

    top = top[0]

    best_features = [col for col, boolean in zip(dataset.columns.values, top)
                     if boolean]
    best_pred = ac.fit(dataset[best_features]).labels_

    std_variances = dataset.std(axis=0)

    result = {
        feature: std_variance for feature,
            std_variance in zip(
            best_features,
            std_variances)}
    result = pd.DataFrame.from_dict(result, orient='index')
    result.columns = ['std']
    MRs = feature_relevance(dataset, y)
    result = pd.concat([result, MRs], axis=1, join='inner')
    result = result.sort_values('std', ascending=False)
    initial_n_features = dataset_matrix.shape[1]
    final_n_features = len(best_features)
    feature_reduction_rate = final_n_features / initial_n_features

    y_prediction = class_cluster_match(y, best_pred)
    cm = confusion_matrix(y, y_prediction)
    cm = pd.DataFrame(data=cm, index=unique_labels(y),
                      columns=unique_labels(best_pred))

    output_summary.write("SUMMARY:\n")
    result_summary = {'initial number of features': [initial_n_features],
                      'feature reduction rate': [feature_reduction_rate],
                      'final number of features': [final_n_features],
                      'adjusted Rand score': [adjusted_rand_score(y, best_pred)],
                      'silhouette score': [silhouette_score(dataset[best_features], best_pred)],
                      'calinski harabaz score': [calinski_harabaz_score(dataset[best_features], best_pred)],
                      'accuracy score': [accuracy_score(y, y_prediction)],
                      'f1 score': [f1_score(y, y_prediction, average='weighted')]}

    result_summary = pd.DataFrame.from_dict(result_summary).transpose()
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', 10000,
                           'display.width', 1000, 'display.height', 1000):
        output_summary.write(result_summary.to_string(header=False) + '\n\n')

    output_summary.write('confusion matrix:' + '\n')
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', 10000,
                           'display.width', 1000, 'display.height', 1000):
        output_summary.write(str(cm) + '\n')
    cm.to_csv(
        os.path.join(
            input_dir,
            'dataset_analysis{0}_confusion_matrix.csv'.format(exec_label)))

    if args.evall_rate:
        correlation = pd.DataFrame.from_dict(correlation)
        criteria_names = ['C_index', 'Calinski_Harabasz',
                          'Davies_Bouldin', 'Dunn', 'Gamma', 'G_plus', 'GDI11', 'GDI12', 'GDI13', 'GDI21',
                          'GDI22', 'GDI23', 'GDI31', 'GDI32', 'GDI33', 'GDI41', 'GDI42', 'GDI43', 'GDI51',
                          'GDI52', 'GDI53', 'McClain_Rao', 'PBM', 'Point_Biserial', 'Ray_Turi',
                          'Ratkowsky_Lance', 'SD_Scat', 'SD_Dis', 'Silhouette',  'Tau', 'Wemmert_Gancarski']
        correlation.columns = criteria_names + [
            'accuracy', 'f1_score', 'adjusted_rand_score', 'silhouette_sklearn', 'min_silhouette_sklearn', 'complexity']
        correlation = correlation.reset_index(drop=True)

        correlation.to_csv(
            os.path.join(input_dir,
                         'dataset_analysis{0}_metrics.csv'.format(exec_label)),
            float_format='%.10f',
            index=True)

    output_summary.write(own_script_text)

    output_summary.close()

    feature_selection_rate = list(map(list, zip(*feature_selection_rate)))
    df = {feature: sel_rate for feature, sel_rate in zip(dataset.columns.values, feature_selection_rate)}
    df = pd.DataFrame.from_dict(df, orient='index')
    df.to_csv(
        os.path.join(input_dir,
                     'dataset_analysis{0}_selection_rate.csv'.format(exec_label)),
        quoting=csv.QUOTE_NONNUMERIC,
        float_format='%.10f',
        index=True)

    dataset['petrofacie'] = y
    dataset['predicted labels'] = pd.Series(y_prediction)
    dataset.index = index
    dataset[best_features +
            ['petrofacie', 'predicted labels']].to_csv(
        os.path.join(input_dir,
                     'dataset_analysis{0}_filtered_dataset.csv'.format(exec_label)),
        quoting=csv.QUOTE_NONNUMERIC,
        float_format='%.10f',
        index=True)

    logging.info("Results in " + str(exec_label))

def evaluate(toolbox, offspring):
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit


if __name__ == '__main__':
    main()
