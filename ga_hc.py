"""Genetic algorithm clustering by hard subspace"""

import argparse
import csv
import glob
import logging
import multiprocessing
import os
import random
import re
import shutil
import sys
import time
import math
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabaz_score, confusion_matrix,
                             f1_score, silhouette_score)
from sklearn.metrics.cluster import class_cluster_match
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

import rpy2.robjects.numpy2ri
from rpy2.robjects import r

rpy2.robjects.numpy2ri.activate()


logging.getLogger().setLevel(logging.INFO)



r('''
    library('clusterCrit')
    unique_criteria <- function(X, labels, criteria) {
        intIdx <- intCriteria(X, as.integer(labels), criteria)
        intIdx
    }
    ''')
def eval_features(X, ac, metric, individual):
    """Evaluate individual according to silhouette score."""
    pred = ac.fit(X*individual).labels_
    index1 = r['unique_criteria'](X, pred, metric)
    # index1 = np.asarray(index1)[0][0]
    # index1 = silhouette_score(X, pred)

    return (index1,)


def perfect_eval_features(X, y, ac, individual):
    """Evaluate individual according to accuracy and f1-score."""
    pred = ac.fit(X*individual).labels_

    y_pred = class_cluster_match(y, pred)

    y_num = class_cluster_match(pred, y)

    return accuracy_score(y, y_pred), f1_score(y_num, pred, average='weighted')


r('''
    library('clusterCrit')
    all_intern_metrics <- function(X, labels) {
        intIdx <- intCriteria(X, as.integer(labels), 'all')
        intIdx
    }
    ''')
def evall_rate_metrics(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according multiple metrics and scores."""
    pred = ac.fit(X*individual).labels_

    y_pred = class_cluster_match(y, pred)

    int_idx = r['all_intern_metrics'](X, pred)
    int_idx = [val[0] for val in list(int_idx)]

    silhouette = silhouette_score(X, pred)
    adj_rand = adjusted_rand_score(y, pred)
    f1 = f1_score(y, y_pred, average='weighted')
    acc = accuracy_score(y, y_pred)
    complexity = int(np.sum(individual))

    return tuple(int_idx) + (acc, f1, adj_rand, silhouette, complexity)


def feature_relevance(X, y):
    """Calculate feature relevance according to the internal and external
       feature relevance."""
    clusters = unique_labels(y)
    features = X.columns.values
    C = 1

    MRs = {feature: [] for feature in features}
    for cluster_i in clusters:
        cluster_instances = X.loc[[i == cluster_i for i in y]]
        not_cluster_instances = X.loc[[i != cluster_i for i in y]]
        for feature in features:
            VI = np.std(cluster_instances[feature])
            VE = np.std(not_cluster_instances[feature])

            MR = VE / (VI + C)
            MRs[feature].append(MR)

    MRs = pd.DataFrame.from_dict(MRs, orient='index')
    MRs.columns = clusters

    return MRs


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')
    parser.add_argument('input_file', type=str, help='''input CSV file''')
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
    parser.add_argument('--fitness-metric', type=str, default='silhouette',
                        help='fitness function to be used from the clusterCrit R package')

    args = parser.parse_args()

    return args


def extract_subtotals(X):
    """Extract subtotals from compositional feature's attributes."""
    compositional_features = [feature for feature in X.columns if ' - ' in feature]

    attributes = {}
    for feature in compositional_features:
        big_group = re.search('\[(.*)\]', feature).group(1)
        feature_attrs = re.sub('\[(.*)\]', '', feature).split(' - ')

        if big_group not in attributes:
            attributes[big_group] = [{} for _ in feature_attrs]

        for i, attribute in enumerate(feature_attrs):
            if attribute not in attributes[big_group][i]:
                attributes[big_group][i][attribute] = [0 for _ in range(X.shape[0])]

    for i, row in enumerate(X.iterrows()):
        for feature in compositional_features:
            if row[1][feature] > 0:
                big_group = re.search('\[(.*)\]', feature).group(1)
                feature_attrs = re.sub('\[(.*)\]', '', feature).split(' - ')

                for j, attribute in enumerate(feature_attrs):
                    attributes[big_group][j][attribute][i] += row[1][feature]

    df = {}
    for big_group in attributes:
        for position, features in enumerate(attributes[big_group]):
            for attribute in features:
                df['['+big_group+']'+str(position)+'-'+attribute] = features[attribute]

    df = pd.DataFrame.from_dict(df)

    return df

def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                used_features = int(np.sum(child))
                if used_features > max:
                    extra_features =  used_features - max
                    used_features_idx = np.flatnonzero(child==1).tolist()
                    turn_off_idx = random.sample(used_features_idx, extra_features)
                    child[turn_off_idx] = 0
                elif used_features < min:
                    missing_features = min - used_features
                    unused_features_idx = np.flatnonzero(child==0).tolist()
                    turn_on_idx = random.sample(unused_features_idx, missing_features)
                    child[turn_on_idx] = 1

            return offspring
        return wrapper
    return decorator


def clear_incomplete_experiments(directory):
    """Search the input directory for incomplete run files and erase them."""
    results_regex = os.path.join(directory,'dataset_analysis'+('[0-9]'*4)+'_'+('[0-9]'*2)+'_'+('[0-9]'*2)+'-'+('[0-9]'*2)+'_'+('[0-9]'*2)+'_'+('[0-9]'*2)+'.txt')
    summary_files = glob.glob(results_regex)

    for summary_file in summary_files:
        file = open(summary_file)
        content = file.read()
        file.close()
        if 'adjusted rand' not in content.lower():
            run_id = re.search('dataset_analysis(.*).txt', summary_file).group(1)
            run_files_regex = os.path.join(directory, 'dataset_analysis'+run_id+'*')
            for run_file in glob.glob(run_files_regex):
                os.remove(run_file)


def weighted_flipBit(individual, negative_w):
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
    clear_incomplete_experiments(input_dir)

    start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    output_summary = open(
        os.path.join(
            input_dir,
            'dataset_analysis' +
            start_time +
            '.txt'),
        'w')

    population_rate = math.ceil(args.evall_rate * args.pop_size)

    output_summary.write(str(args) + '\n')
    
    own_script = open(sys.argv[0])
    own_script_text = own_script.read()
    own_script.close()

    df = pd.read_csv(args.input_file, index_col=0)

    y = df['petrofacie'].as_matrix()
    del df['petrofacie']
    X = df

    if args.use_categorical:
        index = X.index
        X = X.reset_index(drop=True)
        X = pd.concat([X, extract_subtotals(X)], axis=1)
        X.index = index
    index = X.index
    X = X.reset_index(drop=True)
    X_matrix = X.as_matrix()

    logging.info(args)

    samples_dist_matrix = distance.squareform(distance.pdist(X_matrix))

    ac = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(y)),
                                         affinity='manhattan',
                                         linkage='complete')

    creator.create("FitnessMax", base.Fitness, weights=(1, ))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    pool = Pool(multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    toolbox.register("attr_bool", lambda : random.choices([1, 0], weights=[0.01, 0.99], k=1)[0])
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=X_matrix.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if args.perfect:
        toolbox.register("evaluate", perfect_eval_features, X_matrix, y, ac)
    else:
        toolbox.register("evaluate", eval_features, X_matrix, ac, fitness_metric)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", weighted_flipBit, negative_w=0.9)
    toolbox.register("select", tools.selRoulette)

    toolbox.decorate("mate", checkBounds(args.min_features, args.max_features))
    toolbox.decorate("mutate", checkBounds(args.min_features, args.max_features))

    population = toolbox.population(n=args.pop_size)
    fits = toolbox.map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    if args.evall_rate:
        sample_population = random.sample(population, population_rate)
        correlation = list(pool.map(partial(evall_rate_metrics, X_matrix, y, ac, samples_dist_matrix), sample_population))

    NGEN = args.num_gen
    top = []
    for gen in tqdm(range(NGEN)):
        offspring = algorithms.varOr(population, toolbox, args.pop_size, cxpb=0.2, mutpb=0.8)

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        if args.evall_rate:
            sample_offspring = random.sample(offspring, population_rate)
            sample_fits = pool.map(partial(evall_rate_metrics, X_matrix, y, ac, samples_dist_matrix), sample_offspring)
            correlation += sample_fits

        old_top = top
        if top == []:
            top = tools.selBest(offspring + population, k=1)
        else:
            top = tools.selBest(offspring + top, k=1)

        population = toolbox.select(offspring+population, k=len(population))

    top = top[0]

    best_features = [col for col, boolean in zip(X.columns.values, top)
                     if boolean]
    best_pred = ac.fit(X[best_features]).labels_

    std_variances = X.std(axis=0)

    result = {
        feature: std_variance for feature,
        std_variance in zip(
            best_features,
            std_variances)}
    result = pd.DataFrame.from_dict(result, orient='index')
    result.columns = ['std']
    MRs = feature_relevance(X, y)
    result = pd.concat([result, MRs], axis=1, join='inner')
    result = result.sort_values('std', ascending=False)
    initial_n_features = X_matrix.shape[1]
    final_n_features = len(best_features)
    feature_reduction_rate = final_n_features/initial_n_features

    y_pred = class_cluster_match(y, best_pred)
    cm = confusion_matrix(y, y_pred)
    cm = pd.DataFrame(data=cm, index=unique_labels(y),
                      columns=unique_labels(best_pred))

    output_summary.write("SUMMARY:\n")
    result_summary = {'initial number of features': [initial_n_features],
                      'feature reduction rate' : [feature_reduction_rate],
                      'final number of features' : [final_n_features],
                      'adjusted Rand score' : [adjusted_rand_score(y, best_pred)],
                      'silhouette score' : [silhouette_score(X[best_features], best_pred)],
                      'calinski harabaz score' : [calinski_harabaz_score(X[best_features], best_pred)],
                      'accuracy score' : [accuracy_score(y, y_pred)],
                      'f1 score' : [f1_score(y, y_pred, average='weighted')]}

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
            'dataset_analysis' +
            start_time +
            '_confusion_matrix.csv'))

    if args.evall_rate:
        correlation=pd.DataFrame.from_dict(correlation)
        criteria_names = list(map(lambda x: str(x).lower(), r('getCriteriaNames(TRUE)')))
        correlation.columns= criteria_names + [
            'accuracy', 'f1_score', 'adjusted_rand_score', 'silhouette_sklearn', 'complexity']
        correlation = correlation.reset_index(drop=True)

        correlation.to_csv(
            os.path.join(input_dir,
                        'dataset_analysis' +
                        start_time +
                        '_metrics.csv'),
            float_format='%.10f',
            index=True)

    output_summary.write(own_script_text)

    output_summary.close()

    X['petrofacie'] = y
    X.index = index
    X[best_features +
      ['petrofacie']].to_csv(
        os.path.join(input_dir,
                     'dataset_analysis' +
                     start_time +
                     '_filtered_dataset.csv'),
        quoting=csv.QUOTE_NONNUMERIC,
        float_format='%.10f',
        index=True)

    logging.info("Results in " + str(start_time))

if __name__ == '__main__':
    main()
