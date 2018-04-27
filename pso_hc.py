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
import subprocess
from pyswarm import pso
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
    if metric != 'silhouette_sklearn':
        index1 = r['unique_criteria'](X, pred, metric)
        index1 = np.asarray(index1)[0][0]
    else:
        index1 = silhouette_score(X, pred)

    if metric in ['Calinski_Harabasz','Dunn','Gamma','G_plus',
        'GDI11','GDI12','GDI13','GDI21','GDI22','GDI23','GDI31','GDI32',
        'GDI33','GDI41','GDI42','GDI43','GDI51','GDI52','GDI53','PBM',
        'Point_Biserial','Ratkowsky_Lance','Silhouette','Tau',
        'Wemmert_Gancarski', 'silhouette_sklearn']:
        index1 = -index1

    return index1


def perfect_eval_features(X, y, ac, individual):
    """Evaluate individual according to accuracy and f1-score."""
    pred = ac.fit(X*individual).labels_

    y_pred = class_cluster_match(y, pred)

    y_num = class_cluster_match(pred, y)

    return accuracy_score(y, y_pred), f1_score(y_num, pred, average='weighted')


def evall_rate_metrics(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according multiple metrics and scores."""
    pred = ac.fit(X*individual).labels_

    y_pred = class_cluster_match(y, pred)

    indexes = ['C_index','Calinski_Harabasz',
        'Davies_Bouldin','Dunn','Gamma','G_plus','GDI11','GDI12','GDI13','GDI21',
        'GDI22','GDI23','GDI31','GDI32','GDI33','GDI41','GDI42','GDI43','GDI51',
        'GDI52','GDI53','McClain_Rao','PBM','Point_Biserial','Ray_Turi',
        'Ratkowsky_Lance','SD_Scat','SD_Dis','Silhouette','Tau','Wemmert_Gancarski']

    int_idx = r['unique_criteria'](X, pred, indexes)
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
    parser.add_argument('--fitness-metric', type=str, default='silhouette_sklearn',
                        help='fitness function to be used from the clusterCrit R package')

    args = parser.parse_args()

    if args.fitness_metric not in ['C_index','Calinski_Harabasz',
        'Davies_Bouldin','Dunn','Gamma','G_plus','GDI11','GDI12','GDI13','GDI21',
        'GDI22','GDI23','GDI31','GDI32','GDI33','GDI41','GDI42','GDI43','GDI51',
        'GDI52','GDI53','McClain_Rao','PBM','Point_Biserial','Ray_Turi',
        'Ratkowsky_Lance','SD_Scat','SD_Dis','Silhouette','Tau','Wemmert_Gancarski',
        'silhouette_sklearn']:
        raise ValueError(args.fitness_metric + ' is not an acceptable fitness metric')

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

    code_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode("utf-8").replace('\n','')
    start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    exec_label = ','.join([code_version, start_time])
    output_summary = open(
        os.path.join(
            input_dir,
            'dataset_analysis' +
            exec_label +
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

    ac = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(y)),
                                         affinity='manhattan',
                                         linkage='complete')

    samples_dist_matrix = distance.squareform(distance.pdist(X_matrix))

    lb = [0]*X_matrix.shape[1]
    ub = [1]*X_matrix.shape[1]
    fitness_fnc = partial(eval_features, X_matrix, ac, args.fitness_metric)
    top, ftop = pso(fitness_fnc, lb, ub, swarmsize=args.pop_size, maxiter=args.num_gen, processes=multiprocessing.cpu_count())

    print(top)

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
            exec_label +
            '_confusion_matrix.csv'))

    if args.evall_rate:
        correlation=pd.DataFrame.from_dict(correlation)
        criteria_names = ['C_index','Calinski_Harabasz',
        'Davies_Bouldin','Dunn','Gamma','G_plus','GDI11','GDI12','GDI13','GDI21',
        'GDI22','GDI23','GDI31','GDI32','GDI33','GDI41','GDI42','GDI43','GDI51',
        'GDI52','GDI53','McClain_Rao','PBM','Point_Biserial','Ray_Turi',
        'Ratkowsky_Lance','SD_Scat','SD_Dis','Silhouette','Tau','Wemmert_Gancarski']
        correlation.columns= criteria_names + [
            'accuracy', 'f1_score', 'adjusted_rand_score', 'silhouette_sklearn', 'complexity']
        correlation = correlation.reset_index(drop=True)

        correlation.to_csv(
            os.path.join(input_dir,
                        'dataset_analysis' +
                        exec_label +
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
                     exec_label +
                     '_filtered_dataset.csv'),
        quoting=csv.QUOTE_NONNUMERIC,
        float_format='%.10f',
        index=True)

    logging.info("Results in " + str(exec_label))

if __name__ == '__main__':
    main()
