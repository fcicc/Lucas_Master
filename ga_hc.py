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
from functools import partial
from multiprocessing.pool import Pool

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from deap import algorithms, base, creator, tools
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabaz_score, confusion_matrix,
                             f1_score, silhouette_score)
# from sklearn.metrics.cluster import class_cluster_match
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor

import rpy2.robjects.numpy2ri
from rpy2.robjects import r

rpy2.robjects.numpy2ri.activate()


logging.getLogger().setLevel(logging.INFO)



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

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.metrics.cluster import class_cluster_match
    >>> y_true = ["class1", "class2", "class3", "class1", "class1", "class3"]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> y_pred_translated = class_cluster_match(y_true, y_pred)
    >>> y_pred_translated
    ['class1', 'class1', 'class3', 'class3', 'class1', 'class3']
    >>> confusion_matrix(y_true, y_pred_translated)
    array([[2, 0, 1],
           [1, 0, 0],
           [0, 0, 2]])
    """

    classes = unique_labels(y_true).tolist()
    n_classes = len(classes)
    clusters = unique_labels(y_pred).tolist()
    n_clusters = len(clusters)

    if n_clusters > n_classes:
        classes += ['DEF_CLASS'+str(i) for i in range(n_clusters-n_classes)]
    elif n_classes > n_clusters:
        clusters += ['DEF_CLUSTER'+str(i) for i in range(n_classes-n_clusters)]

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

r('''
    library('clusterCrit')
    unique_criteria <- function(X, labels, criteria) {
        intIdx <- intCriteria(X, as.integer(labels), criteria)
        intIdx
    }
    ''')
def eval_features(X, ac, individual):
    """Evaluate individual according to silhouette score."""
    pred = ac.fit(X*individual).labels_
    index1 = r['unique_criteria'](X, pred, 'Dunn')
    index1 = np.asarray(index1)[0][0]
    index2 = r['unique_criteria'](X, pred, 'Banfeld_Raftery')
    index2 = np.asarray(index2)[0][0]

    return (index1,index2)


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
def evall_all_metrics(X, y, ac, samples_dist_matrix, individual):
    """Evaluate individual according multiple metrics and scores."""
    pred = ac.fit(X*individual).labels_

    y_pred = class_cluster_match(y, pred)

    int_idx = r['all_intern_metrics'](X, pred)
    int_idx = [val[0] for val in list(int_idx)]

    adj_rand = adjusted_rand_score(y, pred)
    f1 = f1_score(y, y_pred, average='weighted')
    acc = accuracy_score(y, y_pred)

    return tuple(int_idx) + (acc, f1, adj_rand)


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
    parser.add_argument(
        'input_dir',
        type=str,
        help='''input directory, containing a CSV dataset with name "dataset.csv"
             ''')
    parser.add_argument('n_clusters', type=int,
                        help='number of desired clusters')
    parser.add_argument('--num-gen', type=int, default=500,
                        help='number of generations')
    parser.add_argument('--pop-size', type=int, default=600,
                        help='number of individuals in the population')
    parser.add_argument('-c', '--use-categorical', action='store_true',
                        help='wether to use features attributes as categorical individual data')
    parser.add_argument('-p', '--perfect', action='store_true',
                        help='wether to use the perfect evaluation function')
    parser.add_argument('-e', '--evall-all', action='store_true',
                        help='wether to use all evaluation metrics available')

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


def main():
    """Main function."""
    args = argument_parser()

    clear_incomplete_experiments(args.input_dir)

    start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    output_summary = open(
        os.path.join(
            args.input_dir,
            'dataset_analysis' +
            start_time +
            '.txt'),
        'w')

    output_summary.write(str(args) + '\n')

    output_summary.write('\n\nARGS = ' + str(args) + '\n')
    own_script = open(sys.argv[0])
    own_script_text = own_script.read()
    own_script.close()

    df = pd.read_csv(os.path.join(args.input_dir, 'dataset.csv'))

    y = df['petrofacie'].as_matrix()
    del df[df.columns[0]]
    del df['petrofacie']
    X = df

    if args.use_categorical:
        X = pd.concat([X, extract_subtotals(X)], axis=1)
    X_matrix = X.as_matrix()

    logging.info(args)

    samples_dist_matrix = distance.squareform(distance.pdist(X_matrix))

    alg_parameters = {'n_clusters': [args.n_clusters],
                      'affinity': ['manhattan'],
                      'linkage': ['complete']
                      }
    alg_parameters = ParameterGrid(alg_parameters)
    ac = cluster.AgglomerativeClustering(n_clusters=args.n_clusters,
                                         affinity='manhattan',
                                         linkage='complete')

    creator.create("FitnessMax", base.Fitness, weights=(1, -1))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    pool = Pool(multiprocessing.cpu_count())
    # pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    # toolbox.register("map_async", pool.map_async)
    toolbox.register("map", pool.map)

    toolbox.register("attr_bool", random.choice, [1, 0])
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if args.perfect:
        toolbox.register("evaluate", perfect_eval_features, X_matrix, y, ac)
    else:
        toolbox.register("evaluate", eval_features, X_matrix, ac)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=args.pop_size)
    fits = toolbox.map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    if args.evall_all:
        correlation = list(pool.map(partial(evall_all_metrics, X_matrix, y, ac, samples_dist_matrix),
                       toolbox.select(population, k=10)))
    
    NGEN = args.num_gen
    top = []
    for gen in tqdm(range(NGEN)):
        offspring = algorithms.varAnd(
            population, toolbox, cxpb=0.5, mutpb=0.5)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        if args.evall_all:
            best_offspring = list(tools.selBest(offspring, k=10))
            best_fits = pool.map(partial(evall_all_metrics, X_matrix, y, ac, samples_dist_matrix),
                                best_offspring)
            correlation += best_fits

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

    y_pred = class_cluster_match(y, best_pred)
    cm = confusion_matrix(y, y_pred)
    cm = pd.DataFrame(data=cm, index=unique_labels(y),
                      columns=unique_labels(best_pred))

    output_summary.write('adjusted Rand score: ' +
                         str(adjusted_rand_score(y, best_pred)) + '\n')
    output_summary.write('silhouette score: ' +
                         str(silhouette_score(X[best_features], best_pred)) +
                         '\n')
    output_summary.write('calinski harabaz score: ' +
                         str(calinski_harabaz_score(X[best_features],
                             best_pred)) + '\n')
    output_summary.write('accuracy score: ' +
                         str(accuracy_score(y, y_pred)) + '\n')
    output_summary.write(
        'f1 score: ' + str(f1_score(y, y_pred, average='weighted')) + '\n',)

    output_summary.write('confusion matrix:' + '\n')
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', 10000,
                           'display.width', 1000, 'display.height', 1000):
        output_summary.write(str(cm) + '\n')
    cm.to_csv(
        os.path.join(
            args.input_dir,
            'dataset_analysis' +
            start_time +
            '_confusion_matrix.csv'))

    if args.evall_all:
        correlation=pd.DataFrame.from_dict(correlation)
        correlation = correlation.drop_duplicates()
        criteria_names = list(map(lambda x: str(x).lower(), r('getCriteriaNames(TRUE)')))
        correlation.columns= criteria_names + [
            'accuracy', 'f1_score', 'adjusted_rand_score']
        correlation = correlation.sort_values(by='calinski_harabasz', axis='rows')
        correlation = correlation.reset_index(drop=True)

        output_summary.write('Metric correlations:')
        correlation.to_csv(
            os.path.join(args.input_dir,
                        'dataset_analysis' +
                        start_time +
                        '_metrics.csv'),
            float_format='%.10f',
            index=True)
        correlation.corr().to_csv(
            os.path.join(args.input_dir,
                        'dataset_analysis' +
                        start_time +
                        '_metrics_correlation.csv'),
            float_format='%.10f',
            index=True)

        # plt.figure()
        # ax=correlation[['adjusted_rand_score']].plot(lw=1)
        # correlation[['silhouette']].plot(lw=1, ax=ax, linestyle='--')
        # correlation[['accuracy']].plot(lw=1, ax=ax, linestyle='-.')
        # correlation[['f1_score']].plot(lw=1, ax=ax, linestyle=(0, (5, 10)))
        # correlation[['calinski_harabasz']].plot(
        #     secondary_y=True, ax=ax, lw=.5)
        # plt.savefig(
        #     os.path.join(
        #         args.input_dir,
        #         'dataset_analysis' +
        #         start_time +
        #         '_plot.png'),
        #     format='png', dpi=900)

        objective_space = correlation[['accuracy', 'f1_score', 'ratkowsky_lance']]
        objective_space.drop_duplicates()
        objective_space = objective_space.apply(
                        lambda x: x.map(lambda y: y+random.random()/500))
        # plt.figure()
        # points = plt.scatter(objective_space['accuracy'],
        #                     objective_space['f1_score'],
        #                     c=objective_space['ratkowsky_lance'],
        #                     s=3, cmap='viridis', alpha=0.7)
        # plt.colorbar(points, label='ratkowsky_lance')
        # sns.regplot("accuracy", "f1_score", data=objective_space, scatter=False)
        # plt.savefig(
        #     os.path.join(
        #         args.input_dir,
        #         'dataset_analysis' +
        #         start_time +
        #         '_objective_space.png'),
        #     format='png', dpi=900)

    output_summary.write(own_script_text)

    output_summary.close()

    X['petrofacie'] = y
    X[best_features +
      ['petrofacie']].to_csv(
        os.path.join(args.input_dir,
                     'dataset_analysis' +
                     start_time +
                     '_filtered_dataset.csv'),
        quoting=csv.QUOTE_NONNUMERIC,
        float_format='%.10f',
        index=False)


if __name__ == '__main__':
    main()
