import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import rpy2
from rpy2.rinterface import RRuntimeWarning
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from package.CheatingClustering import CheatingClustering
from package.coclustering import CoClustering
from package.evaluation_functions import DICT_ALLOWED_FITNESSES, eval_multiple
from package.ga_clustering import ALLOWED_FITNESSES, GAClustering
from package.orm_interface import store_results
from package.orm_models import create_db_if_not_exists
from package.pso_clustering import PSOClustering
from package.utils import class_cluster_match
from package.ward_p import WardP

warnings.filterwarnings("ignore", category=RRuntimeWarning)

rpy2.robjects.r['options'](warn=-1)


def argument_parser(args) -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')

    parser.add_argument('input_file', type=str, default='test_scenario',
                        help='input CSV file')
    parser.add_argument('experiment_name', type=str, default='experiment',
                        help='name to be used in output files')
    parser.add_argument('--level', type=str, default='features_groups', choices=['features_group', 'features'],
                        help='input CSV file')
    parser.add_argument('--num-gen', type=int, default=500,
                        help='number of generations')
    parser.add_argument('--pop-size', type=int, default=600,
                        help='number of individuals in the population')
    parser.add_argument('-p', '--perfect', action='store_true',
                        help='whether to use the perfect evaluation as fitness function')
    parser.add_argument('-e', '--eval-rate', type=float, default=0,
                        help='rate of random-sampled individuals to calculate all metrics')
    parser.add_argument('--min-features', type=int, default=4,
                        help='minimum number of features to be considered')
    parser.add_argument('--fitness-metric', type=str, default='silhouette_sklearn',
                        help='fitness function to be used', choices=[fitnes_str for fitnes_str, _ in ALLOWED_FITNESSES])
    parser.add_argument('--cluster-algorithm', type=str, default='agglomerative',
                        help='cluster algorithm to be used', choices=['agglomerative', 'kmeans',
                                                                      'affinity-propagation', 'perfect-classifier'])
    parser.add_argument('-o', '--db-file', type=str, default='./local.db',
                        help='sqlite file to store results')
    parser.add_argument('-s', '--strategy', type=str, default='none',
                        help='ga(Genetic Algorithm) or PSO (Particle Swarm Optimization)', choices=['ga', 'pso',
                                                                                                    'ward_p',
                                                                                                    'random_ga',
                                                                                                    'none'])
    parser.add_argument('--p_ward', type=float, default=2,
                        help='Ward P exponential value')
    parser.add_argument('--scenario', nargs='+', help='List of scenarios of features to be used', required=True)

    args = parser.parse_args(args=args)

    assert args.fitness_metric in [fit[0] for fit in ALLOWED_FITNESSES]

    return args


def run(args=None):
    args = argument_parser(args)

    create_db_if_not_exists(args.db_file)

    dataset, y = parse_excel_dataset_to_df(args)

    clustering_algorithm = select_clustering_algorithm(args, y)

    meta_clustering = select_meta_clustering_algorithm(args, clustering_algorithm, dataset, y)

    start_time = datetime.datetime.now()
    if args.eval_rate:
        meta_clustering.fit(dataset.values, y=y)
    else:
        meta_clustering.fit(dataset.values)
    end_time = datetime.datetime.now()

    if type(clustering_algorithm) == cluster.KMeans:
        clustering_algorithm = cluster.KMeans(n_clusters=len(unique_labels(y)), n_init=100)

    result_id, scores = save_output_to_db(args, clustering_algorithm, dataset, end_time,
                                          meta_clustering, start_time, y)

    print(f'Results stored under the ID {result_id}')

    return scores, result_id


def save_output_to_db(args, clustering_algorithm, dataset, end_time, meta_clustering, start_time, y):
    best_features = []
    best_prediction = []
    if args.strategy == 'none':
        best_features = dataset.columns.values
        best_prediction = meta_clustering.labels_
        meta_clustering.metrics_ = pd.DataFrame()
    elif args.strategy == 'pso':
        best_weights = meta_clustering.global_best_
        best_prediction = clustering_algorithm.fit(dataset.values * best_weights).labels_
        best_features = dataset.columns.values
    elif args.strategy == 'ga':
        best_features = [col for col, boolean in zip(dataset.columns.values, meta_clustering.global_best_)
                         if boolean]
        best_prediction = clustering_algorithm.fit(dataset[best_features]).labels_
    elif args.strategy == 'ward_p':
        best_features = dataset.columns.values
        best_prediction = meta_clustering.labels_
        meta_clustering.metrics_ = pd.DataFrame()
    elif args.strategy == 'random_ga':
        best_features = dataset.columns.values
        best_prediction = meta_clustering.global_best_
        meta_clustering.metrics_ = pd.DataFrame()
    initial_n_features = dataset.values.shape[1]
    final_n_features = len(best_features)
    y_prediction = class_cluster_match(y, best_prediction)
    cm = confusion_matrix(y, y_prediction)
    cm = pd.DataFrame(data=cm, index=unique_labels(y), columns=unique_labels(y))
    best_phenotype = []
    for feature in dataset.columns.values:
        if feature in best_features:
            best_phenotype += [1]
        else:
            best_phenotype += [0]
    scores = calculate_all_scores(best_phenotype, clustering_algorithm, dataset, y)
    result_id = store_results(scores, initial_n_features, final_n_features,
                              start_time, end_time, cm, args, best_features,
                              meta_clustering.metrics_, best_prediction)
    return result_id, scores


def calculate_all_scores(best_phenotype, clustering_algorithm, dataset, y):
    """Calculates every possible metric.
    :rtype: dict
    """
    samples_dist_matrix = distance.squareform(distance.pdist(dataset.values))
    allowed_fitness = list(DICT_ALLOWED_FITNESSES.keys())
    scores = [(fitness_name, fitness_value) for fitness_name, fitness_value in
              zip(allowed_fitness,
                  eval_multiple(dataset.values, clustering_algorithm, allowed_fitness, samples_dist_matrix, y,
                                best_phenotype))]
    scores = dict(scores)
    return scores


def select_meta_clustering_algorithm(args, clustering_algorithm, dataset, y):
    meta_clustering = None
    if args.strategy == 'ga':
        meta_clustering = GAClustering(algorithm=clustering_algorithm, n_generations=args.num_gen, perfect=args.perfect,
                                       min_features=args.min_features,
                                       fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                       pop_eval_rate=args.eval_rate)
    elif args.strategy == 'random_ga':
        meta_clustering = GAClustering(algorithm=None, n_generations=args.num_gen, perfect=args.perfect,
                                       min_features=args.min_features,
                                       fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                       pop_eval_rate=args.eval_rate, n_clusters=len(unique_labels(y)))
    elif args.strategy == 'pso':
        meta_clustering = PSOClustering(algorithm=clustering_algorithm, n_generations=args.num_gen,
                                        perfect=args.perfect,
                                        fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                        pop_eval_rate=args.eval_rate)
    elif args.strategy == 'cocluster':
        meta_clustering = CoClustering(algorithm=clustering_algorithm, n_generations=args.num_gen, perfect=args.perfect,
                                       fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                       pop_eval_rate=args.eval_rate)
    elif args.strategy == 'ward_p':
        kernel_feature = dataset['porosity'].values
        np.delete(dataset.values, dataset.columns.get_loc('porosity'))
        del dataset['porosity']
        meta_clustering = WardP(perfect=args.perfect, kernel_feature=kernel_feature, p=args.p_ward,
                                n_clusters=len(unique_labels(y)))
    elif args.strategy == 'none':
        meta_clustering = clustering_algorithm
    assert meta_clustering is not None
    return meta_clustering


def select_clustering_algorithm(args, y):
    clustering_algorithm = None
    if args.cluster_algorithm == 'agglomerative':
        clustering_algorithm = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(y)),
                                                               linkage='ward')
    elif args.cluster_algorithm == 'kmeans':
        clustering_algorithm = cluster.KMeans(n_clusters=len(unique_labels(y)), n_init=10)
    elif args.cluster_algorithm == 'affinity-propagation':
        clustering_algorithm = cluster.AffinityPropagation(preference=-250)
    elif args.cluster_algorithm == 'perfect-classifier':
        clustering_algorithm = CheatingClustering(y=y)
    assert clustering_algorithm is not None
    return clustering_algorithm


def parse_excel_dataset_to_df(args):
    df = pd.read_excel(args.input_file, index_col=0, header=[0, 1, 2])
    scenario = [subset for subset in args.scenario[0] if subset in df.columns.get_level_values('top_level')]
    df = df[scenario + ['others']]
    df = df.groupby(level=[args.level], axis=1).sum()

    # if 'grain_size' in df.columns:
    #     del df['grain_size']
    if 'Cluster' in df.columns:
        del df['Cluster']
    if 'Cluster label' in df.columns:
        del df['Cluster label']
    if 'phi stdev sorting' in df.columns:
        del df['phi stdev sorting']
    y = df['petrofacie'].values
    del df['petrofacie']
    df = df.reset_index(drop=True)
    return df, y


if __name__ == '__main__':
    run()
