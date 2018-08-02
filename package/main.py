import copy

import pandas as pd
import rpy2

from package.coclustering import CoClustering
# from package.preprocessing import binaryze_column, range_grain_size
from package.pso_clustering import PSOClustering
from sklearn import cluster
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score, \
    accuracy_score, f1_score

import argparse
import datetime
import warnings

from rpy2.rinterface import RRuntimeWarning
from tqdm import tqdm

from sklearn.utils.multiclass import unique_labels

from package.utils import class_cluster_match
from package.ga_clustering import ALLOWED_FITNESSES, GAClustering
from package.orm_interface import store_results
from package.orm_models import create_if_not_exists

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
    parser.add_argument('--max-features', type=int, default=50,
                        help='maximum number of features to be considered')
    parser.add_argument('--fitness-metric', type=str, default='silhouette_sklearn',
                        help='fitness function to be used', choices=[fitnes_str for fitnes_str, _ in ALLOWED_FITNESSES])
    parser.add_argument('--cluster-algorithm', type=str, default='agglomerative',
                        help='cluster algorithm to be used', choices=['agglomerative', 'kmeans'])
    parser.add_argument('-d', '--dont-use-ga', action='store_true',
                        help='disables the use of GA and apply cluster to all dimensions')
    parser.add_argument('-o', '--db-file', type=str, default='./local.db',
                        help='sqlite file to store results')
    parser.add_argument('-s', '--strategy', type=str, default='ga',
                        help='ga(Genetic Algorithm) or PSO (Particle Swarm Optimization)', choices=['ga', 'pso'])
    parser.add_argument('-n', '--run-multiple', type=int, default=1,
                        help='number of multiple runs')
    parser.add_argument('--beta', type=float, default=1,
                        help='silhouette upscale explained by the grain-size')

    args = parser.parse_args(args=args)

    if args.fitness_metric not in [fit[0] for fit in ALLOWED_FITNESSES]:
        raise ValueError(args.fitness_metric +
                         ' is not an acceptable fitness metric')

    return args


def upscale_grain_size(df, beta):
    alpha = 1
    learning_rate = 1

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    ac = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(Y)),
                                         affinity='manhattan',
                                         linkage='complete')

    y = ac.fit_predict(X)
    silhouette = silhouette_score(X, y)
    target_silhouette = silhouette * beta

    internal_df = copy.deepcopy(df)
    alphas = [alpha]
    silhouettes = [silhouette]

    variance_variable = [column for column in df.columns.values if '(mm)' in column][0]

    count_loop = 0
    while silhouette < target_silhouette:
        count_loop += 1
        alpha += learning_rate
        internal_df[variance_variable] = df[variance_variable] * alpha
        X = internal_df.iloc[:, :-1].values
        y = ac.fit_predict(X)
        silhouette = silhouette_score(X, y)
        alphas += [alpha]
        silhouettes += [silhouette]

        if count_loop > 100000:
            print('Breaking!')
            break

    print(f'Alpha = {alpha}')
    print(f'Silhouette = {silhouette}')

    df[variance_variable] = internal_df[variance_variable]

    return df


def run(args=None):
    args = argument_parser(args)

    create_if_not_exists(args.db_file)

    df = pd.read_csv(args.input_file, index_col=0, header=0)

    if 'Microscopic - Sorting:' in df.columns:
        del df['Microscopic - Sorting:']

    if args.beta > 1:
        df = upscale_grain_size(df, args.beta)

    y = df['petrofacie'].values
    del df['petrofacie']

    dataset = df

    dataset = dataset.reset_index(drop=True)
    dataset_matrix = dataset.values

    ac = None
    if args.cluster_algorithm == 'agglomerative':
        ac = cluster.AgglomerativeClustering(n_clusters=len(unique_labels(y)),
                                             affinity='manhattan',
                                             linkage='complete')
    elif args.cluster_algorithm == 'kmeans':
        ac = cluster.KMeans(n_clusters=len(unique_labels(y)), n_init=100)

    if len(unique_labels(y)) > args.min_features:
        args.min_features = len(unique_labels(y))

    results_ids = []
    for _ in tqdm(range(args.run_multiple)):
        strategy_clustering = None
        if args.strategy == 'ga':
            strategy_clustering = GAClustering(algorithm=ac, n_generations=args.num_gen, perfect=args.perfect,
                                               min_features=args.min_features, max_features=args.max_features,
                                               fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                               pop_eval_rate=args.eval_rate)
        elif args.strategy == 'pso':
            strategy_clustering = PSOClustering(algorithm=ac, n_generations=args.num_gen, perfect=args.perfect,
                                                fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                                pop_eval_rate=args.eval_rate)
        elif args.strategy == 'cocluster':
            strategy_clustering = CoClustering(algorithm=ac, n_generations=args.num_gen, perfect=args.perfect,
                                               fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                                               pop_eval_rate=args.eval_rate)

        start_time = datetime.datetime.now()
        if args.eval_rate:
            strategy_clustering.fit(dataset_matrix, y=y)
        else:
            strategy_clustering.fit(dataset_matrix)
        end_time = datetime.datetime.now()

        best_features = [col for col, boolean in zip(dataset.columns.values, strategy_clustering.global_best_)
                         if boolean]
        # population = strategy_clustering.population_.sort(attrgetter('fitness.values'))
        # best_half_population = np.sum(population[:math.ceil(len(population)/2)], axis=0)
        # best_features = [feature != 0 for feature in best_half_population]
        # best_features = [col for col, boolean in zip(dataset.columns.values, best_features)
        #                  if boolean]

        best_prediction = ac.fit(dataset[best_features]).labels_

        std_variances = dataset.std(axis=0)

        result = {
            feature: std_variance for feature, std_variance in zip(best_features, std_variances)}
        result = pd.DataFrame.from_dict(result, orient='index')
        result.columns = ['std']
        initial_n_features = dataset_matrix.shape[1]
        final_n_features = len(best_features)

        y_prediction = class_cluster_match(y, best_prediction)
        cm = confusion_matrix(y, y_prediction)
        cm = pd.DataFrame(data=cm, index=unique_labels(y),
                          columns=unique_labels(best_prediction))

        accuracy = accuracy_score(y, y_prediction)
        f_measure = f1_score(y, y_prediction, average='weighted')
        adj_rand_score = adjusted_rand_score(y, best_prediction)
        silhouette = silhouette_score(dataset[best_features], best_prediction)

        result_id = store_results(accuracy, f_measure, adj_rand_score, silhouette, initial_n_features, final_n_features,
                                  start_time, end_time, cm, args, best_features, args.experiment_name,
                                  strategy_clustering.metrics_, args.db_file, best_prediction)

        results_ids.append(result_id)

    print(f'Results stored under the ID {results_ids}')

    return results_ids


if __name__ == '__main__':
    run()
