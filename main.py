import argparse
import datetime

import warnings
from rpy2.rinterface import RRuntimeWarning

warnings.filterwarnings("ignore", category=RRuntimeWarning)

import pandas as pd
import rpy2
from sklearn import cluster
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score, \
    accuracy_score, f1_score
from sklearn.utils.multiclass import unique_labels
from sqlalchemy import create_engine

from analysis_utils import class_cluster_match
from ga_clustering import ALLOWED_FITNESSES, GAClustering
from orm_interface import store_results
from orm_models import Base, DB_NAME, CONN_STRING

rpy2.robjects.r['options'](warn=-1)


def argument_parser() -> argparse.Namespace:
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
                        help='fitness function to be used')
    parser.add_argument('--cluster-algorithm', type=str, default='agglomerative',
                        help='cluster algorithm to be used')
    parser.add_argument('-d', '--dont-use-ga', action='store_true',
                        help='disables the use of GA and apply cluster to all dimensions')

    args = parser.parse_args()

    if args.fitness_metric not in [fit[0] for fit in ALLOWED_FITNESSES]:
        raise ValueError(args.fitness_metric + ' is not an acceptable fitness metric')

    return args


def run():
    engine = create_engine(CONN_STRING, echo=False)
    Base.metadata.create_all(engine)

    args = argument_parser()

    df = pd.read_csv(args.input_file, index_col=0, header=0)

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

    start_time = datetime.datetime.now()
    ga = GAClustering(algorithm=ac, n_generations=args.num_gen, perfect=args.perfect, min_features=args.min_features,
                      max_features=args.max_features, fitness_metric=args.fitness_metric, pop_size=args.pop_size,
                      pop_eval_rate=args.eval_rate)
    end_time = datetime.datetime.now()

    if args.eval_rate:
        ga.fit(dataset_matrix, y=y)
    else:
        ga.fit(dataset_matrix)

    best_features = [col for col, boolean in zip(dataset.columns.values, ga.top_)
                     if boolean]
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
                              start_time, end_time, cm, args, best_features, args.experiment_name, ga.metrics_)

    print(f'Results stored under the ID {result_id}')


if __name__ == '__main__':
    run()
