import argparse
import csv
import glob
import os
import re
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils.multiclass import unique_labels
from sqlalchemy import asc
from sqlalchemy.exc import OperationalError

from orm_models import Result, local_create_session


def argument_parser() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Analysis over input data''')
    parser.add_argument('--input-file', type=str, help='''input CSV file''', default=None)
    parser.add_argument('-o', '--output_file', type=str, help='''output file''')
    parser.add_argument('-p', '--plot-correlation', action='store_true', help='plot correlation')
    parser.add_argument('--axis1', type=str, help='''first plot axis''', default='silhouette_sklearn')
    parser.add_argument('--axis2', type=str, help='''second plot axis''', default='adjusted_rand_score')
    parser.add_argument('--color', type=str, help='''point color''', default='generation')
    parser.add_argument('-c', '--correlation', action='store_true', help='generates correlation matrix')
    parser.add_argument('-m', '--merge-results', action='store_true', help='')
    parser.add_argument('-n', '--n-last-results', type=int, help='', default=10)
    parser.add_argument('-t', '--petrel', action='store_true', help='')
    parser.add_argument('-u', '--useful-features', action='store_true', help='')
    parser.add_argument('-b', '--clear-incomplete-outputs', action='store_true', help='')
    parser.add_argument('-e', '--melt-results', action='store_true', help='')
    parser.add_argument('-d', '--average-feature-selection', action='store_true', help='')
    parser.add_argument('-s', '--list-results', action='store_true')
    parser.add_argument('-r', '--list-result', action='store_true')
    parser.add_argument('-k', '--confusion-matrix', action='store_true')
    parser.add_argument('-f', '--filter', action='store_true')
    parser.add_argument('--id', type=int, default=None)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--db-file', type=str, default='./local.db', help='sqlite file to store results')

    args = parser.parse_args()

    if sum([args.plot_correlation, args.correlation,
            args.merge_results, args.petrel,
            args.useful_features, args.melt_results,
            args.average_feature_selection, args.list_results,
            args.list_result, args.confusion_matrix,
            args.filter]) != 1:
        raise ValueError("Cannot have this combination of arguments.")

    return args


def confusion_matrix(args):
    if args.id is None and args.exp_name is None:
        raise ValueError(f'Both id and exp-name attributes cannot be empty')

    session = local_create_session(args.db_file)

    cm = []
    if args.id:
        try:
            result: Result = session.query(Result).filter(Result.id == args.id).first()

            cm = result.confusion_matrix.as_dataframe()

        except OperationalError:
            print(f'No results found with id {args.id}')
    else:
        try:
            results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()

            cm = sum([result.confusion_matrix.as_dataframe() for result in results])

        except OperationalError:
            print(f'No results found with name {args.exp_name}')

    if args.output_file:
        cm.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cm)

    session.close()


def plot_correlation(args):
    if args.id is None and args.exp_name is None:
        raise ValueError(f'Both id and exp-name attributes cannot be empty')

    session = local_create_session(args.db_file)

    individual_evaluations = []
    results = []
    if args.id:
        try:
            result: Result = session.query(Result).filter(Result.id == args.id).first()

            individual_evaluations = result.individual_evaluations

            results = [result]

        except OperationalError:
            print(f'No results found with id {args.id}')
    else:
        try:
            results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()

            individual_evaluations = pd.concat([result.individual_evaluations for result in results])

        except OperationalError:
            print(f'No results found with name {args.exp_name}')

    df = individual_evaluations
    # df = df.sample(frac=1 / len(results))

    plt.figure()
    x = df[args.axis1].values
    y = df[args.axis2].values
    c = df[args.color].values
    points = plt.scatter(x, y, c=c, s=1, cmap='viridis', alpha=0.5)
    plt.colorbar(points, label=args.color)

    sns.regplot(args.axis1, args.axis2, data=df, scatter=False, x_jitter=0.005, y_jitter=0.005, order=1, robust=False)

    plt.show()

    session.close()


def melt_results(args):
    summary_files = glob.glob(args.input_file + '/dataset_analysis*_metrics.csv')
    summary_files.sort(key=os.path.getmtime, reverse=True)
    print('Found ' + str(len(summary_files)) + ' files')
    summary_files = summary_files[:min(len(summary_files), args.n_last_results)]
    print('Processing ' + str(len(summary_files)) + ' files:')

    dfs = [pd.read_csv(summary, index_col=0) for summary in summary_files]
    for i, df in enumerate(dfs):
        df['trial'] = pd.Series([i] * df.shape[0])

    df = pd.concat(dfs)
    df = df.reset_index()
    df['index'] = df['index'].apply(lambda x: round(float(x) / 128))

    fig, ax = plt.subplots()

    for key, grp in df.groupby(['trial', 'index']):
        print(grp)
        ax = grp.mean().plot(ax=ax, kind='line', x='index', y='silhouette_sklearn', c=key, label=key)

    plt.legend(loc='best')

    if args.output_file:
        plt.savefig(args.output_file, dpi=600)
    else:
        plt.show()


def correlation_calculation(df, args):
    correlation = df.corr()

    if args.output_file:
        correlation.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(correlation)


def merge_results(args):
    session = local_create_session(args.db_file)

    results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()

    all_results = [[result.accuracy, result.f_measure, result.silhouette] for result in results]
    all_results = pd.DataFrame(all_results, columns=['accuracy', 'f_measure', 'silhouette'])
    all_results = pd.concat([all_results, all_results.describe()])

    session.close()

    if args.output_file:
        all_results.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(all_results)


def correct_wells_names(well_name):
    if well_name == 'ENC-1A-RJS':
        well_name = '1-ENC-0001A-RJS'
    if well_name == 'MRK-5-RJS':
        well_name = '1-MRK-0005-RJS'
    if well_name == 'MRK-4P-RJS':
        well_name = '1-MRK-0004P-RJS'
    if well_name == 'RJS-100':
        well_name = '1-RJS-0100-RJ'
    if well_name == 'RJS-105':
        well_name = '1-RJS-0105-RJ'
    if well_name == 'ENC-2-RJS':
        well_name = '3-ENC-0002-RJS'
    if well_name == 'ENC-3-RJS':
        well_name = '3-ENC-0003-RJS'
    if well_name == 'PRG-1-RJS':
        well_name = '3-PRG-0001-RJS'

    return well_name


def petrofacies_to_petrel(df):
    thin_section_names = df.index.values
    wells = map(partial(re.search, '([\w|-]+) [-+]?[0-9]*\.?[0-9]*'), thin_section_names)
    wells = list(map(lambda x: x.group(1), wells))
    wells = list(map(correct_wells_names, wells))
    depths = map(partial(re.search, ' ([-+]?[0-9]*\.?[0-9]*)'), thin_section_names)
    depths = map(lambda x: x.group(1), depths)
    depths = list(map(float, depths))
    petrofacies = df['petrofacie'].values

    epsilon = 0.01

    for well in set(wells):
        a1 = []
        a2 = []
        for depth, thin_section_name, petrofacie, well_i in zip(depths, thin_section_names, petrofacies, wells):
            if well_i != well:
                continue
            a1.append(['%.2f' % depth, '%.2f' % (depth + epsilon), '"' + ('%.2f' % depth) + '"'])
            a2.append(['%.2f' % depth, '%.2f' % (depth + epsilon), '"' + petrofacie + '"'])
        file1 = open(well + '_1.txt', 'w')
        file1.write('WELL: "' + well + '"\n')
        file2 = open(well + '_2.txt', 'w')
        file2.write('WELL: "' + well + '"\n')
        for line1, line2 in zip(a1[:-1], a2[:-1]):
            file1.write('\t'.join(line1) + '\n')
            file2.write('\t'.join(line2) + '\n')
        file1.write('\t'.join(a1[-1]))
        file2.write('\t'.join(a2[-1]))

        file1.close()
        file2.close()


def find_useful_features(args):
    df = pd.read_csv(args.input_file, index_col=0)
    del df['petrofacie']

    full_std = df.std()

    groups_std = {}
    for key, group in df.groupby('predicted labels'):
        groups_std[key] = full_std / group.std()
        for i, column in enumerate(group):
            if group[column].apply(lambda x: x == 0).all():
                groups_std[key][column] = pd.Series([0])

    df = pd.DataFrame.from_dict(groups_std)
    df['full std'] = full_std
    df.sort_values(by=['full std'])

    if args.output_file:
        df.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(df)


def average_feature_selection(args):
    session = local_create_session(args.db_file)

    results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()
    avg_n_features = np.mean([result.final_n_features for result in results])

    print(f'Average number of features: {avg_n_features}')

    session.close()


def show_results(args):
    session = local_create_session(args.db_file)

    try:
        results = session.query(Result).order_by(asc(Result.start_time)).all()

        for result in results:
            print(result)

    except OperationalError:
        print('No results found')

    session.close()


def show_result(args):
    session = local_create_session(args.db_file)

    try:
        result = session.query(Result).filter(Result.id == args.id).first()
        print(result.details())
    except AttributeError:
        print(f'Result {args.id} not found in {args.db_file}')

    session.close()


def filter_dataset(args):
    df: pd.DataFrame = pd.read_csv(args.input_file, index_col=0)

    session = local_create_session(args.db_file)

    columns = []
    try:
        result = session.query(Result).filter(Result.id == args.id).first()
        columns = [selected_feature.column for selected_feature in result.selected_features]
    except AttributeError:
        print(f'Result {args.id} not found in {args.db_file}')

    session.close()

    df = df.filter(items=columns+['petrofacie'])

    if args.output_file:
        df.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(df)


def main():
    args = argument_parser()

    if args.plot_correlation:
        plot_correlation(args)
    elif args.correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        correlation_calculation(df, args)
    elif args.merge_results:
        merge_results(args)
    elif args.useful_features:
        find_useful_features(args)
    elif args.petrel:
        df = pd.read_csv(args.input_file, index_col=0)
        petrofacies_to_petrel(df)
    elif args.melt_results:
        melt_results(args)
    elif args.average_feature_selection:
        average_feature_selection(args)
    elif args.list_results:
        show_results(args)
    elif args.list_result:
        show_result(args)
    elif args.confusion_matrix:
        confusion_matrix(args)
    elif args.filter:
        filter_dataset(args)


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
    """

    classes = unique_labels(y_true).tolist()
    n_classes = len(classes)
    clusters = unique_labels(y_pred).tolist()
    n_clusters = len(clusters)

    if n_clusters > n_classes:
        classes += ['DEF_CLASS' + str(i) for i in range(n_clusters - n_classes)]
    elif n_classes > n_clusters:
        clusters += ['DEF_CLUSTER' + str(i) for i in range(n_classes - n_clusters)]

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


if __name__ == '__main__':
    main()
