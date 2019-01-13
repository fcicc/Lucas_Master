import argparse
import csv
import glob
import os
import re
from collections import Iterable
from itertools import groupby
from typing import List

import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import asc
from sqlalchemy.exc import OperationalError

from package.orm_models import Result, local_create_session
from package.preprocessing import binaryze_column
from package.utils import class_cluster_match


def argument_parser(args) -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Analysis over input data''')
    parser.add_argument('--input-file', type=str, help='''input CSV file''', default=None)
    parser.add_argument('-o', '--output_file', type=str, help='''output file''')
    parser.add_argument('-p', '--plot-correlation', action='store_true', help='plot correlation')
    parser.add_argument('--axis1', type=str, help='''first plot axis''', default='silhouette_sklearn')
    parser.add_argument('--axis2', type=str, help='''second plot axis''', default='adjusted_rand_score')
    parser.add_argument('--color', type=str, help='''point color''', default='generation')
    parser.add_argument('--correlation', action='store_true', help='generates correlation matrix')
    parser.add_argument('--merge-results', action='store_true', help='')
    parser.add_argument('--n-last-results', type=int, help='', default=10)
    parser.add_argument('--useful-features', action='store_true', help='')
    parser.add_argument('--clear-incomplete-outputs', action='store_true', help='')
    parser.add_argument('--melt-results', action='store_true', help='')
    parser.add_argument('--average-feature-selection', action='store_true', help='')
    parser.add_argument('--list-results', action='store_true')
    parser.add_argument('--detail-result', action='store_true')
    parser.add_argument('--confusion-matrix', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--id', type=int, default=None, nargs='+')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--db-file', type=str, default='./local.db', help='sqlite file to store results')
    parser.add_argument('--select-best', action='store_true')
    parser.add_argument('--logs-folder', type=str, default=None)
    parser.add_argument('--plot-logs', action='store_true')
    parser.add_argument('--export-results', action='store_true')

    args = parser.parse_args(args=args)

    if sum([args.plot_correlation, args.correlation,
            args.merge_results,
            args.useful_features, args.melt_results,
            args.average_feature_selection, args.list_results,
            args.detail_result, args.confusion_matrix,
            args.filter, args.select_best,
            args.plot_logs, args.export_results]) != 1:
        raise ValueError("Cannot have this combination of arguments.")

    return args


def confusion_matrix(args):
    if args.id is None and args.exp_name is None:
        raise ValueError(f'Both id and exp-name attributes cannot be empty')

    session = local_create_session(args.db_file)

    if args.id:
        try:
            results: List[Result] = session.query(Result).filter(Result.id.in_(args.id)).all()

        except OperationalError:
            print(f'No results found with id {args.id}')
    else:
        try:
            results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()

        except OperationalError:
            print(f'No results found with name {args.exp_name}')

    cm = sum([result.confusion_matrix.as_dataframe() for result in results])

    if args.output_file:
        cm.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    # else:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         print(cm)

    session.close()

    return cm


def plot_correlation(db_file, axis1, axis2, color, id=None, exp_name=None):
    if id is None and exp_name is None:
        raise ValueError(f'Both id and exp-name attributes cannot be empty')

    session = local_create_session(db_file)

    individual_evaluations = []
    if id:
        try:
            results: List[Result] = session.query(Result).filter(Result.id.in_(id)).all()
            if len(results) == 0:
                raise ValueError(f'No results found with name {exp_name}')

            if len(results) == 1:
                individual_evaluations = results[0].individual_evaluations
            else:
                individual_evaluations = pd.concat([result.individual_evaluations for result in results])

        except OperationalError:
            print(f'No results found with id {id}')
    else:
        try:
            results: List[Result] = session.query(Result).filter(Result.name == exp_name).all()
            if len(results) == 0:
                raise ValueError(f'No results found with name {exp_name}')

            individual_evaluations = pd.concat([result.individual_evaluations for result in results])

        except OperationalError:
            print(f'No results found with name {exp_name}')

    df = individual_evaluations

    if type(df) == str:
        return

    df.sort_values(by='generation', ascending=True)
    # df = df.sample(frac=1 / len(results))

    plt.figure()
    x = df[axis1].values
    y = df[axis2].values
    c = df[color].values
    points = plt.scatter(x, y, c=c, s=1, cmap='viridis', alpha=0.5)
    plt.colorbar(points, label=color)
    sns.regplot(axis1, axis2, data=df, scatter=False, x_jitter=0.005, y_jitter=0.005, order=1, robust=False)

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

    _, ax = plt.subplots()

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


def plot_logs(args):
    las_files = glob.glob(f'{args.logs_folder}/*.las')
    las_files = list(map(lasio.read, las_files))

    for las in las_files:
        if len(las.curves) > 2:
            las.df().plot()

    plt.show()


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


def find_useful_features(args):
    df = pd.read_csv(args.input_file, index_col=0)
    del df['petrofacie']

    full_std = df.std()

    groups_std = {}
    for key, group in df.groupby('predicted labels'):
        groups_std[key] = full_std / group.std()
        for column in group:
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
    # binarized_grain_size = binaryze_column(df['Main/single size mode(mm):'])
    # df = pd.concat([df, binarized_grain_size], axis=1)

    session = local_create_session(args.db_file)

    columns = []
    # try:
    result = session.query(Result).filter(Result.id == args.id[0]).first()
    columns = [selected_feature.column for selected_feature in result.selected_features]
    labels = [int(result_label.label) for result_label in result.result_labels]
    # except AttributeError:
    #     raise ValueError(f'Result {str(args.id)} not found in {args.db_file}')

    session.close()

    df = df.filter(items=columns + ['petrofacie'])
    df['Cluster'] = pd.Series(labels, index=df.index)
    labels = class_cluster_match(df['petrofacie'].values, labels)
    df['Cluster label'] = pd.Series(labels, index=df.index)

    if args.output_file:
        df.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(df)


def select_best(args):
    session = local_create_session(args.db_file)

    results: List[Result] = session.query(Result).filter(Result.name == args.exp_name).all()

    df = [[result.id, result.accuracy, result.f_measure, result.silhouette] for result in results]
    df = pd.DataFrame(df, columns=['id', 'accuracy', 'f_measure', 'silhouette'])

    best_accuracy = df.iloc[df['accuracy'].idxmax()]
    print(f'''Best Accuracy: {best_accuracy}''')

    best_f_measure = df.iloc[df['f_measure'].idxmax()]
    print(f'''Best F-Measure: {best_f_measure}''')

    best_silhouette = df.iloc[df['silhouette'].idxmax()]
    print(f'''Best Silhouette: {best_silhouette}''')

    session.close()


def export_results(args):
    databases = glob.glob(args.db_file)

    dbs = []
    datasets = []
    scenarios = []
    args_idx = []
    metrics = []
    for db in databases:
        print(f'Processing {db}...')
        session = local_create_session(db)

        query_results: List[Result] = session.query(Result).all()

        key_fnc_i = lambda x: re.search('.*experiment_name\: ([a-z|A-Z|_]+).*', str([str(arg) for arg in x.args])).group(1)
        query_results = sorted(query_results, key=key_fnc_i)
        group_results = groupby(query_results, key_fnc_i)

        names = []
        for experiment_name, group_i in group_results:
            print(f'Experiment name: {experiment_name}')
            names += [experiment_name]
            results_i = list(group_i)

            key_fnc_j = lambda x: re.search('.*scenario\: \[\[(.*?)\]\].*', str([str(arg) for arg in x.args])).group(1)
            results_i = sorted(results_i, key=key_fnc_j)
            group_results_j = groupby(results_i, key_fnc_j)

            for scenario, group_j in group_results_j:
                print(f'Scenario: {scenario}')
                results_j = list(group_j)

                key_fnc_k = lambda x: str([str(arg) for arg in x.args]).replace(experiment_name, '').replace(scenario, '')
                results_k = sorted(results_j, key=key_fnc_k)
                group_results_k = groupby(results_k, key_fnc_k)

                for args_k, group_k in group_results_k:
                    dbs += [db]
                    datasets += [experiment_name]
                    scenarios += [scenario]
                    args_idx += [args_k]

                    accuracies_avg = np.average([result.accuracy for result in results_j])
                    accuracies_std = np.std([result.accuracy for result in results_j])
                    ari_avg = np.average([result.adjusted_rand_score for result in results_j])
                    ari_std = np.std([result.adjusted_rand_score for result in results_j])
                    f_measure_avg = np.average([result.f_measure for result in results_j])
                    f_measure_std = np.std([result.f_measure for result in results_j])

                    metrics += [[accuracies_avg, accuracies_std,
                                 ari_avg, ari_std,
                                 f_measure_avg, f_measure_std]]
        session.close()

        df_index = pd.MultiIndex.from_arrays([dbs, datasets, scenarios, args_idx],
                                             names=['database', 'dataset', 'scenario', 'args'])
        df_columns = [['Accuracy'] * 2 + ['ARI'] * 2 + ['F-Measure'] * 2, ['average', 'std'] * 3]
        df_columns = pd.MultiIndex.from_arrays(df_columns)

        df = pd.DataFrame(data=metrics, index=df_index, columns=df_columns)
        df.sort_index(level=1, inplace=True)

        df.to_excel(args.output_file)


def main(args=None):
    args = argument_parser(args)

    result = None

    if args.plot_correlation:
        plot_correlation(args.db_file, args.axis1, args.axis2, args.color, id=args.id, exp_name=args.exp_name)
    elif args.correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        correlation_calculation(df, args)
    elif args.merge_results:
        merge_results(args)
    elif args.useful_features:
        find_useful_features(args)
    elif args.melt_results:
        melt_results(args)
    elif args.average_feature_selection:
        average_feature_selection(args)
    elif args.list_results:
        show_results(args)
    elif args.detail_result:
        show_result(args)
    elif args.confusion_matrix:
        result = confusion_matrix(args)
    elif args.filter:
        filter_dataset(args)
    elif args.select_best:
        select_best(args)
    elif args.plot_logs:
        plot_logs(args)
    elif args.export_results:
        export_results(args)

    return result


if __name__ == '__main__':
    main()
