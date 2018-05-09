import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse
import random
import glob
import re
import sys
from io import StringIO
from functools import partial

from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils.multiclass import unique_labels


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Analysis over input data''')
    parser.add_argument('input_file', type=str, help='''input CSV file''')
    parser.add_argument('-o', '--output_file', type=str, help='''output file''')
    parser.add_argument('-p', '--plot-correlation', action='store_true', help='plot correlation')
    parser.add_argument('--axis1', type=str, help='''first plot axis''')
    parser.add_argument('--axis2', type=str, help='''second plot axis''')
    parser.add_argument('--color', type=str, help='''point color''', default='index')
    parser.add_argument('-c', '--correlation', action='store_true', help='generates correlation matrix')
    parser.add_argument('-f', '--feature-analysis', action='store_true', help='')
    parser.add_argument('-m', '--merge-results', action='store_true', help='')
    parser.add_argument('-a', '--merge-feature-selection', action='store_true', help='')
    parser.add_argument('-n', '--n-last-results', type=int, help='', default=10)
    parser.add_argument('-t', '--petrel', action='store_true', help='')
    parser.add_argument('-u', '--useful-features', action='store_true', help='')
    parser.add_argument('-b', '--clear-incomplete-outputs', action='store_true', help='')

    args = parser.parse_args()

    if sum([args.plot_correlation, args.correlation,
            args.feature_analysis, args.merge_results,
            args.petrel, args.useful_features, args.merge_feature_selection,
            args.clear_incomplete_outputs]) != 1:
        raise ValueError("Cannot have this combination of arguments.")

    return args


def plot_correlation(df, args):
    plt.figure()
    if args.color == 'index':
        points = plt.scatter(df[args.axis1],
                            df[args.axis2],
                            c=df.index.values,
                            s=3, cmap='viridis', alpha=0.7)
        plt.colorbar(points, label='index')
    else:
        points = plt.scatter(df[args.axis1],
                            df[args.axis2],
                            c=df[args.color],
                            s=3, cmap='viridis', alpha=0.7)
        plt.colorbar(points, label=args.color)
        
    sns.regplot(args.axis1, args.axis2, data=df, scatter=False, x_jitter=0.05, y_jitter=0.05, order=1, robust=False)

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
    summary_files = glob.glob(args.input_file + '/dataset_analysis*.txt')
    summary_files.sort(key=os.path.getmtime, reverse=True)
    print('Found ' + str(len(summary_files)) + ' files')
    summary_files = summary_files[:min(len(summary_files), args.n_last_results)]
    print('Processing ' + str(len(summary_files)) + ' files:')

    all_results = {'Accuracy': [],
                   'Adjusted Rand score': [],
                   'F-Measure': [],
                   'Silhouette': [],
                   'Complexity reduction rate': []}
    for i, file in enumerate(summary_files):
        file_buf = open(file)
        lines = file_buf.readlines()
        metric = re.search('fitness_metric=\'(.*?)\'', lines[0]).group(1)
        print('\t' + str(i + 1) + ': ' + file + ' - ' + metric)
        interest_section = ''.join(lines[2:15])
        all_results['Accuracy'].append(
            float(re.search('accuracy score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        all_results['Adjusted Rand score'].append(
            float(re.search('adjusted Rand score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        all_results['F-Measure'].append(
            float(re.search('f1 score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        all_results['Silhouette'].append(
            float(re.search('silhouette score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        all_results['Complexity reduction rate'].append(
            float(re.search('feature reduction rate\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))

    all_results = pd.DataFrame.from_dict(all_results)

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


def petrofacies_to_petrel(df, args):
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
            if well_i != well: continue
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


def find_useful_features(df, args):
    description = df.groupby('petrofacie').var()

    if args.output_file:
        description.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(description)


def merge_feature_selection(args):
    summary_files = glob.glob(args.input_file + '/dataset_analysis*_selection_rate.csv')
    summary_files.sort(key=os.path.getmtime, reverse=True)
    print('Found ' + str(len(summary_files)) + ' files')
    summary_files = summary_files[:min(len(summary_files), args.n_last_results)]
    print('Processing ' + str(len(summary_files)) + ' files:')

    dfs = list(map(partial(pd.read_csv, index_col=0), summary_files))

    results_df = pd.concat(list(map(lambda df: df[['199']], dfs)), axis=1)
    results_df.columns = ['trial ' + str(i) for i in range(results_df.shape[1])]
    results_df['average'] = results_df.mean(axis=1)
    results_df = results_df.sort_values(by='average', ascending=False)

    if args.output_file:
        results_df.to_csv(args.output_file, quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(results_df)
        

def clear_incomplete_experiments(args):
    """Search the input directory for incomplete run files and erase them."""
    results_regex = os.path.join(args.input_file,
                    "dataset_analysis{0},{1}_{2}_{3}-{4}_{5}_{6}.txt".format(('[0-9a-fA-F]' * 7),('[0-9]' * 4), (
                                        '[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2)))
    summary_files = glob.glob(results_regex)

    for summary_file in summary_files:
        summary_file_name = os.path.basename(summary_file)
        file_id = re.search("dataset_analysis({0},{1}_{2}_{3}-{4}_{5}_{6}).txt".format(('[0-9a-fA-F]' * 7),('[0-9]' * 4), (
                                        '[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2), ('[0-9]' * 2)),
                                        summary_file_name).group(1)
        related_files = glob.glob(args.input_file + '/*{0}*'.format(file_id))
        if len(related_files) < 4:
            print('Erasing files from execution {0}'.format(file_id))
            for file_path in related_files:
                os.remove(file_path)

    print('Done!')


def main():
    args = argument_parser()

    if args.plot_correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        plot_correlation(df, args)
    elif args.correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        correlation_calculation(df, args)
    elif args.merge_results:
        merge_results(args)
    elif args.useful_features:
        df = pd.read_csv(args.input_file, index_col=0)
        find_useful_features(df, args)
    elif args.petrel:
        df = pd.read_csv(args.input_file, index_col=0)
        petrofacies_to_petrel(df, args)
    elif args.merge_feature_selection:
        merge_feature_selection(args)
    elif args.clear_incomplete_outputs:
        clear_incomplete_experiments(args)


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


if __name__ == '__main__':
    main()
