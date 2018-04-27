import pandas as pd
import seaborn as sns
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


from collections import Counter

def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Analysis over input data''')
    parser.add_argument('input_file', type=str, help='''input CSV file''')
    parser.add_argument('-o','--output_file', type=str, help='''output file''')
    parser.add_argument('-p', '--plot-correlation', action='store_true', help='plot correlation')
    parser.add_argument('--axis1', type=str, help='''first plot axis''')
    parser.add_argument('--axis2', type=str, help='''second plot axis''')
    parser.add_argument('-c', '--correlation', action='store_true', help='generates correlation matrix')
    parser.add_argument('-f', '--feature-analysis', action='store_true', help='')
    parser.add_argument('-m', '--merge-results', action='store_true', help='')
    parser.add_argument('-n', '--n-last-results', type=int, help='', default=10)
    parser.add_argument('-t', '--petrel', action='store_true', help='')
    parser.add_argument('-u', '--useful-features', action='store_true', help='')

    args = parser.parse_args()

    if sum([args.plot_correlation, args.correlation,
            args.feature_analysis, args.merge_results,
            args.petrel, args.useful_features]) != 1:
        raise ValueError("Cannot have this combination of arguments.")

    return args


def plot_correlation(df, args):
    plt.figure()
    points = plt.scatter(df[args.axis1],
                        df[args.axis2],
                        c=df['f1_score'],
                        s=3, cmap='viridis', alpha=0.7)
    plt.colorbar(points, label='f1_score')
    sns.regplot(args.axis1, args.axis2, data=df, scatter=False, x_jitter=0.05, y_jitter=0.05, order=1, robust=False)

    if args.output_file:
        plt.savefig(args.output_file, dpi=600)
    else:
        plt.show()

def correlation_calculation(df, args):
    correlation = df.corr()

    if args.output_file:
        correlation.to_csv(args.output_file,  quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(correlation)


def feature_selection_frequency(args):
    filtered_files = glob.glob(args.input_file+'/*_filtered_dataset.csv')
    filtered_files.sort(key=os.path.getmtime, reverse=True)
    filtered_files = filtered_files[:min(len(filtered_files),args.n_last_results)]
    selected_features = []
    for filtered_file in filtered_files:
        df = pd.read_csv(filtered_file, index_col=0)
        selected_features.append(df.columns.values)
    df = pd.DataFrame(selected_features)

    cnt = Counter(df.values.flatten())
    del cnt['petrofacie']
    df = pd.DataFrame.from_dict(cnt, orient='index')
    df = df.sort_values(by=0)
    
    if args.output_file:
        df.to_csv(args.output_file,  quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(df)


def merge_results(args):
    summary_files = glob.glob(args.input_file+'/dataset_analysis*.txt')
    summary_files.sort(key=os.path.getmtime, reverse=True)
    print('Found ' + str(len(summary_files)) + ' files')
    summary_files = summary_files[:min(len(summary_files),args.n_last_results)]
    print('Processing ' + str(len(summary_files)) + ' files:')
    all_results = {}
    for i, file in enumerate(summary_files):
        file_buf = open(file)
        lines = file_buf.readlines()
        metric = re.search('fitness_metric=\'(.*?)\'', lines[0]).group(1)
        print('\t' + str(i+1) + ': ' + file + ' - ' + metric)
        if metric in all_results:
            print('\t\tRepeated metric - BREAK')
            break
        values = []
        interest_section = ''.join(lines[2:15])
        values.append(float(re.search('accuracy score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        values.append(float(re.search('adjusted Rand score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        values.append(float(re.search('f1 score\s+([-+]?[0-9]*\.?[0-9]*)', interest_section).group(1)))
        all_results[metric] = values
        file_buf.close()

    all_results = pd.DataFrame.from_dict(all_results)
    all_results.index = ['accuracy', 'adjusted Rand score', 'f1 score']

    if args.output_file:
        all_results.to_csv(args.output_file,  quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(all_results)


def petrofacies_to_petrel(df, args):
    thin_section_names = df.index.values
    wells = map(partial(re.search, '([\w|-]+) [-+]?[0-9]*\.?[0-9]*'), thin_section_names)
    wells = list(map(lambda x: x.group(1), wells))
    depths = map(partial(re.search, ' ([-+]?[0-9]*\.?[0-9]*)'), thin_section_names)
    depths = map(lambda x: x.group(1), depths)
    depths = list(map(float, depths))
    petrofacies = df['petrofacie'].values

    epsilon = 0.009
    
    for well in set(wells):
        a1 = []
        a2 = []
        for depth, thin_section_name, petrofacie, well_i in zip(depths, thin_section_names, petrofacies, wells):
            if well_i != well: continue
            a1.append([str(depth), str(depth+epsilon), '"'+thin_section_name+'"'])
            a2.append([str(depth), str(depth+epsilon), '"'+petrofacie+'"'])
        file1 = open(well+'_1.txt', 'w')
        file1.write('WELL: "'+ well +'"\n')
        file2 = open(well+'_2.txt', 'w')
        file2.write('WELL: "'+ well +'"\n')
        for line1, line2 in zip(a1, a2):
            file1.write('\t'.join(line1)+'\n')
            file2.write('\t'.join(line2)+'\n')
        file1.close()
        file2.close()


def find_useful_features(df, args):
    description = df.groupby('petrofacie').var()
    
    if args.output_file:
        description.to_csv(args.output_file,  quoting=csv.QUOTE_NONNUMERIC, float_format='%.10f', index=True)
    else:
        print(description)


def main():
    args = argument_parser()

    if args.plot_correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        plot_correlation(df, args)
    elif args.correlation:
        df = pd.read_csv(args.input_file, index_col=0)
        correlation_calculation(df, args)
    elif args.feature_analysis:
        feature_selection_frequency(args)
        merge_results(args)
    elif args.useful_features:
        df = pd.read_csv(args.input_file, index_col=0)
        find_useful_features(df, args)
    elif args.petrel:
        df = pd.read_csv(args.input_file, index_col=0)
        petrofacies_to_petrel(df, args)


if __name__ == '__main__':
    main()
