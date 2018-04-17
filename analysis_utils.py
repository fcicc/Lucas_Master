import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import argparse
import random


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

    args = parser.parse_args()

    if (args.plot_correlation and args.correlation) or not(args.plot_correlation or args.correlation):
        raise ValueError("Cannot have this combination of arguments.")

    return args


def plot_correlation(df, args):
    plt.figure()
    points = plt.scatter(df[args.axis1],
                        df[args.axis2],
                        c=df.index.values,
                        s=3, cmap='viridis', alpha=0.7)
    plt.colorbar(points, label='index')
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


def main():
    args = argument_parser()

    df = pd.read_csv(args.input_file, index_col=0)

    if args.plot_correlation:
        plot_correlation(df, args)
    elif args.correlation:
        correlation_calculation(df, args)

if __name__ == '__main__':
    main()
