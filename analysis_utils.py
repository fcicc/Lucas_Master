import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import random

def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Plot data with seaborn library''')
    parser.add_argument('input_file', type=str, help='''input CSV file''')

    args = parser.parse_args()

    return args

def main():
    args = argument_parser()

    df = pd.read_csv(args.input_file, index_col=0)
    df = df.applymap(lambda x: x+((random.random()-0.5)/500)) # jitter

    plt.figure()
    # points = plt.scatter(df['wemmert_gancarski'],
    #                     df['adjusted_rand_score'],
    #                     c=df.index,
    #                     s=3, cmap='viridis', alpha=0.7)
    # plt.colorbar(points, label='index')
    # sns.regplot("wemmert_gancarski", "adjusted_rand_score", data=df, scatter=False)
    g = sns.PairGrid(df)
    g.map(plt.regplot)
    plt.show()
    # plt.savefig(
    #     os.path.join(
    #         input_dir,
    #         'dataset_analysis' +
    #         start_time +
    #         '_objective_space.png'),
    #     format='png', dpi=900)


if __name__ == '__main__':
    main()
    