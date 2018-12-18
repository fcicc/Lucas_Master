from functools import partial
from glob import glob
from itertools import combinations
from multiprocessing.pool import Pool
from os.path import dirname, sep
from random import uniform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


def run():
    data_paths = glob('datasets/*/subtotals_dataset.xlsx')

    scenarios = ['raw', 'compositional_groups', 'localizational_groups']
    scenarios = sum([list(combinations(scenarios, i+1)) for i, _ in enumerate(scenarios)], [])

    columns = pd.MultiIndex.from_tuples([(path, scenario) for scenario in scenarios for path in data_paths])
    As_df = pd.DataFrame(columns=columns)

    pool = Pool()

    for path in data_paths:
        print(f'Running: {path}')
        for scenario in scenarios:
            print(f'\tScenario: {scenario}')

            df = pd.read_excel(path, index_row=[0, 1, 2], header=[0, 1, 2])
            df = df[list(scenario) + ['others']]
            df = df.groupby(level=['features_groups'], axis=1).sum()

            del df['petrofacie']
            As = hopkins_statistic(df.values, scenario, path, pool)
            print(f'\t\t{path} - {np.mean(As)}')
            As_df[path, scenario] = As

    As_df.sort_index(axis=1, inplace=True)
    As_df.to_excel('cluster_tendency.xlsx')

    pool.close()


def hopkins_statistic(X, scenario, path, pool):
        """ As described in
        @article{Hopkins1954,
        author = {Hopkins, Brian and J., G. Skellam},
        journal = {Annals of Botany},
        number = {70},
        pages = {213--227},
        title = {{A New Method for determining the Type of Distribution of Plant Individuals}},
        url = {https://www.jstor.org/stable/pdf/42907238.pdf},
        volume = {18},
        year = {1954}
        }
        """
        As = multiple_hopkins_statistic(X, pool)

        plt.figure()
        sns.distplot(As).set_title(str(scenario))
        plt.tight_layout()
        plt.savefig(f'{dirname(path)}{sep}hopkins cluster tendency {str(scenario)}.pdf')

        return As


def multiple_hopkins_statistic(X, pool):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    I = [dist for dist in distances[:, 1]]
    min_max = [(min(column), max(column)) for column in X.T]
    As = pool.map(partial(calculate_A, I, X, min_max, nbrs, n), range(1024))
    return As


def calculate_A(I, X, min_max, nbrs, n, _):
    random_points = np.asarray([[uniform(minimum, maximum) for minimum, maximum in min_max] for _ in range(n)])
    distances, indices = nbrs.kneighbors(random_points)

    P = [dist for dist in distances[:, 0]]

    A = sum([p ** 2 for p in P]) / sum([i ** 2 for i in I])

    return A


if __name__ == '__main__':
    run()
