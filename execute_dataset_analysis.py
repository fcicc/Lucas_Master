import os
from itertools import chain

from package.main import run


def main():
    run_multiple = 1
    db_file = 'results.db'

    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '200',
    #     '--pop-size': '600',
    #     '--min-features': '2',
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'copac',
    #     '--k_neighbors': '60',
    #     #'--eps': '30.0',
    #     '--eps': '12.0',
    #     '--min_samples': '1'
    # }

    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '200',
    #     '--pop-size': '600',
    #     '--min-features': '2',
    #     '--strategy': 'ga',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'agglomerative'
    # }

    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '200',
    #     '--pop-size': '600',
    #     '--min-features': '2',
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'agglomerative'
    # }

    # approach #1 - k-means
    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '200',
    #     '--pop-size': '600',
    #     '--min-features': '2',
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'kmeans'
    # }

    # approach #2 - Hierarchical
    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '200',
    #     '--pop-size': '600',
    #     '--min-features': '2',
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'agglomerative'
    # }

    # approach #3 - Hierarchical + Genetic
    args = {
        '-e': '0.1',
        '--num-gen': '200',
        '--pop-size': '600',
        '--min-features': '2',
        '--strategy': 'ga',
        '--db-file': db_file,
        '--cluster-algorithm': 'agglomerative'
    }

    datasets_folder = './datasets/'
    dataset_locations = {
        'campus_basin': datasets_folder + '/CampusBasin/dataset.csv',
        'equatorial_margin': datasets_folder + '/MargemEquatorial/dataset.csv',
        # 'talara_basin': datasets_folder + '/TalaraBasin/subtotals_dataset.xlsx',
        # 'carmopolis':        datasets_folder + '/Carmopolis/subtotals_dataset.xlsx',
        # 'carmopolisGrouped': datasets_folder + '/CarmopolisGrouped/subtotals_dataset.xlsx',
        # 'jequitinhonha': datasets_folder + '/Jequitinhonha/subtotals_dataset.xlsx',
        # 'mucuri': datasets_folder + '/Mucuri/subtotals_dataset.xlsx'
    }

    for name, dataset_file in dataset_locations.items():
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(f'{dataset_file} not found for {name}')

    local_args = args
    fitness_metric = 'silhouette_sklearn'
    local_args['--fitness-metric'] = fitness_metric
    for experiment_name, dataset_file in dataset_locations.items():
        print(f'RUNNING {experiment_name}')

        for i in range(run_multiple):
            print(f'{i+1}/{run_multiple}')

            list_local_args = list(chain.from_iterable(local_args.items()))
            run(args=list_local_args + [dataset_file, experiment_name])


if __name__ == '__main__':
    main()
