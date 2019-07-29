import os
from itertools import chain

from package.main import run


def main():
    run_multiple = 200
    db_file = 'kmeans.db'
    # args = {
    #     '-e': '0.1',
    #     '--num-gen': '2',
    #     '--pop-size': '256',
    #     '--min-features': '2',
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'agglomerative'
    # }

    args = {
        '--strategy': 'none',
        '--db-file': db_file,
        '--cluster-algorithm': 'kmeans'
    }

    datasets_folder = './datasets/'
    dataset_locations = {
        'campus_basin': datasets_folder + '/CampusBasin/subtotals_dataset.xlsx',
        'equatorial_margin': datasets_folder + '/MargemEquatorial/subtotals_dataset.xlsx',
        'talara_basin': datasets_folder + '/TalaraBasin/subtotals_dataset.xlsx',
        # 'carmopolis':        datasets_folder + '/Carmopolis/subtotals_dataset.xlsx',
        'carmopolisGrouped': datasets_folder + '/CarmopolisGrouped/subtotals_dataset.xlsx',
        'jequitinhonha': datasets_folder + '/Jequitinhonha/subtotals_dataset.xlsx',
        'mucuri': datasets_folder + '/Mucuri/subtotals_dataset.xlsx'
    }

    for name, dataset_file in dataset_locations.items():
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(f'{dataset_file} not found for {name}')

    for scenario in [('raw',), ('compositional_groups', 'localizational_groups')]:
        local_args = args
        local_args['--scenario'] = list(scenario)
        fitness_metric = 'silhouette_sklearn'
        local_args['--fitness-metric'] = fitness_metric
        for experiment_name, dataset_file in dataset_locations.items():
            print(f'RUNNING {experiment_name} in scenario {str(scenario)}')

            for i in range(run_multiple):
                print(f'{i+1}/{run_multiple}')

                list_local_args = list(chain.from_iterable(local_args.items()))
                run(args=list_local_args + [dataset_file, experiment_name])


if __name__ == '__main__':
    main()
