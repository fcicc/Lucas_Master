import os
from itertools import chain

from package.main import run


def main():
    run_multiple = 1
    db_file = 'affinityProp.db'

    affinity_preferences = {
        "('raw',)": {
            'campus_basin': -650,
            'equatorial_margin': -400,
            'talara_basin': -400,
            'carmopolisGrouped': -540,
            'jequitinhonha': -470,
            'mucuri': -5300
        },
        "('compositional_groups', 'localizational_groups')": {
            'campus_basin': -800,
            'equatorial_margin': -400,
            'talara_basin': -400,
            'carmopolisGrouped': -540,
            'jequitinhonha': -470,
            'mucuri': -5300
        }
    }

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
        '--cluster-algorithm': 'affinity-propagation',
    }

    # args = {
    #     '--strategy': 'none',
    #     '--db-file': db_file,
    #     '--cluster-algorithm': 'kmeans'
    # }

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
        print(scenario)
        local_args = args
        local_args['--scenario'] = list(scenario)
        fitness_metric = 'silhouette_sklearn'
        local_args['--fitness-metric'] = fitness_metric
        for experiment_name, dataset_file in dataset_locations.items():
            print(f'RUNNING {experiment_name} in scenario {str(scenario)}')

            if local_args['--cluster-algorithm'] == 'affinity-propagation':
                print(affinity_preferences[str(scenario)][experiment_name])
                local_args['--preference'] = str(affinity_preferences[str(scenario)][experiment_name])

            for i in range(run_multiple):
                print(f'{i+1}/{run_multiple}')

                list_local_args = list(chain.from_iterable(local_args.items()))
                run(args=list_local_args + [dataset_file, experiment_name])


if __name__ == '__main__':
    main()
