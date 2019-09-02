from enum import Enum

from package.evaluation_functions import CLUSTER_CRIT_ALLOWED_FITNESSES
from package.main import run, e_scenarios


class e_datasets(Enum):
    CAMPUS_BASIN = './datasets/CampusBasin/subtotals_dataset.xlsx'
    EQUATORIAL_MARGIN = './datasets/MargemEquatorial/subtotals_dataset.xlsx'
    TALARA_BASIN = './datasets/TalaraBasin/subtotals_dataset.xlsx'
    CARMOPOLIS_GROUPED = './datasets/CarmopolisGrouped/subtotals_dataset.xlsx'
    JEQUITINHONHA = './datasets/Jequitinhonha/subtotals_dataset.xlsx'
    MUCURI = './datasets/Mucuri/subtotals_dataset.xlsx'
    # 'carmopolis': DATASETS_FOLDER + '/Carmopolis/subtotals_dataset.xlsx'


class e_meta_clustering_algorithms(Enum):
    GA = 'ga'
    PSO = 'pso'
    WARD_P = 'ward_p'
    RANDOM_GA = 'random_ga'
    NONE = 'none'


class e_clustering_algorithms(Enum):
    AGGLOMERATIVE = 'agglomerative'
    KMEANS = 'kmeans'
    AFFINITY_PROPAGATION = 'affinity-propagation'

PREFERENCES = {
    e_scenarios.RAW: {
        e_datasets.CAMPUS_BASIN: -650,
        e_datasets.EQUATORIAL_MARGIN: -400,
        e_datasets.TALARA_BASIN: -400,
        e_datasets.CARMOPOLIS_GROUPED: -540,
        e_datasets.JEQUITINHONHA: -470,
        e_datasets.MUCURI: -5300
    },
    e_scenarios.COMPOSITIONAL_LOCALIZATIONAL: {
        e_datasets.CAMPUS_BASIN: -1550,
        e_datasets.EQUATORIAL_MARGIN: -900,
        e_datasets.TALARA_BASIN: -750,
        e_datasets.CARMOPOLIS_GROUPED: -1590,
        e_datasets.JEQUITINHONHA: -900,
        e_datasets.MUCURI: -7700
    }
}


def run_experiment(args):
    run(args=args)


if __name__ == '__main__':
    # 1 - Experimentos avaliando os algoritmos de clustering
    # database = 'results_1.db'
    # for algorithm in e_clustering_algorithms:
    #     for dataset in e_datasets:
    #         for scenario in e_scenarios:
    #             input_args = [
    #                 dataset.value,
    #                 '1',
    #                 '--level', 'features_groups',
    #                 # '--num-gen', '0',
    #                 # '--pop-size', '0',
    #                 # '--perfect',
    #                 '--eval-rate', '1',
    #                 '--min-features', '50',
    #                 '--fitness-metric', 'silhouette_sklearn',
    #                 '--cluster-algorithm', f'{algorithm.value}',
    #                 '--db-file', f'{database}',
    #                 '--strategy', 'none',
    #                 # '--p_ward', '0',
    #                 '--preference', str(PREFERENCES[scenario][dataset]),
    #                 f'--scenario', scenario.name
    #             ]
    #             run_experiment(input_args)
    # 2 Experimentos avaliando as métricas internas de qualidade dos clusters
    database = 'results_2.db'
    for metric, _ in CLUSTER_CRIT_ALLOWED_FITNESSES:
        for dataset in e_datasets:
            for scenario in e_scenarios:
                input_args = [
                    dataset.value,
                    '2',
                    '--level', 'features_groups',
                    '--num-gen', '200',
                    '--pop-size', '128',
                    # '--perfect',
                    '--eval-rate', '0.2',
                    '--min-features', '2',
                    f'--fitness-metric', f'{metric}',
                    f'--cluster-algorithm', 'agglomerative',
                    f'--db-file', f'{database}',
                    '--strategy', 'ga',
                    # '--p_ward', '0',
                    # '--preference', '0',
                    f'--scenario', f'{scenario.value}',
                ]
                run_experiment(input_args)
    # # 3 Experimentos avaliando com outras abordagens de seleção de features (PCA,...)
    # database = 'results_3.db'
    # for metric in ALLOWED_METRICS:
    #     for dataset in e_datasets:
    #         for scenario in e_scenarios:
    #             input_args = [
    #                 dataset.value,
    #                 '3',
    #                 '--level features_group',
    #                 '--num-gen 200',
    #                 '--pop-size 128',
    #                 # '--perfect',
    #                 '--eval-rate 0.2',
    #                 '--min-features 2',
    #                 '--fitness-metric silhouette_sklearn',
    #                 f'--cluster_algorithm {algorithm}',
    #                 f'--db-file {database}',
    #                 '--strategy none',
    #                 # '--p_ward 0',
    #                 # '--preference 0',
    #                 f'--scenario {scenario.value}',
    #             ]
    #             run_experiment(input_args)
    # 4 Experimento mostrando o limite teórico do coeficiente de silhuete para os datasets selecionados
    # 5 Experimento rodando a abordagem e comparando o coeficiente de silhueta com os limite teórico e as métricas externas
    # 6 Experimentos mostrando que com reengenharia baseada em ontologia os resultados se aproximam da abordagem sem reengeharia
    # 7 Experimento mostrando o percentual de features desejáveis (de acordo com o expert) mantidas na seleção (talvez ao longo das gerações)
    # 8 Experimentos mostrando que com reengenharia de features a abordagem roda mais rápido
    # 9 Experimentos mostrando que com reengenharia de features a abordagem roda mais rápido
    # 10 Qualitative Analysis
    pass
