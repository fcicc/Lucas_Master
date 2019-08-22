from enum import Enum

from package.main import run


class e_scenarios(Enum):
    RAW = ('raw',)
    COMPOSITIONAL_LOCALIZATIONAL = ('compositional_groups', 'localizational_groups')


class e_datasets(Enum):
    CAMPUS_BASIN = './datasets/CampusBasin/subtotals_dataset.xlsx'
    EQUATORIAL_MARGIN = './datasets/MargemEquatorial/subtotals_dataset.xlsx'
    TALARA_BASIN = './datasets/TalaraBasin/subtotals_dataset.xlsx'
    CARMOPOLIS_GROUPED = './datasets/CarmopolisGrouped/subtotals_dataset.xlsx'
    JEQUITINHONHA = './datasets/Jequitinhonha/subtotals_dataset.xlsx'
    MUCURI = './datasets/Mucuri/subtotals_dataset.xlsx'
    # 'carmopolis': DATASETS_FOLDER + '/Carmopolis/subtotals_dataset.xlsx'


CLUSTERING_ALGORITHMS = ['agglomerative', 'kmeans', 'affinity-propagation']

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
    for algorithm in CLUSTERING_ALGORITHMS:
        for dataset in e_datasets:
            input_args = [
                dataset.value,
                '1',

            ]
            print(f'Algorithm: {algorithm}')
            print(f'Dataset: {dataset.value}')
    # 2 Experimentos avaliando as métricas internas de qualidade dos clusters
    # 3 Experimentos avaliando com outras abordagens de seleção de features (PCA,...)
    # 4 Experimento mostrando o limite teórico do coeficiente de silhuete para os datasets selecionados
    # 5 Experimento rodando a abordagem e comparando o coeficiente de silhueta com os limite teórico e as métricas externas
    # 6 Experimentos mostrando que com reengenharia baseada em ontologia os resultados se aproximam da abordagem sem reengeharia
    # 7 Experimento mostrando o percentual de features desejáveis (de acordo com o expert) mantidas na seleção (talvez ao longo das gerações)
    # 8 Experimentos mostrando que com reengenharia de features a abordagem roda mais rápido
    # 9 Experimentos mostrando que com reengenharia de features a abordagem roda mais rápido
    # 10 Qualitative Analysis
    pass
