import re
from copy import deepcopy
from functools import partial
from os.path import join, dirname
from glob import glob

import pandas as pd
from owlready2 import get_ontology

MACRO_LOCATIONS = ['interstitial', 'framework', 'framework and interstitial']


def calculate_subtotals(target_path, idiom):
    dataset = pd.read_excel(target_path, index_row=[0, 1, 2], header=[0, 1, 2]).groupby(level=['features'], axis=1).sum()

    petr_ont = get_ontology('petroledge_model.owl').load()

    feature_names = list(map(lambda x: x.lower(), dataset.columns.values))
    feature_names = [re.sub('(\[.*\])', '', feature_name) for feature_name in feature_names]

    dataset.columns = feature_names

    others_names = list(set(feature_names) & {'grain_size', 'petrofacie', 'phi stdev sorting', 'sorting'})

    others = dataset[others_names]
    if 'sorting' in others_names:
        del others['sorting']
    dataset = dataset.drop(others_names, axis=1)

    compositional_type = [partial(extract_mineral_group, petr_ont)(column) for column in dataset.columns]

    locational_groups = [extract_locational_group(petr_ont, column) for column in dataset.columns]

    main_groups = ['compositional_groups' for _ in dataset.columns]
    main_groups += ['localizational_groups' for _ in dataset.columns]

    dataset = pd.concat([dataset, dataset], axis=1)
    dataset.columns = pd.MultiIndex.from_tuples(zip(main_groups, compositional_type+locational_groups, dataset.columns.values))
    dataset.columns.names = ['groups_types', 'features_groups', 'features']

    others.columns = pd.MultiIndex.from_tuples(zip(others.columns.values, others.columns.values, others.columns.values))
    dataset = pd.concat([dataset, others], axis=1)

    dataset.index.name = 'sample'
    dataset = dataset.sort_index(axis=1)

    if any(dataset.isna().any().values):
        print(dataset.isna().any())
        raise ValueError('There should not be any NaN values inside the subtotals data frame!')

        dataset['petrofacie'] = dataset['petrofacie']
    return dataset


def extract_compositional_type(column):
    if column.count(' - ') == 2:
        return 'primary'
    elif column.count(' - ') == 6:
        return 'diagenetic'
    elif column.count(' - ') == 5:
        return 'porosity'
    else:
        return column


def extract_mineral_group(ontology, column):
    if column.count(' - ') < 2:
        return column

    constituent = column.split(' - ')[0].replace(' ', '_')

    return ontology[constituent].is_a[0].name


def extract_locational_group(ontology, column):
    attributes = column.split(' - ')
    attributes = [attribute.replace(' ', '_') for attribute in attributes]
    n_attributes = column.count(' - ')

    if n_attributes == 2:
        location = attributes[1]
    elif n_attributes == 6:
        location = attributes[2]
    elif n_attributes == 5:
        location = attributes[1]
    else:
        return column

    location_group = ontology.search_one(is_a=ontology['location'], iri=f'*{location}')
    if location_group is None:
        if n_attributes == 2:
            print('Primary')
        elif n_attributes == 6:
            print('Diagenetic')
        elif n_attributes == 5:
            print('Porosity')
        print(f'{location} NOT FOUND IN THE ONTOLOGY')
        return location
    parent_group = location_group.is_a[0]
    non_able_parents = [ontology[loc] for loc in ['location', 'diagenetic_location', 'porosity_location', 'primary_location']]
    if parent_group not in non_able_parents:
        location_group = parent_group

    return location_group.name


if __name__ == '__main__':
    idiom = 'ENUS'
    target_paths = glob('datasets/*/dataset.xlsx')
    target_pats = ['']

    for target_path in target_paths:
        if 'talara' in target_path.lower():
            continue
        print(f'processing {target_path}')
        # try:
        subtotals_df = calculate_subtotals(target_path, idiom)
        target_save_path = join(dirname(target_path), 'subtotals_dataset.xlsx')
        print(f'saving results to {target_save_path}')
        subtotals_df.to_excel(target_save_path)
        # except AttributeError as e:
        #     print(e)
        #     print(f'ERROR PROCESSING {target_path}!')

    print('Done')
