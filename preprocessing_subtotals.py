import re
from copy import deepcopy
from os.path import join, dirname
from glob import glob

import pandas as pd

MACRO_LOCATIONS = ['interstitial', 'framework', 'framework and interstitial']


def calculate_subtotals(target_path, idiom):
    dataset = pd.read_csv(target_path, delimiter=',', index_col=0)

    feature_names = list(map(lambda x: x.lower(), dataset.columns.values))
    feature_names = [re.sub('(\[.*\])', '', feature_name) for feature_name in feature_names]

    dataset.columns = feature_names

    result_dataset = pd.DataFrame(index=dataset.index)

    def extract_compositional_type(s):
        n_attributes = s.count(' - ') + 1
        if n_attributes == 3:
            return 'primary'
        elif n_attributes == 7:
            return 'diagenetic'
        elif n_attributes == 6:
            return 'porosity'
        else:
            return ''

    compositional_types = [extract_compositional_type(feature_name) for feature_name in feature_names]

    # ==================================================================================================================
    # PRIMARY SUBTOTALS
    primary_attributes_names = ['constituent', 'location', 'modification']
    primary_attributes = [feature_name.split(' - ') for feature_name, compositional_type
                          in zip(feature_names, compositional_types)
                          if compositional_type == 'primary']
    primary_attributes = pd.DataFrame(primary_attributes, columns=primary_attributes_names)
    grouped_primary_attributes = primary_attributes.groupby(['constituent', 'location'])
    for name, group in grouped_primary_attributes:
        group_name = ' - '.join(name)
        result_dataset['[primary-subtotal]'+group_name+' - framework'] = dataset.filter(regex=f'{group_name}.*-[^-]*').sum(axis=1)
    # ==================================================================================================================

    # ==================================================================================================================
    # DIAGENETIC SUBTOTALS
    diagenese_mapping = pd.read_excel(
        './subtotals_instructive_tables/Categorias de Localização Diagenética revDeRos.xlsx')
    diagenese_mapping = diagenese_mapping.apply(lambda x: x.astype(str).str.lower())

    diagenetic_attributes_names = ['consituent', 'habit', 'location', 'modification', 'paragenetic relation',
                                   'paragenetic relation constituents', 'paragenetic relation constituent location']
    diagenetic_attributes = [feature_name.split(' - ') for feature_name, compositional_type
                             in zip(feature_names, compositional_types)
                             if compositional_type == 'diagenetic']
    diagenetic_attributes = pd.DataFrame(diagenetic_attributes, columns=diagenetic_attributes_names)

    for feature in diagenetic_attributes.iterrows():
        feature_name = feature[1]
        feature_values = dataset[' - '.join(feature_name)]
        if feature_name['location'] == '' and feature_name['paragenetic relation constituent location'] == '':
            raise ValueError(f'Line:\n{" - ".join(feature_name)} is not complete enough on file {target_path}.'
                             f'At least its LOCATION or PARAGENETIC RELATION CONSTITUENT LOCATION have to be filled'
                             f'properly.')
        if feature_name['paragenetic relation constituent location'] in MACRO_LOCATIONS:
            subtotal_feature_name = deepcopy(feature_name)
            subtotal_feature_name['macro location'] = query['location'].values[0]
            del subtotal_feature_name['paragenetic relation constituent location']
            subtotal_feature_name = '[diagenetic-subtotal]' + ' - '.join(subtotal_feature_name)
            if subtotal_feature_name not in result_dataset:
                result_dataset[subtotal_feature_name] = \
                    pd.Series([0] * result_dataset.shape[0], index=result_dataset.index)
            result_dataset[subtotal_feature_name] += feature_values
        else:
            query = diagenese_mapping[diagenese_mapping['VALUE_' + idiom] == feature_name['paragenetic relation' \
                                                                                          ' constituent location']]
            if query.empty:
                query = diagenese_mapping[diagenese_mapping['VALUE_' + idiom] == feature_name['location']]
                if query.empty:
                    raise ValueError(f'Could not define macro location for {" - ".join(feature_name)} in file'
                                     f'{target_path}')
                else:
                    subtotal_feature_name = deepcopy(feature_name)
                    subtotal_feature_name['macro location'] = query['location'].values[0]
                    del subtotal_feature_name['location']
                    subtotal_feature_name = '[diagenetic-subtotal]'+' - '.join(subtotal_feature_name)
                    if subtotal_feature_name not in result_dataset:
                        result_dataset[subtotal_feature_name] = \
                            pd.Series([0] * result_dataset.shape[0], index=result_dataset.index)
                    result_dataset[subtotal_feature_name] += feature_values
            else:
                subtotal_feature_name = deepcopy(feature_name)
                subtotal_feature_name['macro location'] = query['location'].values[0]
                del subtotal_feature_name['paragenetic relation constituent location']
                subtotal_feature_name = '[diagenetic-subtotal]'+' - '.join(subtotal_feature_name)
                if subtotal_feature_name not in result_dataset:
                    result_dataset[subtotal_feature_name] = \
                        pd.Series([0] * result_dataset.shape[0], index=result_dataset.index)
                result_dataset[subtotal_feature_name] += feature_values
    # ==================================================================================================================

    # ==================================================================================================================
    # POROSITY SUBTOTALS
    porosity_attributes_names = ['porosity', 'location', 'modification', 'paragenetic relation',
                                 'paragenetic relation constituents',
                                 'paragenetic relation constituent location']
    porosity_attributes = [feature_name.split(' - ') for feature_name, compositional_type
                           in zip(feature_names, compositional_types)
                           if compositional_type == 'porosity']
    porosity_attributes = pd.DataFrame(porosity_attributes, columns=porosity_attributes_names)
    grouped_primary_attributes = porosity_attributes.groupby(['porosity', 'location'])
    for name, group in grouped_primary_attributes:
        group_name = ' - '.join(list(name))
        result_dataset['[porosity-subtotal]'+group_name] = dataset.filter(regex=f'^{group_name}.*-.*-.*-.*-[^-]*').sum(axis=1)
    # ==================================================================================================================

    if any(result_dataset.isna().any().values):
        print(result_dataset.isna().any())
        raise ValueError('There should not be any NaN values inside the subtotals data frame!')

    result_dataset['petrofacie'] = dataset['petrofacie']
    result_dataset['grain_size'] = dataset['porosity']
    result_dataset['phi stdev sorting'] = dataset['phi stdev sorting']

    return result_dataset


if __name__ == '__main__':
    idiom = 'ENUS'
    target_paths = glob('datasets/*/dataset.csv')

    for target_path in target_paths:
        print(f'processing {target_path}')
        subtotals_df = calculate_subtotals(target_path, idiom)
        subtotals_df.to_csv(join(dirname(target_path), 'subtotals_dataset.csv'))
        print('DONE')
