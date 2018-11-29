import argparse
import copy
import csv
import os
from glob import glob

import numpy as np
from functools import partial
from os.path import isfile, join
import pandas as pd


def argument_parser() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')

    parser.add_argument('input_folder', type=str, default='test_scenario',
                        help='input CSV folder')

    parser.add_argument('output_file', type=str, default='dataset.csv',
                        help='output CSV file')

    args = parser.parse_args()

    return args


def map_attribute_name(old_name):
    new_name = old_name.lower()
    df = pd.read_excel('mapping_old_attribute_names.xlsx')
    for index, row in df.iterrows():
        splitted_attribute = new_name.split(' - ')
        if row['old_name'] in splitted_attribute:
            new_name = new_name.replace(row['old_name'], row['new_name'])
    return new_name


def run():
    args = argument_parser()

    print('GATHERING FILES at ' + args.input_folder)
    csv_file_names = [
        file_name for file_name in os.listdir(args.input_folder)
        if isfile(join(args.input_folder, file_name))
        and file_name.endswith('.csv')
        and 'dataset' not in file_name
        and file_name != args.output_file
    ]
    print('DONE')

    print('READING THIN SECTION FILES')
    csv_data_files = [
        pd.read_csv(open(join(args.input_folder, csv_file_name)), index_col=0, encoding='utf-8')
        for csv_file_name in csv_file_names]
    dfs = []
    for csv_file in csv_data_files:
        df = csv_file.applymap(partial(pd.to_numeric, errors='ignore'))
        df.index = csv_file.index.map(map_attribute_name)
        # df.drop('', axis='columns', )
        dfs.append(df)
    csv_data_files = dfs

    print('DONE')

    # print('DUPLICATES:')
    df_list = []
    for df, file_name in zip(csv_data_files, csv_file_names):
        # print(file_name)
        for index in df.index.values:
            if df.index.values.tolist().count(index) > 1:
                df_list.append(index)
                # print('\t' + str(index))

    print('Removing rows:')
    csv_data_files_filtered = []
    for df in csv_data_files:
        df = df.loc[df.index.drop_duplicates(), :]
        df_clone = copy.deepcopy(df)
        for index in df.index:
            if type(index) != str:
                continue
            n_attributes = index.count(' - ') + 1
            if n_attributes in [3, 7, 6]:
                class_str = None
                if n_attributes == 3:
                    class_str = '[primary]'
                elif n_attributes == 7:
                    class_str = '[diagenetic]'
                elif n_attributes == 6:
                    class_str = '[porosity]'
                df_clone.rename({index: class_str + index}, inplace=True, axis='index')
            elif 'sorting' in index.lower():
                df_clone.rename({index: 'sorting'}, inplace=True, axis='index')
            elif '(mm)' in index.lower() and ('main' in index.lower() or 'principal' in index.lower()):
                df_clone.rename({index: 'grain_size'}, inplace=True, axis='index')
            elif index.lower() == 'porosity':
                df_clone.rename({index: 'porosity'}, inplace=True, axis='index')
            elif index.lower() == 'petrofacie':
                pass
            else:
                print(f'DROP {index}')
                df_clone.drop(index=index, inplace=True)
        csv_data_files_filtered += [df_clone]

    csv_data_files = csv_data_files_filtered

    csv_data_files = [df.dropna(axis=0, how='all') for df in csv_data_files]

    csv_data_files = [df.groupby(df.index, axis=0).sum(axis=0) for df in csv_data_files]

    # print(sum(df.shape[0] for df in csv_data_files))
    # print(sum(df.shape[1] for df in csv_data_files))
    print('MERGING DATA')

    full_csv = pd.concat(csv_data_files, axis=1, sort=False)

    full_csv = full_csv[full_csv.columns.drop(list(full_csv.filter(regex='Unnamed*')))]
    if 'petrofacie' in full_csv.columns.values:
        full_csv = full_csv.dropna(subset=['petrofacie'], axis=1)
    full_csv = full_csv.fillna(value=0)

    full_csv: pd.DataFrame = full_csv.transpose()

    def phi_translate(val):
        """

        :type val: str
        """
        if isinstance(val, str):
            lower = val.lower()
            if lower == 'very well sorted' or lower == 'muito bem selecionado':
                return 0.175
            elif lower == 'well sorted' or lower == 'bem selecionado':
                return .425
            elif lower == 'moderately sorted' or lower == 'moderadamente selecionado':
                return .75
            elif lower == 'poorly sorted' or lower == 'mal selecionado':
                return 1.5
            elif lower == 'very poorly sorted' or lower == 'muito mal selecionado':
                return 3
        else:
            return 0

    if 'sorting' in full_csv.columns.values:
        full_csv['phi stdev sorting'] = full_csv['sorting'].map(phi_translate)

    cols = list(full_csv.columns.values)
    full_csv = full_csv[cols]
    full_csv = full_csv.loc[:, (full_csv != 0).any(axis=0)]

    columns = pd.MultiIndex.from_arrays([['raw' for _ in full_csv.columns.values], full_csv.columns.values, full_csv.columns.values],
                                        names=['top_level', 'features_groups', 'features'])
    full_csv.columns = columns
    full_csv.index.name = 'samples'

    try:
        drop_indexes = [row for row, val in zip(full_csv.index, np.sum(full_csv.values, axis=1)) if val < 95]
        full_csv.drop(drop_indexes, axis=0, inplace=True)
    except TypeError as e:
        pass

    full_csv.to_excel(args.output_file)

    print('DONE!')


if __name__ == '__main__':
    run()
