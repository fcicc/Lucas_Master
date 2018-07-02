import argparse
import csv
import os
from functools import partial
from os.path import isfile, join
import pandas as pd


def argument_parser() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')

    parser.add_argument('input_folder', type=str, default='test_scenario',
                        help='input CSV file')

    args = parser.parse_args()

    return args


def run():
    args = argument_parser()

    print('GATHERING FILES at ' + args.input_folder)
    csv_file_names = [
        file_name for file_name in os.listdir(args.input_folder)
        if isfile(join(args.input_folder, file_name))
           and file_name.endswith('.csv')
           and file_name != 'dataset.csv'
    ]
    print('DONE')

    print('READING THIN SECTION FILES')
    csv_data_files = [
        pd.read_csv(open(join(args.input_folder, csv_file_name)), index_col=0)
        for csv_file_name in csv_file_names]
    dfs = []
    for csv_file in csv_data_files:
        dfs.append(csv_file.applymap(partial(pd.to_numeric, errors='ignore')))
    csv_data_files = dfs

    for csv_file in csv_data_files:
        features = csv_file.index.values
        processed_features = []
        for feature in features:
            n_attributes = feature.count(' - ') + 1
            if n_attributes == 3:
                processed_features.append('[primary]' + feature)
            elif n_attributes == 7:
                processed_features.append('[diagenetic]' + feature)
            elif n_attributes == 6:
                processed_features.append('[porosity]' + feature)
            else:
                processed_features.append(feature)

        csv_file.index = processed_features

    print('DONE')

    print('DUPLICATES:')
    df_list = []
    for df, file_name in zip(csv_data_files, csv_file_names):
        print(file_name)
        for index in df.index.values:
            if df.index.values.tolist().count(index) > 1:
                df_list.append(index)
                print('\t' + str(index))

    def is_number(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def local_filter(x):
        if x.name == 'petrofacie' or x.name == 'Microscopic - Sorting:':
            return True
        elif all(x.apply(is_number)):
            return True
        else:
            print(x.name)
            return False

    print('Removing columns:')
    csv_data_files = [df[df.apply(local_filter, axis=1)] for df in csv_data_files]

    csv_data_files = [df.dropna(axis=0, how='any') for df in csv_data_files]

    csv_data_files = [df.groupby(df.index).sum() for df in csv_data_files]

    print(sum(df.shape[0] for df in csv_data_files))
    print(sum(df.shape[1] for df in csv_data_files))
    print('MERGING DATA')

    full_csv = pd.concat(csv_data_files, axis=1)
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

    full_csv['Phi stdev sorting'] = full_csv['Microscopic - Sorting:'].map(phi_translate)

    full_csv.to_csv(join(args.input_folder, 'dataset.csv'), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC,
                    float_format='%.10f')

    print('DONE!')


if __name__ == '__main__':
    run()
