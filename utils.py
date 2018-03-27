import re
import pandas as pd
import numpy as np

def heatmap_features(input_file):

    df = pd.read_csv(input_file, index_col=0)

    attributes_counter = {}

    for feature in df.columns:
        if feature.startswith('['):
            typee = re.search('\[(.*?)\]', feature).group(0)
            attributes = feature.replace(typee, '')
            attributes = attributes.split(' - ')
            if typee not in attributes_counter:
                attributes_counter[typee] = [[] for _ in attributes]
            for i, attribute in enumerate(attributes):
                if attribute not in attributes_counter[typee][i]:
                    attributes_counter[typee][i].append(attribute)

    df_heatmap = {}
    for typee in attributes_counter:
        df_heatmap[typee] = [[] for _ in attributes_counter[typee]]
        for i, l in enumerate(attributes_counter[typee]):
            heat_df = pd.DataFrame(np.zeros((df.shape[0], len(l))), index=df['petrofacie'], columns=l)
            heat_df = heat_df.groupby(heat_df.index).sum()
            df_heatmap[typee][i] = heat_df

    petrofacies_freq = df['petrofacie'].value_counts()

    count_cols = 0
    for index, row in df.iterrows():
        for col, val in zip(row.index, row):
            if  col.startswith('[') and val > 0.1:
                typee = re.search('\[(.*?)\]', col).group(0)
                attributes = col.replace(typee, '')
                attributes = attributes.split(' - ')
                for i,attribute in enumerate(attributes):
                    df_heatmap[typee][i].loc[row['petrofacie'], attribute] += val/petrofacies_freq.loc[row['petrofacie']]


    for typee in df_heatmap:
        for i, df in enumerate(df_heatmap[typee]):
    #         df_heatmap[typee][i] = df.reindex_axis(df.mean().sort_values(ascending=False).index, axis=1)
            df_heatmap[typee][i].sort_index(inplace=True, axis=0)
            df_heatmap[typee][i].sort_index(inplace=True, axis=1)

    return df_heatmap