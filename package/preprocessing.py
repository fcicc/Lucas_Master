import pandas as pd

GRAIN_SIZE_CLASSES = [
    # Clay
    0.0039,
    # Very fine silt
    0.0078,
    # Fine silt
    0.0156,
    # Medium silt
    0.0312,
    # Coarse silt
    0.625,
    # Very fine sand
    0.125,
    # Fine sand
    0.25,
    # Medium sand
    0.5,
    # Coarse sand
    1,
    # Very coarse sand
    2,
    # Granules
    4,
    # Pebbles
    64,
    # Cobbles
    256
    # Boulders
]


def binaryze_main_grain_size(value):
    """

    :type value: int
    """
    result = [0] * (len(GRAIN_SIZE_CLASSES) + 1)
    for i, _ in enumerate(GRAIN_SIZE_CLASSES):
        if value < GRAIN_SIZE_CLASSES[i]:
            result[i] = 100
            break
    sum_result = sum(result)
    if sum_result == 0:
        result[-1] = 100

    return result


def binaryze_column(series):
    """

    :type series: pd.Series
    """
    result = list(map(binaryze_main_grain_size, series))

    result = pd.DataFrame(result, columns=['Clay', 'Very fine silt', 'Fine silt', 'Medium silt', 'Coarse silt',
                                           'Very fine sand', 'Fine sand', 'Medium sand', 'Coarse sand',
                                           'Very coarse sand', 'Granules', 'Pebbles', 'Cobbles', 'Boulders'],
                          index=series.index)

    return result


def range_grain_size(df: pd.DataFrame):
    """

    :type df: pd.DataFrame
    """
    df['Minimum grain size'] = (df['Main/single size mode(mm):'] - df['Phi stdev sorting']) * 100
    df['Maximum'] = (df['Main/single size mode(mm):'] + df['Phi stdev sorting']) * 100

    return df
