import matplotlib.pyplot as plt
from numpy import arange
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from package.copac import copac
from package.main import parse_csv_dataset_to_df

input_file = './datasets/MargemEquatorial/dataset.csv'
args = type('', (object,), {'input_file': input_file})()
X, y = parse_csv_dataset_to_df(args)
samples_dist_matrix = distance.squareform(distance.pdist(X.values))
results = []

for eps in arange(8.0, 12.0, 0.1):
    y_pred = copac(X, k=20, mu=1, eps=eps)
    if 1 < len(set(y_pred)) < len(samples_dist_matrix):
        score = silhouette_score(samples_dist_matrix, y_pred, metric='precomputed')
    else:
        score = 0.0

    results.append((eps, score))
    print('Computed for eps=%s' % eps)

plt.plot(*zip(*results))
plt.show()

# CB: k=20, mu=1, eps=25.0
# EM: k=123, mu=1, eps=10.0
