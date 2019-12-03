import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from package.copac import copac
from package.main import parse_csv_dataset_to_df

input_file = './datasets/MargemEquatorial/dataset.csv'
args = type('', (object,), {'input_file': input_file})()
X, y = parse_csv_dataset_to_df(args)
samples_dist_matrix = distance.squareform(distance.pdist(X.values))
results = []

for k in range(2, len(X)):
    y_pred = copac(X, k=k)
    if 1 < len(set(y_pred)) < len(samples_dist_matrix):
        score = silhouette_score(samples_dist_matrix, y_pred, metric='precomputed')
    else:
        score = 0.0
    results.append((k, score))
    print('Computed for k=%s' % k)

plt.plot(*zip(*results))
plt.show()

# CB: k=20
