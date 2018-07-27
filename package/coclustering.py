import sklearn
# from sklearn.cluster.bicluster import S


class CoClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):

    def __init__(self, algorithm=None, n_generations=100, perfect=False,
                 min_features=5, max_features=1000, fitness_metric='silhouette_sklearn',
                 pop_size=128, pop_eval_rate=0):
        self.algorithm = algorithm
        self.n_generations = n_generations
        self.perfect = perfect
        self.min_features = min_features
        self.max_features = max_features
        self.fitness_metric = fitness_metric
        self.pop_size = pop_size
        self.pop_eval_rate = pop_eval_rate

    def run(self, X, y=None):
        pass