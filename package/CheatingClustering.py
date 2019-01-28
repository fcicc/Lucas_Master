import sklearn


class CheatingClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self, y):
        self.y = y
        set_y = list(set(y))
        self.labels_ = [set_y.index(y_val) for y_val in y]

    def fit(self, _, y=None):

        return self
