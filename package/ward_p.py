from copy import deepcopy

import numpy as np
import sklearn
from scipy.spatial.distance import minkowski
from sklearn.feature_selection import VarianceThreshold


def update_all_distances(cluster_centers, W, p, Nk, all_distances, k1_min, k2_min, initial_k):
    # remove k2 from cluster_centers
    all_distances[k2_min, :] = np.inf
    all_distances[:, k2_min] = np.inf

    # update the distances related to z1
    for k in range(initial_k):
        if k == k1_min or Nk[k, 0] == 0 or k == k2_min:
            continue
        avg_w = ((W[k1_min, :] + W[k, :]) / 2) ** p
        all_distances[k1_min, k] = ((Nk[k1_min, 0] * Nk[k, 0]) / (Nk[k1_min, 0] + Nk[k, 0])) * sum((abs(cluster_centers[k1_min, :] - cluster_centers[k, :]) ** p) * avg_w)
        all_distances[k, k1_min] = all_distances[k1_min, k]

    return all_distances


def get_new_W(Data, W, U, cluster_centers, K, M, p, kernel_feature):
    # D = np.zeros((K, M))
    # for l in range(K):
    #     for j in range(M):
    #         D[l, j] = np.sum(np.abs(Data[U == l, j] - cluster_centers[l, j]) ** p)
    #
    # D = D + 0.0001
    #
    # # Calculate the actual Weight for each column
    # if p != 1:
    #     exp = 1 / (p - 1)
    #     for l in range(K):
    #         for j in range(M):
    #             W[l, j] = 1 / sum((np.full(M, D[l, j]) / D[l, :]) ** exp)
    # else:
    #     for l in range(K):
    #         MinIndex = np.argmin(D[l, :])
    #         W[l, :M] = 0  # necessary to zero all others
    #         W[l, MinIndex] = 1
    #
    # return W

    epsilon = 1e-5
    for l in range(K):
        for j in range(M):
            W[l, j] = 1 / (np.std(Data[U==l, j]))


def merge(k1_min, k2_min, U, Data, Nk, M, cluster_centers, p, K):
    Nk[k1_min, 0] = Nk[k1_min, 0] + Nk[k2_min, 0]
    U[U == k2_min] = k1_min
    Nk[k2_min, 0] = 0
    cluster_centers[k1_min, :] = new_cmt(Data[U == k1_min, :], p)
    cluster_centers[k2_min, :] = np.full((1, M), np.inf)
    K = K - 1

    return U, Nk, cluster_centers, K


def new_cmt(Data, p):
    # Calculates the Minkowski center at a given p.
    # Data MUST BE EntityxFeatures and standardised.
    N, M = Data.shape
    if p == 1:
        data_center = np.median(Data, axis=0)
        return data_center
    elif p == 2:
        data_center = np.mean(Data, axis=0)
        return data_center
    elif N == 1:
        data_center = Data
        return data_center
    else:
        gradient = np.full(M, 0.001)
        # zeroes_index = np.zeros((N, 1), dtype=np.int8)
        data_center = np.sum(Data, axis=0) / N
        distance_to_data_center = np.sum(abs(Data - np.tile(data_center, (N, 1))) ** p, axis=0)
        new_data_center = data_center + gradient
        distance_to_new_data_center = np.sum(abs(Data - np.tile(new_data_center, (N, 1))) ** p)
        gradient[distance_to_data_center < distance_to_new_data_center] = gradient[distance_to_data_center < distance_to_new_data_center] * -1
        while True:
            new_data_center = data_center + gradient
            distance_to_new_data_center = np.sum(abs(Data - np.tile(new_data_center, (N, 1))) ** p, axis=0)
            gradient[distance_to_new_data_center >= distance_to_data_center] = gradient[distance_to_new_data_center >= distance_to_data_center] * 0.9

            data_center[distance_to_new_data_center < distance_to_data_center] = new_data_center[distance_to_new_data_center < distance_to_data_center]
            distance_to_data_center[distance_to_new_data_center < distance_to_data_center] = distance_to_new_data_center[distance_to_new_data_center < distance_to_data_center]
            if all(abs(gradient) < 0.0001):
                break
    return data_center


class WardP(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):

    def __init__(self, kernel_feature, perfect=False, p=2, n_clusters=10):
        self.kernel_feature = kernel_feature
        self.perfect = perfect
        self.p = p
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        dataset = X

        # Dropping zero-variation features
        dataset = dataset[:, ~np.all(np.isnan(dataset), axis=0)]
        dataset = VarianceThreshold().fit_transform(dataset)

        for j in range(dataset.shape[1]):
            v_avg = np.average(dataset[:, j])
            v_range = np.max(dataset[:, j]) - np.min(dataset[:, j])
            for i in range(dataset.shape[0]):
                dataset[i, j] = (dataset[i, j] - v_avg) / v_range

        N, M = dataset.shape
        self.U_ = np.vstack(np.arange(N, dtype=np.int8))
        K = N
        Nk = np.ones((N, 1))  # cluster sizes
        cluster_centers = deepcopy(dataset)
        self.W_ = np.full((K, M), 1 / M)
        InitialK = K

        AllDistances = np.full((InitialK, InitialK), np.inf)
        for k1 in range(InitialK):
            for k2 in range(InitialK)[k1+1:]:
                AllDistances[k1, k2] = 0.5 * sum((abs(cluster_centers[k1, :] - cluster_centers[k2, :]) ** self.p) *
                                                 (self.W_[0, :] ** self.p))
                AllDistances[k2, k1] = AllDistances[k1, k2]

        UIndex = 0
        while K > self.n_clusters:
            # 1st steep: look for clusters to merge
            k2Min, k1Min = np.unravel_index(AllDistances.argmin(), AllDistances.shape)
            assert k2Min != k1Min
            # merge clusters k1Min and k2Min
            UIndex = UIndex + 1
            self.U_ = np.concatenate((self.U_, np.zeros((N, 1), dtype=np.int8)), axis=1)
            self.U_[:, UIndex], Nk, cluster_centers, K = merge(k1Min, k2Min, deepcopy(self.U_[:, UIndex - 1]), dataset, Nk, M, cluster_centers, self.p, K)
            self.W_ = get_new_W(dataset, self.W_, self.U_[:, UIndex], cluster_centers, InitialK, M, self.p, self.kernel_feature)
            AllDistances = update_all_distances(cluster_centers, self.W_, self.p, Nk, AllDistances, k1Min, k2Min, InitialK)

        self.labels_ = self.U_[:, UIndex]