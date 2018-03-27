"""Genetic algorithm clustering by hard subspace."""

import argparse
import csv
import logging
import multiprocessing
import os
import random
import sys
import time
import re
import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functools import partial
from multiprocessing.pool import Pool
from deap import algorithms, base, creator, tools
from sklearn import cluster
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabaz_score, confusion_matrix,
                             f1_score, silhouette_score)
from sklearn.metrics.cluster import class_cluster_match
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def eval_host(X, samples_dist_matrix, cs):
    """Evaluate compactness and connectivity of host."""
    host_centers = np.zeros((len(cs), X.shape[1]))
    for i, symbiont in enumerate(cs):
        for mean, _, attr in symbiont:
            host_centers[i, attr] = mean
    
    # dist_matrix = distance.cdist(host_centers, X)
    # assigned_centers = np.argmin(dist_matrix, axis=0)
    # dist_to_closest_centers = [min(col) for col in dist_matrix.T]

    # compactness = np.sum(dist_to_closest_centers)

    # m = max(10, 0.01 * X.shape[1])
    # connectivity = 0
    # for i, row_i in enumerate(samples_dist_matrix):
    #     closest_neighbours = np.argsort(row_i)[:m]
    #     for j, close_idx in enumerate(closest_neighbours):
    #         if assigned_centers[i] == assigned_centers[close_idx]:
    #             connectivity += 1/(j+1)

    # return compactness, connectivity




def feature_relevance(X, y):
    """Calculate feature relevance according to the internal and external
       feature relevance."""
    clusters = unique_labels(y)
    features = X.columns.values
    C = 1

    MRs = {feature: [] for feature in features}
    for cluster_i in clusters:
        cluster_instances = X.loc[[i == cluster_i for i in y]]
        not_cluster_instances = X.loc[[i != cluster_i for i in y]]
        for feature in features:
            VI = np.std(cluster_instances[feature])
            VE = np.std(not_cluster_instances[feature])

            MR = VE / (VI + C)
            MRs[feature].append(MR)

    MRs = pd.DataFrame.from_dict(MRs, orient='index')
    MRs.columns = clusters

    return MRs


def argument_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='''Code implementation from "A Clustering-based Approach to
                       Identify Petrofacies from Petrographic Data".''')
    parser.add_argument(
        'input_dir',
        type=str,
        help='''input directory, containing a CSV dataset with name "dataset.csv"
             ''')
    parser.add_argument('--n-clusters', type=int, default=10,
                        help='number clusters')
    parser.add_argument('--cc-pop-size', type=int, default=50,
                        help='number of individuals in the symbiont population')
    parser.add_argument('--cs-pop-size', type=int, default=50,
                        help='number of individuals in the host population')
    parser.add_argument('--num-gen', type=int, default=100,
                        help='number of generations of the GA')
    parser.add_argument('-c', '--use-categorical', action='store_true',
                        help='wether to use features attributes as categorical individual data')

    args = parser.parse_args()

    return args


def extract_subtotals(X):
    """Extract subtotals from compositional feature's attributes."""
    compositional_features = [feature for feature in X.columns if ' - ' in feature]

    attributes = {}
    for feature in compositional_features:
        big_group = re.search('\[(.*)\]', feature).group(1)
        feature_attrs = re.sub('\[(.*)\]', '', feature).split(' - ')

        if big_group not in attributes:
            attributes[big_group] = [{} for _ in feature_attrs]

        for i, attribute in enumerate(feature_attrs):
            if attribute not in attributes[big_group][i]:
                attributes[big_group][i][attribute] = [0 for _ in range(X.shape[0])]
    
    for i, row in enumerate(X.iterrows()):
        for feature in compositional_features:
            if row[1][feature] > 0:
                big_group = re.search('\[(.*)\]', feature).group(1)
                feature_attrs = re.sub('\[(.*)\]', '', feature).split(' - ')

                for j, attribute in enumerate(feature_attrs):
                    attributes[big_group][j][attribute][i] += row[1][feature]

    df = {}
    for big_group in attributes:
        for position, features in enumerate(attributes[big_group]):
            for attribute in features:
                df['['+big_group+']'+str(position)+'-'+attribute] = features[attribute]

    df = pd.DataFrame.from_dict(df)

    return df
    
def estimate_n_clusters(X):
    "Find the best number of clusters through maximization of the log-likelihood from EM."
    last_log_likelihood = None
    kf = KFold(n_splits=10, shuffle=True)
    components = range(50)[1:]
    for n_components in components:
        gm = GaussianMixture(n_components=n_components)

        log_likelihood_list = []
        for train, test in kf.split(X):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                gm.fit(X[train, :])
            if not gm.converged_:
                raise Warning("GM not converged")
            log_likelihood = -gm.score_samples(X[test, :])

            log_likelihood_list += log_likelihood.tolist()

        avg_log_likelihood = np.average(log_likelihood_list)

        if last_log_likelihood is None:
            last_log_likelihood = avg_log_likelihood
        elif avg_log_likelihood+10E-6 <= last_log_likelihood:
            return n_components-1
        last_log_likelihood = avg_log_likelihood
    

def dynamic_GM_means(X):
    """Dynamically calculate gaussian means for the input data."""
    n_clusters = estimate_n_clusters(X)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return np.unique(GaussianMixture(n_components=n_clusters).fit(X).means_)


def cluster_grid_generation(X, pool):
    """Generate clusters grid from S-ESC."""
    cluster_1d_grid = []
    cols = [X[:,i].reshape(X[:,i].shape[0], 1)for i in range(X.shape[1])]
    cluster_1d_grid = pool.map(dynamic_GM_means, cols)

    cluster_1d_grid = [(mean, index, attr) for attr, cluster in enumerate(cluster_1d_grid) for index, mean in enumerate(cluster)]

    return cluster_1d_grid


def slm(individual, cc_population):
    """Single-level mutation."""
    prob = 1
    while random.random() <= prob:
        # Phase 1
        individual.remove(random.choice(individual))

        # Phase 2
        individual.append(random.choice(cc_population))

        # Phase 3
        individual.remove(random.choice(individual))
        individual.append(random.choice(cc_population))

        prob = prob/10
    
    return individual


def mlm(symbiont, cluster_1d_grid):
    """Multi-level mutation."""
    prob = 1
    while random.random() <= prob:
        # Phase 4
        symbiont.remove(random.choice(symbiont))

        # Phase 5
        symbiont.append(random.choice(cluster_1d_grid))

        # Phase 6
        symbiont.remove(random.choice(symbiont))
        symbiont.append(random.choice(cluster_1d_grid))

        prob = prob/10
    
    return symbiont


def clear_cc_population(cc_population, cs_population):
    """Eliminate symbionts not referenced by any host."""
    keep_cc_s = [False]*len(cc_population)

    for host in cs_population:
        for symbiont in host:
            keep_cc_s[cc_population.index(symbiont)] = True

    cc_population = [host for i, host in enumerate(cc_population) if keep_cc_s[i]]

    return cc_population


def pick_random_means(cluster_1d_grid, min, max):
    """Select non-repeated means in non-repeated features."""
    n_attrs = int(np.ceil(random.random()*(max-min))+min)
    means = []
    for _ in range(n_attrs):
        means.append(random.choice([mean for mean in cluster_1d_grid if mean[2] not in [attr for _, _, attr in means]]))
        
    return means


def host_pred(host, X):   
    host_centers = np.zeros((len(host), X.shape[1]))
    for i, symbiont in enumerate(host):
        for mean, _, attr in symbiont:
            host_centers[i, attr] = mean

    dist_matrix = distance.cdist(host_centers, X)
    pred = np.argmin(dist_matrix, axis=0)

    return pred


def main():
    """Main function."""
    args = argument_parser()

    start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    output_summary = open(
        os.path.join(
            args.input_dir,
            'dataset_analysis' +
            start_time +
            '.txt'),
        'w')

    output_summary.write(str(args) + '\n')

    output_summary.write('\n\nARGS = ' + str(args) + '\n')
    own_script = open(sys.argv[0])
    own_script_text = own_script.read()
    own_script.close()

    pool = Pool(multiprocessing.cpu_count())

    df = pd.read_csv(os.path.join(args.input_dir, 'dataset.csv'))
    y = df['petrofacie'].as_matrix()
    del df[df.columns[0]]
    del df['petrofacie']
    X = df

    if args.use_categorical:
        X = pd.concat([X, extract_subtotals(X)], axis=1)
    X_matrix = X.as_matrix()

    X_matrix = StandardScaler(with_std=False).fit(X_matrix).transform(X_matrix)

    samples_dist_matrix = distance.squareform(distance.pdist(X_matrix))


    ac = cluster.AgglomerativeClustering(n_clusters=args.n_clusters,
                                         affinity='manhattan',
                                         linkage='complete')

    logging.info('Step 1')
    cluster_1d_grid = cluster_grid_generation(X_matrix, pool)

    creator.create("fitness_max", base.Fitness, weights=(1, 1))
    creator.create("individual", list, fitness=creator.fitness_max)

    toolbox = base.Toolbox()
    # toolbox.register("map", pool.map)

    logging.info('Step 2')
    # toolbox.register("clstr_centr", random.choice, cluster_1d_grid)
    toolbox.register(
        "symbiont",
        pick_random_means,
        cluster_1d_grid,
        2, 20)
    toolbox.register("cc_population", tools.initRepeat, list, toolbox.symbiont)
    cc_population = toolbox.cc_population(n=args.cc_pop_size)

    logging.info('Step 3 (Step 4 ignored)')
    toolbox.register("clstr_sol", random.choice, cc_population)
    toolbox.register(
        "host",
        tools.initRepeat,
        creator.individual,
        toolbox.clstr_sol,
        n=int(np.ceil(random.random()*4+8)))
    toolbox.register("cs_population", tools.initRepeat, list, toolbox.host)
    cs_population = toolbox.cs_population(n=args.cs_pop_size)

    toolbox.register("evaluate", eval_host, X_matrix, samples_dist_matrix, ac)
    toolbox.register("select_parent_host", tools.selTournament, tournsize=4)
    toolbox.register("select_parent_symbiont", tools.selRandom, k=1)
    toolbox.register("select_host", tools.selNSGA2)

    fits = toolbox.map(toolbox.evaluate, cs_population)
    for fit, ind in zip(fits, cs_population):
        ind.fitness.values = fit

    logging.info('Step 5')
    NGEN = args.num_gen
    for gen in tqdm(range(NGEN)):
        # logging.info('Generation ' + str(gen + 1) + ' of ' + str(NGEN))

        cs_offspring = []
        # logging.info('Step 5.a.i')
        parent_hosts = toolbox.select_parent_host(cs_population, k=args.cs_pop_size)
        # logging.info('Step 5.a')
        for parent_host in parent_hosts:
    
            # logging.info('Step 5.a.ii')
            cloned_host = copy.deepcopy(parent_host)

            # logging.info('Step 5.a.iii')
            cloned_host = slm(cloned_host, cc_population)

            cs_offspring.append(cloned_host)

            # logging.info('Step 5.a.iv')
            parent_symbiont = random.choice(parent_host)

            # logging.info('Step 5.a.v')
            cloned_symbiont = copy.deepcopy(parent_symbiont)

            # logging.info('Step 5.a.vi')
            cloned_symbiont = mlm(cloned_symbiont, cluster_1d_grid)

            cc_population.append(cloned_symbiont)

        # logging.info('Step 5.b')
        fits = toolbox.map(toolbox.evaluate, cs_offspring)
        for fit, ind in zip(fits, cs_offspring):
            ind.fitness.values = fit

        # logging.info('Step 5.c, Step 5.d')
        cs_population = toolbox.select_host(cs_population+cs_offspring, k=len(cs_population))

        # logging.info('Step 5.e (Step 5.f ignored)')
        cc_population = clear_cc_population(cc_population, cs_population)

        # logging.info('Step 5.g (continue for loop)')

    logging.info('Step 6')
    for ind in cs_population:
        ind.fitness.values = (-ind.fitness.values[0], -ind.fitness.values[1])
    top_cs = toolbox.select_host(cs_population, k=1)[0]

    output_summary.write(own_script_text)

    best_pred = host_pred(top_cs, X_matrix)

    # y_pred = class_cluster_match(y, best_pred)
    # cm = confusion_matrix(y, y_pred)
    # cm = pd.DataFrame(data=cm, index=unique_labels(y),
    #                   columns=unique_labels(y))
    # print(cm)
    # print(len(top_cs))
    print('Adjusted Rand score:' + str(adjusted_rand_score(y, best_pred)))

    output_summary.close()

    print('Compactness/Connectivity:' + str(top_cs.fitness.values))

    objective_space = [ind.fitness.values for ind in cs_population]
    objective_space = pd.DataFrame(objective_space)
    objective_space.columns = ['compactness', 'connectivity']
    nsga_front = toolbox.select_host(cs_population, k=len(cs_population))
    objective_space['position_in_nsga'] = np.asarray([nsga_front.index(ind) for ind in cs_population])
    objective_space['front_adj_rand'] = [adjusted_rand_score(y, host_pred(host, X_matrix)) for host in nsga_front]
    plt.figure()
    points = plt.scatter(objective_space['compactness'],
                         objective_space['connectivity'],
                         c=objective_space['position_in_nsga'],
                         cmap='viridis', alpha=0.7)
    plt.xlabel('compactness')
    plt.xlabel('connectivity')
    plt.colorbar(points, label='NSGA-II score')

    plt.figure()
    sns.regplot(x='position_in_nsga', y='front_adj_rand', data=objective_space)

    plt.show()

if __name__ == '__main__':
    main()
