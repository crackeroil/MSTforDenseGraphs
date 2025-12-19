import math
import random
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from datetime import datetime
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
from pyspark import SparkConf, SparkContext

from Plotter import *
from DataReader import *
from BenchmarkLogger import *

import sys

# A lot of communication rounds happen for small epsilon values
sys.setrecursionlimit(10**6)


def get_clustering_data():
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=0.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=0.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                        cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    swiss_roll = make_swiss_roll(n_samples, noise=0.05)

    s_shape = make_s_curve(n_samples, noise=0.05)

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {}),
        (swiss_roll, {}),
        (s_shape, {})]

    return datasets

def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    vertices = []
    size = 0
    three_d = False
    for line in dataset:
        if len(line) == 2:
            vertices.append([line[0], line[1]])
        elif len(line) == 3:
            vertices.append([line[0], line[1], line[2]])
            three_d = True
    if three_d:
        dict = {}
        for i in range(len(dataset)):
            dict2 = {}
            for j in range(i + 1, len(dataset)):
                dict2[j] = np.sqrt(np.sum(np.square(dataset[i] - dataset[j])))
                size += 1
            dict[i] = dict2
    else:
        d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
        dict = {}
        # Run with less edges
        for i in range(len(d_matrix)):
            dict2 = {}
            for j in range(i, len(d_matrix)):
                if i != j:
                    size += 1
                    dict2[j] = d_matrix[i][j]
            dict[i] = dict2
    return dict, size, vertices


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def find_mst(E_list):
    """
    finds the mst of graph G given list of edges
    :param E_list: edges of the graph as list [(u, v, w), ...]
    :return: the mst edges
     """
    vertices = set()
    for e in E_list:
        vertices.add(e[0])
        vertices.add(e[1])
        
    E = sorted(E_list, key=get_key)
    connected_component = set()
    mst = []
    
    # Quick Kruskal with DSU
    parent = {v: v for v in vertices}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    for edge in E:
        u, v, w = edge
        if union(u, v):
            mst.append(edge)
            
    return mst


def compute_local_mst(edges):
    """
    Wrapper for map function to compute MST on specific partition
    """
    return find_mst(list(edges))


def flatten_edges(E_dict):
    """
    Flattens the dict of dicts edge representation to a list of tuples
    """
    flat_edges = []
    for u, neighbors in E_dict.items():
        for v, w in neighbors.items():
            flat_edges.append((u, v, w))
    return flat_edges


def mst_dense_1(sc, V_count, edges, epsilon, logger=None, dataset_name=None, recursion_depth=0):
    """
    Implements the MST-Dense-1 algorithm using PySpark
    :param sc: SparkContext
    :param V_count: Number of vertices |V| = n
    :param edges: List of edges [(u, v, w), ...]
    :param epsilon: parameter epsilon
    :return: MST edges
    """
    n = V_count
    m = len(edges)
    y = n ** (1 + epsilon)

    if m <= y:
        return find_mst(edges)
    
    x = math.ceil(m / y)
    
    # Randomly partition E into x sets
    # We use PySpark to parallelize this
    rdd_edges = sc.parallelize(edges)
    
    # Map each edge to a random key [0, x-1]
    # Group by key to form partitions
    # Compute MST for each partition
    # Collect results
    start_reduce = datetime.now()
    sub_msts_edges = rdd_edges.map(lambda e: (random.randint(0, x-1), e)) \
                              .groupByKey() \
                              .mapValues(compute_local_mst) \
                              .flatMap(lambda pair: pair[1]) \
                              .collect()
    duration_reduce = (datetime.now() - start_reduce).total_seconds()

    if logger:
        logger.log('MST-Dense-1', dataset_name, epsilon, 'Map-Reduce Step', recursion_depth, len(edges), duration_reduce, f'x={x}')
    
    return mst_dense_1(sc, V_count, sub_msts_edges, epsilon, logger, dataset_name, recursion_depth + 1)

def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help='Used for smaller dataset and testing', action='store_true')
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    args = parser.parse_args()
    
    # Initialize Spark with increased memory
    conf = SparkConf().setAppName("MST-Dense-1").setMaster("local[*]").set("spark.driver.memory", "16g")
    sc = SparkContext(conf=conf)
    
    logger = BenchmarkLogger()

    start_time = datetime.now()

    datasets_syn = get_clustering_data()
    names_datasets = ['TwoCircles', 'TwoMoons', 'Varied', 'Aniso', 'Blobs', 'Random', 'swissroll', 'sshape']
    
    num_clusters = [2, 2, 3, 3, 3, 2, 2, 2]
    cnt = 0
    time = []
    file_location = 'my_results/assignment/'
    plotter = Plotter(None, None, file_location)
    data_reader = DataReader()
    
    # --- Synthetic Datasets ---
    print("\n--- Processing Synthetic Datasets ---")
    for dataset in datasets_syn:
        timestamp = datetime.now()
        dataset_name = names_datasets[cnt]
        E_dict, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        
        plotter.set_vertex_coordinates(vertex_coordinates)
        plotter.set_dataset(dataset_name + '_eps' + str(args.epsilon))
        plotter.update_string()
        plotter.reset_round()
        
        V_count = len(vertex_coordinates)
        timestamp = datetime.now()
        
        # Flatten edges for the new algorithm
        flat_edges = flatten_edges(E_dict)
        
        logger.log('MST-Dense-1', dataset_name, args.epsilon, 'Start', 0, len(flat_edges), 0)
        
        mst = mst_dense_1(sc, V_count, flat_edges, args.epsilon, logger, dataset_name)
        
        duration = (datetime.now() - timestamp).total_seconds()
        time.append(duration)
        
        logger.log('MST-Dense-1', dataset_name, args.epsilon, 'End', 'Final', len(mst), duration)
        
        timestamp = datetime.now()
        if len(vertex_coordinates[0]) > 2:
            plotter.plot_mst_3d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        else:
            plotter.plot_mst_2d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        cnt += 1

    # --- Custom Datasets ---
    print("\n--- Processing Custom Datasets ---")

    datasets_custom = ['datasets/gplus_combined.txt', 'datasets/twitter_combined.txt', 'datasets/kron_g500-logn16.txt']
    
    for dataset_path in datasets_custom:
        dataset_name = os.path.basename(dataset_path)
        timestamp = datetime.now()
        
        vertex_list, size, E_dict = data_reader.read_and_filter_dataset(dataset_path, min_degree=100)
        
        V_count = len(vertex_list)
        read_duration = (datetime.now() - timestamp).total_seconds()
        
        logger.log('MST-Dense-1', dataset_name, args.epsilon, 'DataLoad', 0, size, read_duration)
        
        timestamp = datetime.now()
        
        flat_edges = flatten_edges(E_dict)
        
        mst = mst_dense_1(sc, V_count, flat_edges, args.epsilon, logger, dataset_name)
        
        duration = (datetime.now() - timestamp).total_seconds()
        time.append(duration)
        
        logger.log('MST-Dense-1', dataset_name, args.epsilon, 'Complete', 'Final', len(mst), duration)
    
    print('Done...')
    for i in range(len(names_datasets)):
        if i < len(time):
             print(f'Dataset {names_datasets[i]} took: ', time[i])
    
    for i in range(len(datasets_custom)):
        idx = len(names_datasets) + i
        if idx < len(time):
            print(f'Dataset {datasets_custom[i]} took: ', time[idx])
    
    sc.stop()


if __name__ == '__main__':
    main()
