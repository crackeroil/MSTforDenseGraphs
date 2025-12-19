import math
import csv
import random
import scipy.spatial

from argparse import ArgumentParser
from datetime import datetime
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
from pyspark import SparkConf, SparkContext

from Plotter import *
from DataReader import *
from BenchmarkLogger import *
import os


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



def partion_vertices(vertices, k):
    """
    Partitioning of the vertices in k smaller subsets (creates a partitioning twice
    :param vertices: all vertices
    :param k: number of subsets that need to be created
    :return: the partitioning in list format
    """
    U = []
    V = []
    random.shuffle(vertices)
    verticesU = vertices.copy()
    random.shuffle(vertices)
    verticesV = vertices.copy()
    for i in range(len(vertices)):
        if i < k:
            U.append({verticesU[i]})
            V.append({verticesV[i]})
        else:
            U[i % k].add(verticesU[i])
            V[i % k].add(verticesV[i])
    return U, V


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def find_mst(U, V, E):
    """
    finds the mst of graph G = (U union V, E)
    :param U: vertices U
    :param V: vertices V
    :param E: edges of the graph
    :return: the mst and edges not in the mst of the graph
     """
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    E = sorted(E, key=get_key)
    connected_component = set()
    mst = []
    remove_edges = set()
    while len(mst) < len(vertices) - 1 and len(connected_component) < len(vertices):
        if len(E) == 0:
            break
        change = False
        i = 0
        while i < len(E):
            if len(connected_component) == 0:
                connected_component.add(E[i][0])
                connected_component.add(E[i][1])
                mst.append(E[i])
                change = True
                E.remove(E[i])
                break
            else:
                if E[i][0] in connected_component:
                    if E[i][1] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][1])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                elif E[i][1] in connected_component:
                    if E[i][0] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][0])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                else:
                    i += 1
        if not change:
            if len(E) != 0:
                connected_component.add(E[0][0])
                connected_component.add(E[0][1])
                mst.append(E[0])
                E.remove(E[0])
    for edge in E:
        remove_edges.add(edge)
    if len(mst) != len(vertices) - 1 or len(connected_component) != len(vertices):
        print('Warning: parition cannot have a full MST! Missing edges to create full MST.')
        # print('Error: MST found cannot be correct \n Length mst: ', len(mst), '\n Total connected vertices: ',
        #       len(connected_component), '\n Number of vertices: ', len(vertices))
    return mst, remove_edges


def get_edges(U, V, E):
    """
    :param U: subset of vertices (u_j)
    :param V: subset of vertices (v_i)
    :param E: all edges of the whole graph
    :return: all edges that are part of the graph u_j U v_j
    """
    edges = set()
    for node1 in U:
        for node2 in V:
            if node1 in E:
                if node2 in E[node1]:
                    edges.add((node1, node2, E[node1][node2]))
                elif node2 in E:
                    if node1 in E[node2]:
                        edges.add((node2, node1, E[node2][node1]))
            elif node2 in E:
                if node1 in E[node2]:
                    edges.add((node2, node1, E[node2][node1]))
    edge_list = []
    for edge in edges:
        edge_list.append(edge)
    return U, V, edge_list

def flatten_edges(E_dict):
    """
    Flattens the dict of dicts edge representation to a list of tuples
    """
    flat_edges = []
    for u, neighbors in E_dict.items():
        for v, w in neighbors.items():
            flat_edges.append((u, v, w))
    return flat_edges

def reduce_edges(sc, vertices, E, c, epsilon, logger=None, dataset_name=None, round_num=0):
    """
    Uses PySpark to distribute the computation of the MSTs,
    Randomly partition the vertices twice in k subsets (U = {u_1, u_2, .., u_k}, V = {v_1, v_2, .., v_k})
    For every intersection between U_i and V_j, create the subgraph and find the MST in this graph
    Remove all edges from E that are not part of the MST in the subgraph
    :param sc: SparkContext
    :param vertices: vertices in the graph
    :param E: edges of the graph
    :param c: constant
    :param epsilon:
    :return:The reduced number of edges
    """
    
    n = len(vertices)
    k = math.ceil(n ** ((c - epsilon) / 2))
    print("k: ", k)
    U, V = partion_vertices(vertices, k)

    # Broadcast E to avoid sending the whole dictionary to every task serially
    E_broadcast = sc.broadcast(E)

    start_reduce = datetime.now()
    rddUV = sc.parallelize(U).cartesian(sc.parallelize(V)).map(lambda x: get_edges(x[0], x[1], E_broadcast.value)).map(
        lambda x: (find_mst(x[0], x[1], x[2])))
    both = rddUV.collect()
    duration_reduce = (datetime.now() - start_reduce).total_seconds()
    
    if logger:
        logger.log('MST-Fast', dataset_name, epsilon, 'Map-Reduce Step', round_num, len(flatten_edges(E)), duration_reduce, f'k={k}')
    
    # Unpersist to free memory
    E_broadcast.unpersist()

    mst = []
    removed_edges = set()
    for i in range(len(both)):
        mst.append(both[i][0])
        for edge in both[i][1]:
            removed_edges.add(edge)

    return mst, removed_edges


def remove_edges(E, removed_edges):
    """
    Removes the edges, which are removed when generating msts
    :param E: current edges
    :param removed_edges: edges to be removed
    :return: return the updated edge dict
    """
    for edge in removed_edges:
        if edge[1] in E[edge[0]]:
            del E[edge[0]][edge[1]]
    return E


def create_mst(sc, V, E, epsilon, size, vertex_coordinates, plot_intermediate=False, plotter=None, logger=None, dataset_name='Unknown'):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param sc: SparkContext
    :param plotter: class to plot graphs
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param size: number of edges
    :param plot_intermediate: boolean to indicate if intermediate steps should be plotted
    :param vertex_coordinates: coordinates of vertices
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    n = len(V)
    c = math.log(size / n, n)
    print("C", c)
    total_runs = 0
    
    if logger:
        logger.log('MST-Fast', dataset_name, epsilon, 'Start', 0, size, 0)
        
    while size > np.power(n, 1 + epsilon):
        total_runs += 1
        round_start = datetime.now()
        if plotter is not None:
            plotter.next_round()
            
        mst, removed_edges = reduce_edges(sc, V, E, c, epsilon, logger, dataset_name, total_runs)
        
        if plot_intermediate and plotter is not None:
            if len(vertex_coordinates[0]) > 2:
                plotter.plot_mst_3d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)
            else:
                plotter.plot_mst_2d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)
        E = remove_edges(E, removed_edges)
        print('Total edges removed in this iteration', len(removed_edges))
        size = size - len(removed_edges)
        print('New total of edges: ', size)
        
        duration_round = (datetime.now() - round_start).total_seconds()
        if logger:
            logger.log('MST-Fast', dataset_name, epsilon, 'Round', total_runs, size, duration_round, f'Removed {len(removed_edges)}')
            
        c = (c - epsilon) / 2
    # Now the number of edges is reduced and can be moved to a single machine
    # Use the original vertices V (passed as arg), converted to set to ensure compatibility
    # instead of range(n) as this breaks for non-integer or non-contiguous IDs
    V_set = set(V)
    items = E.items()  # returns [(x, {y : 1})]
    edges = []
    for item in items:
        valid_u = item[0] in V_set
        items2 = item[1].items()
        for item2 in items2:
            edges.append((item[0], item2[0], item2[1]))
             
    mst, removed_edges = find_mst(V_set, V_set, edges)
    print("#####\n\nTotal runs: ", total_runs, "\n\n#####")
    return mst

def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help='Used for smaller dataset and testing', action='store_true')
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    args = parser.parse_args()

    print('Start generating MST')
    
    conf = SparkConf().setAppName('MST_Algorithm').set("spark.driver.memory", "16g").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)
    
    logger = BenchmarkLogger()

    start_time = datetime.now()
    print('Starting time:', start_time)

    # --- Synthetic Datasets ---
    print("\n--- Processing Synthetic Datasets ---")
    datasets_syn = get_clustering_data()
    names_datasets = ['TwoCircles', 'TwoMoons', 'Varied', 'Aniso', 'Blobs', 'Random', 'swissroll', 'sshape']
    
    num_clusters = [2, 2, 3, 3, 3, 2, 2, 2]
    cnt = 0
    time = []
    file_location = 'my_results/original/'
    plotter = Plotter(None, None, file_location)
    data_reader = DataReader()
    
    for dataset in datasets_syn:
        if cnt < 0:
            cnt += 1
            continue
        dataset_name = names_datasets[cnt]
        timestamp = datetime.now()
        print(f'Start creating Distance Matrix for {dataset_name}...')
        E, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        plotter.set_vertex_coordinates(vertex_coordinates)
        plotter.set_dataset(dataset_name + '_eps' + str(args.epsilon))
        plotter.update_string()
        plotter.reset_round()
        V = list(range(len(vertex_coordinates)))
        print('Size dataset: ', len(vertex_coordinates))
        print('Created distance matrix in: ', datetime.now() - timestamp)
        
        print('Start creating MST...')
        timestamp = datetime.now()
        
        mst = create_mst(sc, V, E, epsilon=args.epsilon, size=size, vertex_coordinates=vertex_coordinates,
                            plot_intermediate=False, plotter=plotter, logger=logger, dataset_name=dataset_name)
                            
        duration = (datetime.now() - timestamp).total_seconds()
        print('Found MST in: ', duration)
        time.append(duration)
        
        logger.log('MST-Fast', dataset_name, args.epsilon, 'Complete', 'Final', len(mst), duration)
        
        print('Start creating plot of MST...')
        timestamp = datetime.now()
        if len(vertex_coordinates[0]) > 2:
            plotter.plot_mst_3d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        else:
            plotter.plot_mst_2d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        print('Created plot of MST in: ', datetime.now() - timestamp)
        cnt += 1

    # --- Custom Datasets ---
    print("\n--- Processing Custom Datasets ---")
    datasets_custom = ['datasets/gplus_combined.txt', 'datasets/twitter_combined.txt', 'datasets/kron_g500-logn16.txt']
    
    for dataset_path in datasets_custom:
        dataset_name = os.path.basename(dataset_path)
        timestamp = datetime.now()
        print(f'Start reading and filtering {dataset_path}...')
        
        vertex_list, size, E = data_reader.read_and_filter_dataset(dataset_path, min_degree=100)
        
        V = vertex_list
        print('Size dataset: ', len(V))
        read_duration = (datetime.now() - timestamp).total_seconds()
        print('Created/Filtered graph in: ', read_duration)
        
        logger.log('MST-Fast', dataset_name, args.epsilon, 'DataLoad', 0, size, read_duration)
        
        print('Start creating MST...')
        timestamp = datetime.now()
        
        mst = create_mst(sc, V, E, epsilon=args.epsilon, size=size, vertex_coordinates=None,
                         plot_intermediate=False, plotter=None, logger=logger, dataset_name=dataset_name)
                         
        duration = (datetime.now() - timestamp).total_seconds()
        print('Found MST in: ', duration)
        time.append(duration)
        
        print(f"MST Size: {len(mst)}")
        logger.log('MST-Fast', dataset_name, args.epsilon, 'Complete', 'Final', len(mst), duration)
        
        print('Skipping plot (no coordinates available)')
        
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
    # Initial call to main function
    main()
