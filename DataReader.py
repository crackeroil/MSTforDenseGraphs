import json
import csv
import random
import numpy as np
import scipy
import pandas as pd


def get_edge_weight(num_clusters=5, sigma=1, mu=None):
    if mu is None:
        mu = [5, 10, 15, 20, 25]
    edge_cluster = np.ceil(random.uniform(0, num_clusters))
    return np.random.normal(mu[edge_cluster], sigma)


class DataReader:

    def __init__(self):
        pass

    def read_json(self, file_loc):
        with open(file_loc) as file:
            data = json.load(file)

            x = data['x']
            y = data['y']
            edge_start = data['edge_i']
            edge_end = data['edge_j']

            vertex_coordinates = []
            vertices = []
            for i in range(data['n']):
                vertices.append(i)
                vertex_coordinates.append([x[i], y[i]])

            edges = dict()
            for i in range(data['m']):
                if edge_start[i] not in edges:
                    edges[edge_start[i]] = {edge_end[i]: np.sqrt((y[edge_start[i]] - y[edge_end[i]]) ** 2 +
                                                                 (x[edge_start[i]] - x[edge_end[i]]) ** 2)}
                else:
                    edges[edge_start[i]][edge_end[i]] = np.sqrt((y[edge_start[i]] - y[edge_end[i]]) ** 2 +
                                                                (x[edge_start[i]] - x[edge_end[i]]) ** 2)

        return vertices, data['m'], edges, vertex_coordinates

    def read_data_set_from_txtfile(self, file_location, edge_weights=False, distribution='zipf'):
        edges = {}
        vertices = set()
        size = 0
        with open(file_location) as file:
            for row in file:
                if row.startswith("#"):
                    continue
                stripped_row = row.strip()
                splitted_row = stripped_row.split("\t")
                # splitted_row = stripped_row.split(' ')
                vertex1 = int(splitted_row[0])
                vertex2 = int(splitted_row[1])
                vertices.add(vertex1)
                vertices.add(vertex2)
                if edge_weights:
                    if vertex1 < vertex2:
                        if vertex1 in edges:
                            edges[vertex1][vertex2] = float(splitted_row[2])
                            size += 1
                        else:
                            edges[vertex1] = {vertex2: float(splitted_row[2])}
                            size += 1
                    else:
                        if vertex2 in edges:
                            edges[vertex2][vertex1] = float(splitted_row[2])
                            size += 1
                        else:
                            edges[vertex2] = {vertex1: float(splitted_row[2])}
                            size += 1
                else:
                    if distribution == 'Gaussian':
                        edge_weight = get_edge_weight()
                    elif distribution == 'zipf':
                        edge_weight = np.random.zipf(2)
                    else:
                        edge_weight = random.uniform(0, 1)
                    if vertex1 < vertex2:
                        if vertex1 in edges:
                            if vertex2 not in edges[vertex1]:
                                edges[vertex1][vertex2] = edge_weight
                                size += 1
                        else:
                            edges[vertex1] = {vertex2: edge_weight}
                            size += 1
                    else:
                        if vertex2 in edges:
                            if vertex1 not in edges[vertex2]:
                                edges[vertex2][vertex1] = edge_weight
                                size += 1
                        else:
                            edges[vertex2] = {vertex1: edge_weight}
                            size += 1
        vertex_list = []
        for vertex in vertices:
            vertex_list.append(vertex)
        file.close()
        return vertex_list, size, edges

    def read_data_set_from_csvfile(self, file_location):
        edges = {}
        vertices = set()
        size = 0
        with open(file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # line_count = 0
            for row in csv_reader:
                vertex1 = row[0]
                vertex2 = row[1]
                vertices.add(vertex1)
                vertices.add(vertex2)
                if vertex1 < vertex2:
                    if vertex1 in edges:
                        edges[vertex1][vertex2] = row[2]
                        size += 1
                    else:
                        edges[vertex1] = {vertex2: row[2]}
                        size += 1
                else:
                    if vertex2 in edges:
                        edges[vertex2][vertex1] = row[2]
                        size += 1
                    else:
                        edges[vertex2] = {vertex1: row[2]}
                        size += 1
        vertex_list = []
        for vertex in vertices:
            vertex_list.append(vertex)
        return vertex_list, size, edges

    def create_distance_matrix(self, dataset, full_dm=False):
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
            max_weight = 0
            dict = {}
            for i in range(len(dataset)):
                dict2 = {}
                for j in range(i + 1, len(dataset)):
                    value = np.sqrt(np.sum(np.square(dataset[i] - dataset[j])))
                    max_weight = max(value, max_weight)
                    dict2[j] = value
                    size += 1
                dict[i] = dict2
        else:
            d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
            dict = {}
            max_weight = 0
            # Run with less edges
            for i in range(len(d_matrix)):
                dict2 = {}
                if full_dm:
                    for j in range(len(d_matrix)):
                        if i != j:
                            size += 1
                            max_weight = max(d_matrix[i][j], max_weight)
                            dict2[j] = d_matrix[i][j]
                    dict[i] = dict2
                else:
                    for j in range(i, len(d_matrix)):
                        if i != j:
                            size += 1
                            max_weight = max(d_matrix[i][j], max_weight)
                            dict2[j] = d_matrix[i][j]
                    dict[i] = dict2
        return dict, size, vertices, max_weight

    def read_vertex_list(self, file_location):
        vertices = []
        with open(file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                vertices.append((float(row[0]), float(row[1])))
        return vertices, len(vertices)

        return V, len(V)

    def read_and_filter_dataset(self, file_location, min_degree=100, dataset_name=None):
        """
        Reads a dataset using a 2-pass approach to filter vertices by degree.
        Pass 1: Count degrees.
        Pass 2: Store edges where both vertices have degree > min_degree.
        """
        print(f"Reading {file_location} (Pass 1: Counting degrees)...")
        degree_count = {}
        
        # Pass 1: Count degrees
        with open(file_location, 'r') as f:
            for line in f:
                parts = line.strip().split()
                u, v = parts[0], parts[1]
                degree_count[u] = degree_count.get(u, 0) + 1
                degree_count[v] = degree_count.get(v, 0) + 1

        print(f"Pass 1 complete. Found {len(degree_count)} unique vertices.")
        
        # Pass 2: Filter and Build Graph
        print(f"Reading {file_location} (Pass 2: Filtering and Building Graph)...")
        edges = {}
        vertices = set() 
        with open(file_location, 'r') as f:
            for line in f:
                parts = line.strip().split()
                u, v = parts[0], parts[1]
                if degree_count.get(u, 0) > min_degree and degree_count.get(v, 0) > min_degree:
                    w = random.uniform(0, 1) # can change to different distribution
                    vertices.add(u)
                    vertices.add(v)
                    
                    # Store only one direction (u < v) to match create_distance_matrix behavior
                    # and avoid duplicate edges in processing
                    p1, p2 = (u, v) if u < v else (v, u)
                    if p1 not in edges:
                        edges[p1] = {}
                    edges[p1][p2] = w
        size = sum(len(neighbors) for neighbors in edges.values())
        vertex_list = list(vertices)
        print(f"Pass 2 complete. Filtered Graph: |V|={len(vertex_list)}, |E|={size}")
        
        return vertex_list, size, edges