'''
Preprocessing the different graphs.
Note that graphs are provided in different formats and we transform them into a universal format 
that contains following information:
- nodedata: a hash map contaoining all node information per node: node attributes such as keywords 
and node label(for classification tasks).
- a networkx unweighted, undirected graph.
- a reduced graph where some edges have beeen removed. (Used for link prediction evaluation).

We recommend to look into the individual notebooks which implement the methods here and should be easier to follow.
'''


import os
import sys
import networkx as nx
import pandas as pd
import json
import copy
import numpy as np
import argparse
import random
import scipy.io
from scipy.sparse import csc_matrix

def get_edges_citation(data_dir, graphname):
    edges_df = pd.read_csv(os.path.join(data_dir, graphname.lower() + ".cites"), \
                        sep='\t', header=None, names=["target", "source"])
    edges = []
    for idx, row in edges_df.iterrows():
        edges.append((str(row['target']).strip(), str(row['source']).strip()))
    return edges

def get_edges_mat(mat):
    edges_data = mat['network']
    edges = []
    rows_edges, cols_edges = edges_data.nonzero()
    for r, c in zip(rows_edges, cols_edges):
        edges.append((str(r).strip(), str(c).strip()))
    return edges

def get_nodedata_citation(nodedata_df):

    nodedata = {}
    for idx, row in nodedata_df.iterrows():
        nodedata[str(row['node'])] = ('label=' + row['label'], {})
        if len(nodedata) % 1000 == 0:
            print("processed nodes: {}".format(len(nodedata)))
        for c in nodedata_df.columns:
            if c[0] == 'w':
                if row[c] != 0:
                    nodedata[str(row['node'])][1][c] = 1
    jsonpath = data_dir + "/data/nodedata.json"
    with open(jsonpath, 'w') as outfile:
        json.dump(nodedata, outfile)
    return nodedata

def get_nodedata_mat(data_dir, mat):
    nodedata = {}
    labels = mat['group']
    edges = mat['network']
    rows_labels, cols_labels = labels.nonzero()
    nodelabels = {}
    for r, c in zip(rows_labels, cols_labels):
        nodelabels.setdefault(str(r), "label=")
        nodelabels[str(r)] += str(c) + "="
    for r, label in nodelabels.items():
        rnd = random.random()
        if rnd >= 0.8:
            label = "None" + label
        nodedata[str(r)] = (label, {r:1}) 
    jsonpath = data_dir + "/data/nodedata.json" 
    with open(jsonpath, 'w') as outfile:
        json.dump(nodedata, outfile)
    return nodedata

def write_graph_data(nodedata, data_dir):
    nodes = set()
    labels = set()
    word_indices = set()
    for node, features in nodedata.items():
        nodes.add(node)
        labels.add(features[0])
        for w in features[1]:
            word_indices.add(str(w))
            
    nodes_path = data_dir + "/data/graph_nodes.txt"
    with open(nodes_path, 'w') as outfile:
        for node in nodes:
            outfile.write(str(node) + '\n')
            
    labels_path = data_dir + "/data/labels.txt" 
    with open(labels_path, 'w') as outfile:
        for label in labels:
            outfile.write(label + '\n')
            
    words_path = data_dir + "/data/words_indices.txt" 
    with open(words_path, 'w') as outfile:
        for wi in word_indices:
            outfile.write(wi + '\n')
    return nodedata

def collect_nodes_edges_citation(data_dir, graphname):
    edges_df = pd.read_csv(os.path.join(data_dir, graphname.lower() + ".cites"), \
                        sep='\t', header=None, names=["target", "source"])

    content = pd.read_csv(os.path.join(data_dir, graphname.lower() +".content"), \
                        sep='\t', header=None)

    feature_names = ["w-{}".format(ii) for ii in range(content.shape[1]-2)]
    column_names =  ['node'] + feature_names + ["label"]
    nodedata_df = pd.read_csv(os.path.join(data_dir, graphname.lower() +".content"), \
                            sep='\t',  names=column_names)

    nodedata_df[['node', 'label']].to_csv(data_dir + '/data/nodes_with_labels.csv', index=False)

    nodedata = get_nodedata_citation(nodedata_df)
    edges = get_edges_citation(data_dir, graphname)
    return nodedata, edges

def collect_nodes_edges_pubmed(data_dir, graphname):
    nodes = set()
    edges = []
    with open(os.path.join(data_dir, graphname.lower() + ".cites"), 'r') as edgefile:
        for line in edgefile:
            line_split = line.split('|')
            if len(line_split) > 1:
                l0 = line_split[0]
                l1 = line_split[1]
                u = l0.split(':')[1]
                v = l1.split(':')[1]
                nodes.add(str(u).strip())
                nodes.add(str(v).strip())
                edges.append((str(u).strip(), str(v).strip()))
    nodedata = {}
    with open(os.path.join(data_dir, graphname.lower() + ".content"), 'r') as contentfile:
        for line in contentfile:
            line_split = line.split()
            if len(line_split) < 3:
                continue
            if line_split[0] not in nodes:
                continue
            nodewords = {}
            for i in range(2, len(line_split)):
                w = line_split[i]
                w_split = w.split('=')
                if w_split[0] == 'summary':
                    continue
                nodewords[w.split('=')[0]] = float(w.split('=')[1])
            nodedata[line_split[0]] = (line_split[1], nodewords)
    jsonpath = data_dir + "/data/nodedata.json" 
    with open(jsonpath, 'w') as outfile:
        json.dump(nodedata, outfile)

    return nodedata, edges

def collect_nodes_edges_mat(data_dir, graphname):
    data_dir = os.path.expanduser("../Graphs/" + graphname)
    mat = scipy.io.loadmat(data_dir + '/graph.mat')
    nodedata = get_nodedata_mat(data_dir, mat)
    edges = get_edges_mat(mat)
    return nodedata, edges

def create_graph(edges, nodedata):
    G = nx.Graph()
    for edge in edges:
        u = edge[0]
        v = edge[1]
        if u in nodedata and v in nodedata:
            G.add_edge(u, v)
    return G


def remove_edges(G, threshold):
    removed_edges = set()
    H = copy.deepcopy(G)
    while len(removed_edges) < threshold*G.number_of_edges():
        if len(removed_edges)%1000 == 0:
            print("removed edges: {}, \ntotal number of edges: {}".format(len(removed_edges), G.number_of_edges()))
        i = np.random.randint(low=0, high=H.number_of_edges())
        edge = list(H.edges())[i]
        u = edge[0]
        v = edge[1]
        if H.degree[u] > 1 and H.degree[v] > 1:
            H.remove_edge(u, v)
            removed_edges.add((u, v))
    return H, removed_edges

def write_edges_to_file(G, H, removed_edges, data_dir):
    edges_path = data_dir + "/data/all_graph_edges.txt" 
    with open(edges_path, 'w') as outfile:
        for edge in G.edges():
            outfile.write(edge[0] + ':' + edge[1] + '\n')

    edges_path = data_dir + "/data/graph_edges_reduced.txt"
    with open(edges_path, 'w') as outfile:
        for edge in H.edges():
            outfile.write(edge[0] + ':' + edge[1] + '\n')

    if len(removed_edges) > 0:
        removed_edges_path = data_dir + "/data/removed_edges.txt" 
        with open(removed_edges_path, 'w') as outfile:
            for edge in removed_edges:
                outfile.write(edge[0] + ':' + edge[1] + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoNe sampling embeddings")
    parser.add_argument('--graphname', nargs='?', default='Cora', help='Input graph name')
    parser.add_argument('--threshold', nargs='?', default='0.2', help='Input graph name')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    graphname = args.graphname 
    try:
        threshold = float(args.threshold)
    except:
        raise Exception("The threshold must be a float number in [0, 1]")
    if threshold > 1 or threshold < 0:
        raise Exception("The threshold must be a float number in [0, 1]")

    if graphname not in ['Cora', 'Citeseer', 'Pubmed', 'PPI', 'Wikipedia', 'BlogCatalog']:
        raise Exception("Supported graphs are Cora, Citeseer, Pubmed, PPI, Wikipedia and BlogCatalog")
    if graphname == 'BlogCatalog':
        print("Not yet implemented. Please run the notebook preprocess_blogcatalog.ipynb.")
        sys.exit(0)
    data_dir = os.path.expanduser("../Graphs/" + graphname)
    if not os.path.exists(data_dir):
        raise Exception("No such graph directory")
    if not os.path.exists(data_dir + "/data/"):
        os.makedirs(data_dir + "/data/")

    if graphname in ['Cora', 'Citeseer']:
        nodedata, edges = collect_nodes_edges_citation(data_dir, graphname)
        print(len(edges))
    elif graphname == 'Pubmed':
        nodedata, edges = collect_nodes_edges_pubmed(data_dir, graphname)
    elif graphname in ['Wikipedia', 'PPI']:
        nodedata, edges = collect_nodes_edges_mat(data_dir, graphname)
    else:
        raise Exception("Supported graphs are Cora, Citeseer, Pubmed, PPI, Wikipedia and BlogCatalog")
    G = create_graph(edges, nodedata)
    H, removed_edges = remove_edges(G, threshold)
    write_edges_to_file(G, H, removed_edges, data_dir)