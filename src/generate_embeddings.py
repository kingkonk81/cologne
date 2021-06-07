'''
Sampling nodes from the k-hop neighbohood and generating embeddings of
given dimensionality.
'''


import os
import sys
import networkx as nx
import pandas as pd
import json
import random
import math
import copy
import time
import argparse

def read_graph_from_edge_list(filename, data_dir):
    with open(data_dir + "/data/nodedata.json", "r") as read_file:
        nodedata = json.load(read_file)
    G = nx.Graph()
    path = data_dir + "/data/" + filename
    cnt = 0
    with open(path, 'r') as edgefile: 
        for line in edgefile:
            cnt += 1
            line_split = line.split(':')
            if len(line_split) > 1:
                l0 = line_split[0]
                l1 = line_split[1]
                u = l0.strip()
                v = l1.strip()
                if u in nodedata and v in nodedata:
                    G.add_edge(u, v)
        
    print("Number of nodes: {} \nNumber of edges: {}".format(G.number_of_nodes(), G.number_of_edges()))
    return G

def get_labels(label):
    if label[:4] == 'None':
        return []
    else:
        labels = label.split('=')
        return labels[1:len(labels)-1]

# generate a random int in [start, end]
def get_rnd_int_in_range(start, end):
    r = random.randint(0, 1e10)
    diff = end - start + 1
    rval = r%diff
    return rval+start

# generate a random value in (min_val, 1]
def get_rnd_value(min_val):
    if min_val >=1:
        raise Exception("Minimum must be less than 1")
    rval = 0
    while rval < min_val:
        rval = random.random()
    return rval

# a standard random walk starting from a node for 'depth' hops 
# sample a feature(attribute) for each node at random
def random_walk(G, node, depth, features):
    node = str(node)
    cnt = 0
    curr_node = node
    while cnt < depth and G.degree[curr_node] > 0:
        nbrs = [curr_node] + [nbr for nbr in G.neighbors(curr_node)]
        curr_node = nbrs[get_rnd_int_in_range(0, len(nbrs)-1)]
        cnt += 1
    subject, features_node = features[curr_node]
    # return a random feature describing the node
    if len(features_node.values())==0:
        print(features[curr_node])
    random.seed(get_rnd_int_in_range(0, len(G.nodes())))    
    w = random.choices(population=list(features_node.keys()), weights=list(features_node.values()), k=1)[0]
    return curr_node, subject, w

def get_nodedata(data_dir):
    nodedata_path = data_dir + "/data/nodedata.json" 
    nodedata={}
    with open(nodedata_path, "r") as read_file:
        nodedata = json.load(read_file)
    return nodedata

# ********************** Random Walk functions ****************************

# for each node generate a number of samples, i.e. the embedding size, by random walks
def all_nodes_random_walk(G, depth, nr_walks, features):
    vectors = {}
    for node in G.nodes():
        vectors[node] = [None for _ in range(nr_walks)]
        for walk in range(nr_walks):
            sample, subject, feature = random_walk(G, node, depth, features)
            vectors[node][walk] = (sample, subject, feature)
    return vectors

def generate_random_walks_neiborhood(G, emb_size, depth):
    # random walks on the full graph
    vectors_rw_all = []
    for d in range(depth+1):
        vectors_rw = all_nodes_random_walk(G, d, emb_size, features=nodedata)
        vectors_rw_all.append(vectors_rw)
    return vectors_rw_all


# ********************** NodeSketch functions ****************************

def update_dict(d, k, min_val, seed):
    random.seed(seed)
    if k not in d:
        d[k] = random.random() 

def ioffe_sampling(arr, weights):
    min_val = 1e6
    node_sample = None
    feature_sample = None
    weight_sample = 0
    label_sample = None
    for node, vals in arr.items():
        feature, weight, label = vals[0], vals[1], vals[2]
        rnd_val = -math.log(weights[node])/weight # consider feature weights, by default all weights are equal
        if rnd_val < min_val:
            min_val = rnd_val
            node_sample = node
            feature_sample = feature
            weight_sample = weight
            label_sample = label
    return node_sample, feature_sample, weight_sample, label_sample

def update_arr(arr, new_node):
    if new_node[0] in arr:
        arr[new_node[0]] = (new_node[1], arr[new_node[0]][1] + new_node[2], new_node[3])
    else:
        arr[new_node[0]] = (new_node[1], new_node[2], new_node[3])
    return arr


def nodesketch_iter(G, nodedata, depth, emb_size):
    
    min_val = 1e-6
    feats_rnd = [{} for _ in range(emb_size)]
    cnt = 0
    for i in range(emb_size):
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            for f, weight_f in feats[1].items():
                cnt += 1
                update_dict(feats_rnd_i, f, min_val, seed=13*cnt)
                
    node_labels = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        node_labels_i = node_labels[i]
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            arr = {}
            for f, weight_f in feats[1].items():
                arr[f] = (f, weight_f, feats[0])
            _, feature_sample, weight_sample, label_sample = ioffe_sampling(arr, feats_rnd_i)
            node_labels_i[node] = (node, feature_sample, weight_sample, label_sample)
    
    node_rnd_vals_all = [{} for _ in range(emb_size)]
    for t in range(emb_size):
        random.seed(1223*t)
        for u in G.nodes():
            node_rnd_vals_all[t][u] = random.random()
            
    node_labels_all = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    node_labels_all[0] = node_labels
    for d in range(depth):
        node_labels_iter = node_labels_all[d]
        random.seed(31*d)
        
        for t in range(emb_size):
            node_labels_iter_t = node_labels_iter[t]
            for u in G.nodes():
                node_sample_u, feature_sample_u, weight_sample_u, label_u = node_labels_iter_t[u]
                arr_u = {node_sample_u: (feature_sample_u, weight_sample_u, label_u)} 
                for v in G.neighbors(u):
                    node_sample_v, feature_sample_v, weight_sample_v, label_v = node_labels_iter_t[v]
                    update_arr(arr_u, (node_sample_v, feature_sample_v, weight_sample_v, label_v))
                node_labels_all[d+1][t][u] = ioffe_sampling(arr_u, node_rnd_vals_all[t]) 
                
    node_embeddings = [{n:[] for n in G.nodes()} for _ in range(depth+1)]
    for d in range(depth+1):
        for u in G.nodes():
            for nl in node_labels_all[d]:
                node_embeddings[d][u].append((nl[u][0], nl[u][3], nl[u][1]))
    return node_embeddings


# ********************** L0 sampling functions ****************************   

# initialize random numbers for nodes and features for each embedding 
def init_dicts_l0(nodedata, emb_size):
    min_val = 1e-6
    cnt = 0
    feats_rnd = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            for f, weight_f in feats[1].items():
                cnt += 1
                update_dict(feats_rnd_i, f, min_val, seed=17*cnt)
    return feats_rnd 


def generate_minwise_samples(G, nodedata, depth, emb_size):
    feats_rnd = init_dicts_l0(nodedata, emb_size)
    node_labels = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        node_labels_i = node_labels[i]
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            min_feature_value = 1e3
            min_feature = None
            for f in feats[1]:
                if feats_rnd_i[f] < min_feature_value:
                    min_feature = f
                    min_feature_value = feats_rnd_i[f]
            node_labels_i[node] = (min_feature_value, node, feats[0], min_feature)
            
    node_labels_all = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    node_labels_all[0] = node_labels
    for d in range(depth):
        node_labels_iter = node_labels_all[d]
        for u in G.nodes():
            for t in range(emb_size):
                w_u = node_labels_iter[t][u]
                for v in G.neighbors(u):
                        w_u = min(node_labels_iter[t][v], w_u)
                node_labels_all[d+1][t][u] = w_u
            
    node_embeddings = [{n:[] for n in G.nodes()} for _ in range(depth+1)]
    for d in range(depth+1):
        for u in G.nodes():
            for nl in node_labels_all[d]:
                node_embeddings[d][u].append((nl[u][1], nl[u][2], nl[u][3]))
    return node_embeddings



# ********************** L1 sampling functions ****************************

# generating L1 samples from the k-hop neighborhood
# top is the summary size of the frequent items mining algorithm
def generate_L1_samples(G, nodedata, depth, emb_size, top):
    
    min_val = 1e-6
    feats_rnd = [{} for _ in range(emb_size)]
    cnt = 0
    for i in range(emb_size):
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            for f, weight_f in feats[1].items():
                cnt += 1
                update_dict(feats_rnd_i, f, min_val, seed=23*cnt)
    
    sketches = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    
    min_val = 1e-6
    cnt = 0
    # generate the random values for each node (attribute)
    for u, (subject, features) in nodedata.items():
        cnt += 1
        # if cnt % 1000 == 0:
        #     print('nodes processed', cnt)
        for i in range(emb_size):
            feats_rnd_i = feats_rnd[i]
            sketches[0][i][u] = {}
            max_w = 0
            max_f = None
            for f, w_f in features.items():
                w_rnd = w_f/feats_rnd_i[f]   # get_rnd_value(rand_gen, min_val) 
                if w_rnd > max_w:
                    max_w = w_rnd
                    max_f = f
            if max_w > 0:
                sketches[0][i][u] = {u : (max_w, max_f)}
    
    # iterate over neighborhoods and maintain the heaviest nodes
    for d in range(depth):
        # print('ITERATION', d)
        for emb in range(emb_size):
            sketch_iter_emb = copy.deepcopy(sketches[d][emb])
            new_sketches = {}
            for u in G.nodes():
                if u not in sketch_iter_emb:
                    continue
                sketch_u = copy.deepcopy(sketch_iter_emb[u])
                for v in G.neighbors(u):
                    sketch_v = sketch_iter_emb[v]
                    for t, (w_f, f) in sketch_v.items():
                        sketch_u.setdefault(t, (0, None))
                        weight = sketch_u[t][0] + w_f
                        sketch_u[t] = (weight, f)
                triples = []
                for node, feat_node in sketch_u.items():
                    triples.append((feat_node[0], feat_node[1], node))
                
                # mining heavy hitters using the Frequent algorithm
                triples = sorted(triples, reverse=True)
                to_subtract = 0
                if len(triples) > top:
                    to_subtract = triples[top][0]
                top_triples = triples[:top]
                
                new_sketches[u] = {tr[2] : (tr[0]-to_subtract, tr[1]) for tr in top_triples}
            sketches[d+1][emb] = new_sketches
    return sketches

# ********************** L2 sampling functions ****************************

# generating L2 samples from the k-hop neighborhood
# top is the summary size of the frequent items mining algorithm
def generate_L2_samples(G, nodedata, depth, emb_size, top):

    min_val = 1e-6
    cnt = 0
    feats_rnd = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            for f, weight_f in feats[1].items():
                cnt += 1
                update_dict(feats_rnd_i, f, min_val, seed=cnt)
    
    sketches = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    
    min_val = 1e-6
    cnt = 0
    # generate the random values for each node (attribute)
    for u, (subject, features) in nodedata.items():
        cnt += 1
        # if cnt % 1000 == 0:
        #     print('nodes processed', cnt)
        for i in range(emb_size):
            feats_rnd_i = feats_rnd[i]
            sketches[0][i][u] = {}
            max_w = 0
            max_f = None
            for f, w_f in features.items():
                w_rnd = w_f/math.sqrt(feats_rnd_i[f])   
                if w_rnd > max_w:
                    max_w = w_rnd
                    max_f = f
            if max_w > 0:
                sketches[0][i][u] = {u : (max_w, max_f)}
    
    # iterate over neighborhoods and maintain the heaviest nodes
    for d in range(depth):
        # print('ITERATION', d)
        for emb in range(emb_size):
            sketch_iter_emb = copy.deepcopy(sketches[d][emb])
            new_sketches = {}
            for u in G.nodes():
                if u not in sketch_iter_emb:
                    continue
                sketch_u = copy.deepcopy(sketch_iter_emb[u])
                for v in G.neighbors(u):
                    sketch_v = sketch_iter_emb[v]
                    for t, (w_f, f) in sketch_v.items():
                        sketch_u.setdefault(t, (0, None))
                        weight = sketch_u[t][0] + w_f
                        sketch_u[t] = (weight, f)
                triples = []
                for node, feat_node in sketch_u.items():
                    triples.append((feat_node[0], feat_node[1], node))
                top_triples = sorted(triples, reverse=True)[:top]
                new_sketches[u] = {tr[2] : (tr[0], tr[1]) for tr in top_triples}
            sketches[d+1][emb] = new_sketches
    return sketches


def write_vectors_to_file(vectors, data_dir, filename, emb_size):
    if not os.path.exists(data_dir + "/vectors/"):
        os.makedirs(data_dir + "/vectors/")
    for d in range(len(vectors)):
        jsonpath = data_dir + "/vectors/" + filename + str(emb_size) + "_hop_" + str(d) + ".json"
        with open(jsonpath, 'w') as outfile:
            json.dump(vectors[d], outfile)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LoNe sampling embeddings")
    parser.add_argument('--graphname', nargs='?', default='Cora', help='Input graph name')
    parser.add_argument('--emb_size', nargs='?', default='25', help='The dimensionality of the embeddings')
    parser.add_argument('--depth', nargs='?', default='2', help='The depth of the neiborhood')
    parser.add_argument('--sketch_size', nargs='?', default='10', help='The sketch size for L0 and L1 sampling')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    print(args)
    graphname = args.graphname 
    data_dir = os.path.expanduser("../Graphs/"+graphname)
    if not os.path.exists(data_dir):
        raise Exception("No such graph directory")
    try:
        emb_size = int(args.emb_size)
    except:
        raise Exception("Embedding size must be a positive integer number")
    if emb_size < 1:
        raise Exception("Embedding size must be a positive integer number")
    try:
        depth = int(args.depth)
    except:
        raise Exception("Depth must be a positive integer number")
    if depth < 1:
        raise Exception("Depth must be a positive integer number")
    try:
        top = int(args.sketch_size)
    except:
        raise Exception("Sketch size must be a positive integer number")
    if top < 1:
        raise Exception("Sketch size must be a positive integer number")

    G = read_graph_from_edge_list("all_graph_edges.txt", data_dir)
    nodedata = get_nodedata(data_dir)

    # Random walk sampling
    start = time.time()
    vectors_rw = generate_random_walks_neiborhood(G, emb_size, depth)
    print('Elapsed time Random Walks', time.time()-start)
    write_vectors_to_file(vectors_rw, data_dir, "vectors_rw_all_", emb_size)

    # NodeSketch embeddings
    start = time.time()
    vectors_ns = nodesketch_iter(G, nodedata, depth=depth, emb_size=emb_size)
    print('Elapsed time Nodesketch', time.time()-start)
    write_vectors_to_file(vectors_ns, data_dir, "vectors_ns_all_", emb_size)


    # L0 sampling
    start = time.time() 
    vectors_l0 = generate_minwise_samples(G, nodedata, depth=depth, emb_size=emb_size)
    end = time.time()
    print('Elapsed time L0', time.time()-start)
    write_vectors_to_file(vectors_l0, data_dir, "vectors_l0_all_", emb_size)

    # L1 sampling
    start = time.time()
    vectors_l1 = generate_L1_samples(G, nodedata, depth=depth, emb_size=emb_size, top=top)
    print('Elapsed time L1', time.time()-start)
    write_vectors_to_file(vectors_l1, data_dir, "vectors_l1_all_", emb_size)

    # L2 sampling
    start = time.time()
    vectors_l2 = generate_L2_samples(G, nodedata, depth=depth, emb_size=emb_size, top=top)
    print('Elapsed time L2', time.time()-start)
    write_vectors_to_file(vectors_l2, data_dir, "vectors_l2_all_", emb_size)