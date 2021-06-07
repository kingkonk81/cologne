'''
An implementation of LoNe Samples
'''

import argparse
import numpy as np
import networkx as nx


def parse_args():
	'''
	Parse the necessary arguments.
	'''
	parser = argparse.ArgumentParser(description="Train LoNe sampling embeddings")

	parser.add_argument('--input', nargs='?', default='graph/cora.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 25.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')


	parser.set_defaults(weighted=False)


	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G


	
	return

def main(args):
	'''
	Pipeline for representation learning for all nodes in a graph.
	'''
	nx_G = read_graph()

if __name__ == "__main__":
	args = parse_args()
	main(args)