The directory contains following folders:

1. random: contains a single file with random numbers from the Marsaglia CD-ROM generator.

2. Graphs: the original publicly available datasets. We create following directories: i) data:  graph edges and node information in a format that is suitable for training. ii) vectors: the sampled nodes and their attributes that define the embedding vectors, and iii)  w2v: continuous vectors trained from the sampled nodes using word2vec.

3. code: A prototype implementation of the COLOGNE algorithms in several jupyter notebooks:
	- preprocess.ipynb: Convert the orginal datset to a format to be used by the algorithms.
	- generate_features.ipynb: The main methods behind COLOGNE for sample generation using 			random walks, minwise independent hashing and L1 sampling.
	- train_embeddings.ipynb: Training contnuous embeddings with word2vec using the 		generated samples in the previous notebook.
	- link_prediction.ipynb
	- clustering.ipynb

   
