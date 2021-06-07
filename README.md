# COLOGNE

The directory contains code for the COLOGNE suite of algorithms (Coordinated Local Graph Neighborhood Sampling): https://arxiv.org/abs/2102.04770

The code is implemented in Python 3.

One has to run following steps in order to reproduce the results in the paper:

1. Download the graph data for Cora (https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz), Citeseer (https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz), Pubmed(https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz), HomoSapiens, or PPI, (https://snap.stanford.edu/node2vec/Homo_sapiens.mat), Wikipedia (https://snap.stanford.edu/node2vec/POS.mat) and BlogCatalog (https://figshare.com/articles/dataset/BlogCatalog_dataset/11923611). Extract the graphs in the corresponding folders in Graphs.

2. Navigate to src. Preprocess the graphs by running **python preprocess_graph.py --graphname <the name of the graph\>**. The default name is Cora. There is also the option --threshold which specifies how many edges to delete (used for link predictions)
The above code is based on several Jupyter notebooks which are also provided. They can be easier to work with.

3. Generate embeddings by running generate_embeddings running 
**python _generate_embeddings.py --graphname <Cora\> --emb_size <25\> --depth <2\> --sketch_size <10\>**

- emb_size is the embeddings size, i.e., the number of samples, 
- sketch_size is used only for L1 and L2 sampling.

4. Run a notebook for link prediction or node classification. Once we have generated and stored the embeddings for each node, we can run results. Again, create manually the folder 'results' where we will write the results.

5. Optionally, one can plot results using the notebook plot_results.
