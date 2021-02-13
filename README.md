# COLOGNE

The directory contains code for the COLOGNE suite of algorithms (Coordinated Local Graph Neighborhood Sampling): https://arxiv.org/abs/2102.04770

The code is organized in several notebooks.

One has to run following steps:
1. Download the graph data for Cora (https://graphsandnetworks.com/the-cora-dataset/), Citeseer (http://networkrepository.com/citeseer.php), Pubmed(https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz), HomoSapiens, or PPI, (https://snap.stanford.edu/node2vec/Homo_sapiens.mat), Wikipedia (https://snap.stanford.edu/node2vec/POS.mat) and BlogCatalog (https://figshare.com/articles/dataset/BlogCatalog_dataset/11923611). Extract the graphs in the corresponding folders in Graphs.

2. Prepprocess the graphs by running the code in the corresponding notebooks (some paths might need to be adjusted).

3. Generate embeddings by running generate_geatures.ipynb

4. Run a notebook for link prediction or node classification.

5. You can also plot some results in the plot results notebook.
