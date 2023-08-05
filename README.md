# PSGL

Both the structural and position information play important roles in node embedding. The classic GNNs in the literature could not tackle these two kinds of information well. To address this issue, we propose a novel PSGL framework to jointly learn the local structure and the global position information.

Please cite our paper, if you use our source code.

"Position and Structure-aware Graph Learning"
# Code Structure
There are two code folders:

Link Prediction and Pairwise Node Classification (src_PSGL)

Node Classification (src_PSGL_NC)
 
# Experimental Setup
Python 3.7.6
Pytorch 1.4.0
NetworkX 2.3
Cuda 10.0
Libraries Required
Pytorch
PyTorch Geometric
torch-scatter
torch-sparse
torch-cluster
networkx
tensorboardx

# USAGE
The models can be run using the shell file in respective folders. USAGE: ./cmds.sh

# Data
The data used for the experiments are all from publicly available datasets.  They can be downloaded from the following link. https://github.com/JiaxuanYou/P-GNN/tree/master/data
