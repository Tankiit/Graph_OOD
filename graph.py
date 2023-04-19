import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
#from torch_geometric.datasets import Planetoid
#import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
#from torch_geometric.utils import to_networkx
import torch_geometric as tg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.utils import to_undirected
#import numba as nb

import pdb
#import faiss



def create_graph(features,k=5):
    from sklearn.neighbors import kneighbors_graph
    graph = kneighbors_graph(features, k, mode='connectivity', include_self=True)
    return graph


if __name__=="__main__":
    id_data=np.load('/disk/tanmoy/Out_of_Distribution_Detection/nonparametric_density_estimation/cache/CIFAR-10_resnet18-cifar_in_last_layer.npy')
    ood_data=np.load('/disk/tanmoy/Out_of_Distribution_Detection/nonparametric_density_estimation/cache/SVHN_resnet18-cifar_in_last_layer.npy')
   
    id_feats=id_data[:,:512]
    ood_feats=ood_data[:,:512]

    id_labels=id_data[:,512]
    ood_labels=ood_data[:,512]

    idata=create_graph(id_feats,k=5)
    #pdb.set_trace()
    odata=create_graph(ood_feats,k=5)

    #id_edge_index = torch.tensor(id_adj_matrix.nonzero(), dtype=torch.long)
    #ood_edge_index= torch.tensor(ood_adj_matrix.nonzero(), dtype=torch.long)

    #idata = tg.data.Data(edge_index=id_edge_index.t().contiguous())
    #oodata = tg.data.Data(edge_index=ood_edge_index.t().contiguous())

    # Convert the graph to an undirected graph
    #idata = to_undirected(idata)
    #oodata = to_undirected(oodata)

    
