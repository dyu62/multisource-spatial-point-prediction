import numpy as np

import networkx as nx
from networkx.utils import UnionFind

from typing import Optional
import torch
from torch import Tensor

from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from math import pi as PI
import torch.nn.functional as F
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
def pos2key(pos):
    pos=pos.reshape(-1)
    key="{:08.4f}".format(pos[0])+'_'+"{:08.4f}".format(pos[1])
    return key
def get_angle(v1: Tensor, v2: Tensor):
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1))
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1))
    return torch.atan2( 
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=-PI, stop=PI, num_gaussians=12):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians) 
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2  
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def triplets(edge_index, num_nodes):
    row, col = edge_index  

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[col] 
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    idx_i = row.repeat_interleave(num_triplets)
    idx_j = col.repeat_interleave(num_triplets)
    edx_1st = value.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    edx_2nd = adj_t_row.storage.value() 
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  
    mask = ~(mask1 | mask2 | mask3) 
    idx_i, idx_j, idx_k, edx_1st, edx_2nd = idx_i[mask], idx_j[mask], idx_k[mask], edx_1st[mask], edx_2nd[mask]
    
    num_triplets_real = torch.cumsum(num_triplets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_triplets, dim=0)-1]

    return torch.stack([idx_i, idx_j, idx_k]), num_triplets_real.to(torch.long), edx_1st, edx_2nd


if __name__ == '__main__':
    1
    
    