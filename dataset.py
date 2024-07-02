from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import sys
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
import pickle
import time

from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from typing import Any
def mycollate(data_list):
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None
    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    return data, slices
def myseparate(cls, batch: BaseData, idx: int, slice_dict: Any) -> BaseData:
    data = cls().stores_as(batch)
    # We iterate over each storage object and recursively separate all its attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        attrs = set(batch_store.keys())
        for attr in attrs:
            slices = slice_dict[attr]
            data_store[attr] = _separate(attr, batch_store[attr], idx, slices,
                                         batch, batch_store)
    return data
def _separate(
    key: str,
    value: Any,
    idx: int,
    slices: Any,
    batch: BaseData,
    store: BaseStorage,
) :
        # Narrow a `torch.Tensor` based on `slices`.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = value.narrow(cat_dim or 0, start, end - start)
        return value

def load_point(datasetname="south",k=5,small=[False,50,100]):
    """ 
    load point and build graph pairs
    """
    print("loading")
    time1=time.time()
    if small[0]:
        print("small south dataset k=5")
        datasetname="south"
        k=5
        filename=os.path.join("data",datasetname,datasetname+f'_{k}.pt')
        [data_graphs1,slices_graphs1,data_graphs2,slices_graphs2]=torch.load(filename)
        flattened_list_graphs1 = [myseparate(cls=data_graphs1.__class__, batch=data_graphs1,idx=i,slice_dict=slices_graphs1) for i in range(small[1]*2)]
        flattened_list_graphs2 = [myseparate(cls=data_graphs2.__class__, batch=data_graphs2,idx=i,slice_dict=slices_graphs2) for i in range(small[2]*2)]
        unflattened_list_graphs1= [flattened_list_graphs1[n:n+2] for n in range(0, len(flattened_list_graphs1), 2)]
        unflattened_list_graphs2= [flattened_list_graphs2[n:n+2] for n in range(0, len(flattened_list_graphs2), 2)]
        print(f"Load data used {time.time()-time1:.1f} seconds")
        return unflattened_list_graphs1,unflattened_list_graphs2
    return process(datasetname,k)
def process(datasetname="south",k=5):
    time1=time.time()
    """ 
    build graph pairs
    """
    point_path= os.path.join("data",datasetname,datasetname+".pkl")
    with open(point_path, 'rb') as f:
        data = pickle.load(f)
    graphs1=[]
    graphs2=[]
    for day in data:
        day_d1=day[0]
        day_d2=day[1]
        assert(len(day_d1)<len(day_d2))
        pos1=day_d1[:,-2:]
        edge_index1=knn_graph(pos1,k=k)
        pos2=day_d2[:,-2:]
        edge_index2=knn_graph(pos2,k=k)
        """ 
        iterately mask point in day_d1, the high fidelity data, to build high fidelity graphs, which share the same structure
        """
        for i in range(day_d1.shape[0]):
            day_d1_copy=day_d1.clone().detach()
            target=day_d1[i,0]
            day_d1_copy[i,0]=0
            target_index=torch.tensor(i,dtype=torch.long)
            is_source = torch.ones(day_d1.shape[0] ,dtype=torch.bool)
            is_source[i]=False
            graph1=Data(x=day_d1_copy,pos=pos1,edge_index=edge_index1,target=target[None],target_index=target_index[None],is_source=is_source,datasource=torch.tensor(0,dtype=torch.long)[None])
            """ 
            build pairing low fidelity graphs, which add the masked point in day_d1, so structure is changing
            """            
            day_plus2=torch.cat([day_d1_copy[i][None,:],day_d2])
            pos_plus2=day_plus2[:,-2:]
            edge_index_plus2=knn_graph(pos_plus2,k=k)
            is_source = torch.ones(day_d2.shape[0]+1 ,dtype=torch.bool)
            is_source[0]=False
            graph2=Data(x=day_plus2,pos=pos_plus2,edge_index=edge_index_plus2,target=target[None],target_index=torch.tensor(0,dtype=torch.long)[None],is_source=is_source,datasource=torch.tensor(0,dtype=torch.long)[None])
            graphs1.append([graph1,graph2])
        """ 
        iterately mask point in day_d2, the low fidelity data, to build low fidelity graphs, which share the same structure
        """
        for i in range(day_d2.shape[0]):
            day_d2_copy=day_d2.clone().detach()
            target=day_d2[i,0]
            day_d2_copy[i,0]=0
            target_index=torch.tensor(i,dtype=torch.long)
            is_source = torch.ones(day_d2.shape[0] ,dtype=torch.bool)
            is_source[i]=False
            graph2=Data(x=day_d2_copy,pos=pos2,edge_index=edge_index2,target=target[None],target_index=target_index[None],is_source=is_source,datasource=torch.tensor(1,dtype=torch.long)[None])
            """ 
            build pairing high fidelity graphs, which add the masked point in day_d2, so structure is changing
            """            
            day_plus1=torch.cat([day_d2_copy[i][None,:],day_d1])
            pos_plus1=day_plus1[:,-2:]
            edge_index_plus1=knn_graph(pos_plus1,k=k)
            is_source = torch.ones(day_d1.shape[0]+1 ,dtype=torch.bool)
            is_source[0]=False
            graph1=Data(x=day_plus1,pos=pos_plus1,edge_index=edge_index_plus1,target=target[None],target_index=torch.tensor(0,dtype=torch.long)[None],is_source=is_source,datasource=torch.tensor(1,dtype=torch.long)[None])
            graphs2.append([graph1,graph2])
    np.random.shuffle(graphs1)
    np.random.shuffle(graphs2)
    return [graphs1,graphs2]

class MergeNeighborDataset(torch.utils.data.Dataset):
    """ Customized dataset for each domain"""
    def __init__(self,X):
        self.X = X                           # set data
    def __len__(self):
        return len(self.X)                   # return length
    def __getitem__(self, idx):
        return self.X[idx] 
def kneighbor_point(datasetname="south",k=1,daily=False):
    """ 
    build k neighbor pairing
    """
    ranking_path= os.path.join("data",datasetname,datasetname+"_ranking.pkl")
    with open(ranking_path, 'rb') as f:
        rankings = pickle.load(f)
    point_path= os.path.join("data",datasetname,datasetname+".pkl")
    with open(point_path, 'rb') as f:
        days = pickle.load(f)
    samples=[]
    for i in range(len(days)):
        day_d1=days[i][0]
        day_d2=days[i][1]
        ranking=rankings[i]
        """ 
        iterately get point in day_d1, the high fidelity data, to build samples
        """
        sample1 = []
        for j in range(day_d1.shape[0]):
            point1=day_d1[j]
            point1_neighbors=day_d2[ranking[j,:k]]
            point1_neighbor=torch.mean(point1_neighbors,axis=0)
            sample1.append([point1,point1_neighbor])
        if daily:
            samples.append(sample1)
        else:
            samples.extend(sample1)
    if not daily:
        return [samples]
    return samples

if __name__ == '__main__':
    1
