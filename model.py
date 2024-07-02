import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
# from dataset import 
from torch_geometric.nn.inits import glorot, zeros

from torch_scatter import scatter
from utils.utils import triplets,get_angle,GaussianSmearing
from torch.nn import ModuleList
from math import pi as PI
import math

"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""
def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list
class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", ffn = None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


        self.input_embed_dim = self.cal_input_dim()
        self.ffn = ffn
        
    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)


    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        elif type(coords)  == torch.Tensor:
            assert self.coord_dim == (coords.shape)[2]
            coords=coords.detach().cpu().numpy()
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds
    
        
    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds) 
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds
theoryencoder=TheoryGridCellSpatialRelationEncoder(spa_embed_dim=8)

class GFusion(nn.Module):
    def __init__(self,  h_channel=16,input_featuresize=32,localdepth=2,num_interactions=3,finaldepth=3,num_of_datasources=2,share=True,batchnorm="False"):
        super(GFusion,self).__init__()
        self.training=True
        self.h_channel = h_channel
        self.input_featuresize=input_featuresize
        self.localdepth = localdepth
        self.num_interactions=num_interactions
        self.finaldepth=finaldepth
        self.batchnorm = batchnorm        
        self.activation=nn.ReLU()

        num_gaussians=(1,12)
        self.theta_expansion = GaussianSmearing(-PI, PI, num_gaussians[1])
        self.mlps_list = ModuleList()
        if int(share[0])==1:
            mlp_geo = ModuleList()
            for i in range(self.localdepth):
                if i == 0:
                    mlp_geo.append(Linear(sum(num_gaussians), h_channel))
                else:
                    mlp_geo.append(Linear(h_channel, h_channel))
                if self.batchnorm == "True":
                    mlp_geo.append(nn.BatchNorm1d(h_channel))
                mlp_geo.append(self.activation)            
            for i in range(num_of_datasources):
                self.mlps_list.append(mlp_geo)
        else:
            for i in range(num_of_datasources):
                mlp_geo = ModuleList()
                for i in range(self.localdepth):
                    if i == 0:
                        mlp_geo.append(Linear(sum(num_gaussians), h_channel))
                    else:
                        mlp_geo.append(Linear(h_channel, h_channel))
                    if self.batchnorm == "True":
                        mlp_geo.append(nn.BatchNorm1d(h_channel))
                    mlp_geo.append(self.activation)
                self.mlps_list.append(mlp_geo)         
        self.mlps_list_backup = ModuleList()
        for i in range(num_of_datasources):
            mlp_geo = ModuleList()
            for i in range(self.localdepth):
                if i == 0:
                    mlp_geo.append(Linear(4, h_channel)) # for FN version
                else:
                    mlp_geo.append(Linear(h_channel, h_channel))
                if self.batchnorm == "True":
                    mlp_geo.append(nn.BatchNorm1d(h_channel))
                mlp_geo.append(self.activation)
            self.mlps_list_backup.append(mlp_geo)            
        self.translinear=Linear(input_featuresize+1, self.h_channel)
        self.interactions_list = ModuleList()
        if int(share[1])==1:
            interactions= ModuleList()
            for i in range(self.num_interactions):
                block = SPNN(
                    in_ch=self.input_featuresize,
                    hidden_channels=self.h_channel,
                    activation=self.activation,
                    finaldepth=self.finaldepth,
                    batchnorm=self.batchnorm,
                    num_input_geofeature=self.h_channel
                )
                interactions.append(block)
            for i in range(num_of_datasources):
                self.interactions_list.append(interactions)
        else:          
            for i in range(num_of_datasources):
                interactions= ModuleList()
                for i in range(self.num_interactions):
                    block = SPNN(
                        in_ch=self.input_featuresize,
                        hidden_channels=self.h_channel,
                        activation=self.activation,
                        finaldepth=self.finaldepth,
                        batchnorm=self.batchnorm,
                        num_input_geofeature=self.h_channel
                    )
                    interactions.append(block)
                self.interactions_list.append(interactions)          
        self.finalMLP_list = ModuleList()
        if int(share[2])==1:
            finalMLP=ModuleList()
            for i in range(self.finaldepth + 1):
                finalMLP.append(Linear(self.h_channel, self.h_channel))  
                if self.batchnorm == "True":
                    finalMLP.append(nn.BatchNorm1d(self.h_channel))
                finalMLP.append(self.activation)
            finalMLP.append(Linear(self.h_channel, 1))
            for i in range(num_of_datasources):
                self.finalMLP_list.append(finalMLP)
        else:
            for i in range(num_of_datasources):
                finalMLP=ModuleList()
                for i in range(self.finaldepth + 1):
                    finalMLP.append(Linear(self.h_channel, self.h_channel))  
                    if self.batchnorm == "True":
                        finalMLP.append(nn.BatchNorm1d(self.h_channel))
                    finalMLP.append(self.activation)
                finalMLP.append(Linear(self.h_channel, 1))
                self.finalMLP_list.append(finalMLP)               
        self.reset_parameters()
    def reset_parameters(self):
        for i in range(len(self.mlps_list)):
            for lin in self.mlps_list[i]:
                if isinstance(lin, Linear):
                    torch.nn.init.xavier_uniform_(lin.weight)
                    lin.bias.data.fill_(0)
        for i in range(len(self.interactions_list)):
            for block in self.interactions_list[i]:
                block.reset_parameters()
        for finalMLP in self.finalMLP_list:
            for lin in finalMLP:
                if isinstance(lin, Linear):
                    torch.nn.init.xavier_uniform_(lin.weight)
                    lin.bias.data.fill_(0)  

    def single_forward(self, coords,edge_index,edge_index_2rd, edx_2nd,batch,input_feature,is_source,edge_rep,datasource_idx):
        distances={}
        thetas={}
        if edge_rep:
            i, j, k = edge_index_2rd 
            distances[1]=(coords[edge_index[0]] - coords[edge_index[1]]).norm(p=2, dim=1)
            theta_ijk = get_angle(coords[j] - coords[i], coords[k] - coords[j])
            v1 = torch.cross(F.pad(coords[j] - coords[i],(0,1)), F.pad(coords[k] - coords[j],(0,1)), dim=1)[...,2]
            flag = torch.sign((v1))
            flag[flag==0]=-1
            thetas[1] = scatter(theta_ijk*flag ,edx_2nd,dim=0,dim_size=edge_index.shape[1],reduce='min')
            thetas[1]=self.theta_expansion(thetas[1])
            geo_encoding_1st=distances[1][:,None]
            geo_encoding_1st[geo_encoding_1st==0]=1E-10
            geo_encoding_1st=torch.pow(geo_encoding_1st,-1)        
            geo_encoding_2nd = thetas[1]
            geo_encoding=torch.cat([geo_encoding_1st,geo_encoding_2nd],dim=-1)
        else:
            # coords=theoryencoder(coords[None,:])
            # coords=coords[0].to("cuda")
            
            coords_j = coords[edge_index[0]]
            coords_i = coords[edge_index[1]]
            geo_encoding=torch.cat([coords_j,coords_i],dim=-1)
        if edge_rep:
            for lin in self.mlps_list[datasource_idx]:
                geo_encoding=lin(geo_encoding)
        else:
            for lin in self.mlps_list_backup[datasource_idx]:
                geo_encoding=lin(geo_encoding)
            geo_encoding=torch.zeros_like(geo_encoding,device=geo_encoding.device,dtype=geo_encoding.dtype)            
        node_feature=self.translinear(input_feature[:,:-2])
        for interaction in self.interactions_list[datasource_idx]:
            node_feature =  interaction(node_feature,geo_encoding,edge_index,is_source)
        return node_feature
    def forward(self, coords,edge_index,edge_index_2rd, edx_2nd,batch,input_feature,is_source,edge_rep):
        outputs=[]
        for i in range(len(coords)):
            output=self.single_forward(coords[i],edge_index[i],edge_index_2rd[i], edx_2nd[i],batch[i],input_feature[i],is_source[i],edge_rep,i)
            for lin in self.finalMLP_list[i]:
                output=lin(output)
            outputs.append(output)
        return outputs
    
class SPNN(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_channels,
        activation=torch.nn.ReLU(),
        finaldepth=3,
        batchnorm="False",
        num_input_geofeature=13
    ):
        super(SPNN, self).__init__()
        self.activation = activation
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.num_input_geofeature=num_input_geofeature
        self.att = Parameter(torch.Tensor(1, hidden_channels),requires_grad=True)

        self.WMLP = ModuleList()
        for i in range(self.finaldepth + 1):
            if i == 0:
                self.WMLP.append(Linear(hidden_channels*2+num_input_geofeature, hidden_channels))
            else:
                self.WMLP.append(Linear(hidden_channels, hidden_channels))  
            if self.batchnorm == "True":
                self.WMLP.append(nn.BatchNorm1d(hidden_channels))
            self.WMLP.append(self.activation)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.WMLP:
            if isinstance(lin, Linear):
                torch.nn.init.xavier_uniform_(lin.weight)
                lin.bias.data.fill_(0)
        glorot(self.att)
    def forward(self, node_feature,geo_encoding,edge_index,is_source):
        j, i = edge_index
        input_feature=node_feature.clone()
        if node_feature is None:
            concatenated_vector = geo_encoding
        else:
            node_attr_0st = node_feature[i]
            node_attr_1st = node_feature[j]
            concatenated_vector = torch.cat(
                [
                    node_attr_0st,
                    node_attr_1st,
                    geo_encoding,
                ],
                dim=-1,
            )
        x_i = concatenated_vector
        for lin in self.WMLP:
            x_i=lin(x_i)    
        input_feature_j=input_feature[edge_index[0]]
        x_i = F.leaky_relu(x_i)
        alpha = F.leaky_relu(x_i * self.att).sum(dim=-1)
        alpha = softmax(alpha, edge_index[1])
        
        message=input_feature_j * alpha.unsqueeze(-1)
        out_feature = scatter(message, edge_index[1], dim=0, reduce='add')    
        out_feature=input_feature+out_feature
     
        return out_feature


