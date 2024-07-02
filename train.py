import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import time
import torch.optim as optim

from matplotlib import cm
import matplotlib.pyplot as plt
import json
from model import GFusion
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.utils import add_self_loops
from torch.nn.functional import softmax
from torch_geometric.nn import knn_graph
import copy

torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve, auc
from sklearn.feature_selection import r_regression
import pickle
from utils.utils import triplets,unique,pos2key
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default="True")
    parser.add_argument('--loadmodel', type=str, default="False")
    parser.add_argument('--split_dataset', type=str, default="False")
    parser.add_argument('--model', type=str, default="GFusion")
    
    # ablation
    parser.add_argument('--edge_rep', type=str, default="True")
    parser.add_argument('--single_high', type=str, default="False")
    parser.add_argument('--fidelity_train', type=str, default="True")
    parser.add_argument('--fidelity_low_weight', type=float, default=-1.0)
    parser.add_argument('--share', type=str, default="101")

    parser.add_argument('--dataset', type=str, default='flu')
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--man_seed', type=int, default=12345)
    parser.add_argument('--test_per_round', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30)  #scheduler
    parser.add_argument('--nepoch', type=int, default=201)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--activation', type=str, default='relu')#'lrelu'
    parser.add_argument('--batchSize', type=int, default=512)

    parser.add_argument('--num_neighbors', type=int, default=3)
    parser.add_argument('--regression_loss', type=str, default='l2')

    parser.add_argument('--h_ch', type=int, default=16)
    parser.add_argument('--localdepth', type=int, default=1) # mlp(distance) mlp(theta) >=1
    parser.add_argument('--num_interactions', type=int, default=1) #>=1
    parser.add_argument('--finaldepth', type=int, default=3) # mlp(concat node_attr and geo_encoding)
    
    args = parser.parse_args()
    args.log=True if args.log=="True" else False
    args.loadmodel=True if args.loadmodel=="True" else False
    args.split_dataset=True if args.split_dataset=="True" else False
    args.edge_rep=True if args.edge_rep=="True" else False
    args.single_high=True if args.single_high=="True" else False
    args.fidelity_train=True if args.fidelity_train=="True" and args.single_high is False and args.fidelity_low_weight==-1.0 else False
    args.manualSeed=True if args.manualSeed=="True" else False
    args.save_dir=os.path.join('./save/',args.dataset)
    return args

def main(args,train_Loader,val_Loader,test_Loader):
    if flag:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    measure_Pearsonr=r_regression
    criterion_l1 = torch.nn.L1Loss() #reduction='sum'
    criterion_l2 = torch.nn.MSELoss()
    criterion=criterion_l1 if args.regression_loss=='l1' else criterion_l2
    if args.model in ['GFusion']:
        def myL1(pred,true,weight=None,reduction='mean'):
            loss=(abs(pred-true))
            num=len(pred)
            if weight is not None:
                loss=[weight[i]*loss[i] for i in range(num)]
            loss=sum(loss)
            if reduction=='mean':
                loss=loss/num
            return loss
        def myL2(pred,true,weight=None,reduction='mean'):
            loss=((pred-true)**2)
            num=len(pred)
            if weight is not None:
                loss=[weight[i]*loss[i] for i in range(num)]
            loss=sum(loss)
            if reduction=='mean':
                loss=loss/num
            return loss
        criterion=myL1 if args.regression_loss=='l1' else myL2
        num_of_fidelities=len(train_graphs[0])
    
        def reweight_fidelity():
            if args.single_high:
                weighted_fidelity_weight[0]=1
                weighted_fidelity_weight[1]=0
            elif args.fidelity_low_weight!=-1.0:
                weighted_fidelity_weight[0]=1
                weighted_fidelity_weight[1]=args.fidelity_low_weight
            else:
                exped_f=[torch.exp(fidelity_weight[i]) for i in range(num_of_fidelities)]
                fsum=sum(exped_f)
                for i in range(num_of_fidelities):
                    weighted_fidelity_weight[i]=exped_f[i]/fsum
        fidelity_weight,weighted_fidelity_weight=[],[]
        if args.dataset in ['south',"north","flu"]:
            for i in range(num_of_fidelities):
                fidelity_weight+=[torch.tensor(1.0/num_of_fidelities,dtype=torch.float32).requires_grad_()]
                weighted_fidelity_weight+=[0]     
        elif args.dataset in ["syn"]:
            fidelity_weight=[torch.tensor(1,dtype=torch.float32).requires_grad_(),torch.tensor(0.0,dtype=torch.float32).requires_grad_()]
            for i in range(num_of_fidelities):
                # fidelity_weight+=[torch.tensor(1.0/num_of_fidelities,dtype=torch.float32).requires_grad_()]
                weighted_fidelity_weight+=[0]      
        reweight_fidelity()
        if args.dataset in ['south',"north"]:
            x_in=30
        elif args.dataset in ['flu']:
            x_in=0
        elif args.dataset=='syn':
            x_in=1
        else:
            raise Exception('Dataset not recognized.')
    if args.model=="GFusion":
        GFusion_model=GFusion(h_channel=args.h_ch,input_featuresize=x_in,\
                            localdepth=args.localdepth,num_interactions=args.num_interactions,finaldepth=args.finaldepth,share=args.share)
        GFusion_model.to(device)
        optimizer = torch.optim.Adam( list(GFusion_model.parameters()), lr=args.lr)
        if args.fidelity_train:
            optimizer2 = torch.optim.Adam(fidelity_weight, lr=optimizer.param_groups[0]['lr']*10)
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, factor=0.1, patience=args.patience, min_lr=1e-8)   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   

    def train(GFusion_model):
        epochloss=0
        y_hat, y_true,y_hat_logit = [], [], []        
        optimizer.zero_grad()
        if args.fidelity_train: optimizer2.zero_grad()
        if args.model=="GFusion":
            GFusion_model.train()
            for i,data in enumerate(train_Loader):
                if num_of_fidelities==2:
                    x1, pos1,edge_index1, batch1,target_index1,target1,is_source1 = data[0].x, data[0].pos,data[0].edge_index, data[0].batch,data[0].target_index,data[0].target,data[0].is_source
                    x2, pos2,edge_index2, batch2,target_index2,target2,is_source2 = data[1].x, data[1].pos,data[1].edge_index, data[1].batch,data[1].target_index,data[1].target,data[1].is_source
                    if args.dataset=='syn':
                        x1[:,1]=x1[:,1]+x1[:,2]
                        x1=x1[:,[0,1,3,4]]
                        x2[:,1]=x2[:,1]+x2[:,2]
                        x2=x2[:,[0,1,3,4]]
                    x1,pos1,target1,x2,pos2,target2=x1.to(torch.float32),pos1.to(torch.float32),target1.to(torch.float32),x2.to(torch.float32),pos2.to(torch.float32),target2.to(torch.float32)
                    x2[x2[:,0]>6666,0]=6666
                    # edge_index,_=add_self_loops(edge_index,num_nodes=x.size(0))
                    datasource=data[0].datasource
                    Y = target1
                    assert(torch.equal(target1,target2))
                    Y[Y>6666]=6666
                    x1, pos1,edge_index1, batch1, target_index1,is_source1 = x1.to(device),pos1.to(device), edge_index1.to(device), batch1.to(device),target_index1.to(device),is_source1.to(device)
                    x2, pos2,edge_index2, batch2, target_index2,is_source2 = x2.to(device),pos2.to(device),edge_index2.to(device), batch2.to(device),target_index2.to(device),is_source2.to(device)
                    """ 
                    triplets are not the same for graphs when training 
                    """
                    num_nodes1=x1.shape[0]
                    num_nodes2=x2.shape[0]
                    edge_index_2rd_1, _, _, edx_2nd_1 = triplets(edge_index1, num_nodes1)
                    edge_index_2rd_2, _, _, edx_2nd_2 = triplets(edge_index2, num_nodes2)
                    
                    pm25_1,pm25_2=GFusion_model([pos1,pos2],[edge_index1,edge_index2],[edge_index_2rd_1,edge_index_2rd_2],\
                                            [edx_2nd_1,edx_2nd_2],[batch1,batch2],[x1,x2],[is_source1,is_source2],args.edge_rep)      
                    pm25_1,pm25_2=pm25_1[target_index1],pm25_2[target_index2]

                    if args.dataset=='syn':
                        pred=((pm25_1*weighted_fidelity_weight[0]+pm25_2*weighted_fidelity_weight[1]).cpu())
                    else:
                        pred=F.relu((pm25_1*weighted_fidelity_weight[0]+pm25_2*weighted_fidelity_weight[1]).cpu())
                    
                    loss_weight= [weighted_fidelity_weight[i] for i in datasource]
                    loss1 = criterion(pred.reshape(-1, 1), Y.reshape(-1, 1),loss_weight)
                """
                record predictions
                """                 
                y_hat += list(pred.detach().numpy().reshape(-1))
                y_true += list(Y.detach().numpy().reshape(-1))                 
                loss=loss1
                loss.backward()
                epochloss+=loss
                optimizer.step()
                optimizer.zero_grad()
                if args.fidelity_train:
                    optimizer2.step()
                    optimizer2.zero_grad()
                reweight_fidelity()
        return epochloss.item()/len(train_Loader),y_hat, y_true

    def test(loader,GFusion_model,fidelity_weight):
        if not args.single_high:
            weighted_fidelity_weight=[i.detach() for i in fidelity_weight]
            exped_f=[torch.exp(fidelity_weight[i]) for i in range(num_of_fidelities)]
            fsum=sum(exped_f)
            for i in range(num_of_fidelities):
                weighted_fidelity_weight[i]=exped_f[i]/fsum
        else:
            weighted_fidelity_weight=[1,0]
        y_hat, y_true,y_hat_logit = [], [], []
        loss_total, pred_num = 0, 0
        GFusion_model.eval()
        for i,data in enumerate(loader):
            if num_of_fidelities==2:
                x1, pos1,edge_index1, batch1,target_index1,target1,is_source1 = data[0].x, data[0].pos,data[0].edge_index, data[0].batch,data[0].target_index,data[0].target,data[0].is_source
                x2, pos2,edge_index2, batch2,target_index2,target2,is_source2 = data[1].x, data[1].pos,data[1].edge_index, data[1].batch,data[1].target_index,data[1].target,data[1].is_source
                if args.dataset=='syn':
                    x1[:,1]=x1[:,1]+x1[:,2]
                    x1=x1[:,[0,1,3,4]]
                    x2[:,1]=x2[:,1]+x2[:,2]
                    x2=x2[:,[0,1,3,4]]               
                x1,pos1,target1,x2,pos2,target2=x1.to(torch.float32),pos1.to(torch.float32),target1.to(torch.float32),x2.to(torch.float32),pos2.to(torch.float32),target2.to(torch.float32)
                x2[x2[:,0]>6666,0]=6666
                # edge_index,_=add_self_loops(edge_index,num_nodes=x.size(0))
                datasource=data[0].datasource
                Y = target1
                assert(torch.equal(target1,target2))
                Y[Y>6666]=6666
                x1, pos1,edge_index1, batch1, target_index1,is_source1 = x1.to(device),pos1.to(device), edge_index1.to(device), batch1.to(device),target_index1.to(device),is_source1.to(device)
                x2, pos2,edge_index2, batch2, target_index2,is_source2 = x2.to(device),pos2.to(device),edge_index2.to(device), batch2.to(device),target_index2.to(device),is_source2.to(device)

                num_nodes1=x1.shape[0]
                num_nodes2=x2.shape[0]
                edge_index_2rd_1, num_2nd_neighbors_1, edx_1st_1, edx_2nd_1 = triplets(edge_index1, num_nodes1)
                edge_index_2rd_2, num_2nd_neighbors_2, edx_1st_2, edx_2nd_2 = triplets(edge_index2, num_nodes2)
                pm25_1,pm25_2=GFusion_model([pos1,pos2],[edge_index1,edge_index2],[edge_index_2rd_1,edge_index_2rd_2],\
                                        [edx_2nd_1,edx_2nd_2],[batch1,batch2],[x1,x2],[is_source1,is_source2],args.edge_rep)             
                pm25_1,pm25_2=pm25_1[target_index1],pm25_2[target_index2]
                with torch.no_grad(): 
                    if args.dataset=='syn':
                        pred=((pm25_1*weighted_fidelity_weight[0]+pm25_2*weighted_fidelity_weight[1]).cpu())
                    else:
                        pred=F.relu((pm25_1*weighted_fidelity_weight[0]+pm25_2*weighted_fidelity_weight[1]).cpu())
                    assert(all(datasource==0))
                    loss1 = criterion(pred.reshape(-1, 1), Y.reshape(-1, 1))*weighted_fidelity_weight[0]
            """
            record predictions
            """                 
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(Y.detach().numpy().reshape(-1)) 
            pred_num += len(Y.reshape(-1, 1))               
            loss=loss1      
            loss_total += loss.detach() * len(Y.reshape(-1, 1))
        return loss_total/pred_num, y_hat, y_true
    if args.loadmodel:
        try:
            suffix='Oct31-11:50:30'
            GFusion_model.load_state_dict(torch.load(os.path.join("save",args.dataset,'model','best_GFusion_model_'+suffix+'.pth')),strict=True)
            best_GFusion_model = copy.deepcopy(GFusion_model)
        except OSError:
            pass    
    else:
        best_val_trigger = 1e3
        old_lr=1e3
        suffix="{}{}-{}:{}:{}".format(datetime.now().strftime("%h"),
                                        datetime.now().strftime("%d"),
                                        datetime.now().strftime("%H"),
                                        datetime.now().strftime("%M"),
                                        datetime.now().strftime("%S"))        
        if args.log: 
            writer = SummaryWriter(os.path.join(tensorboard_dir,suffix))
        for epoch in range(args.nepoch):
            if args.model in ['GFusion']: train_loss,y_hat, y_true=train(GFusion_model)
            if args.log: 
                writer.add_scalar('loss/Train', train_loss, epoch)
            if args.dataset in ['south',"north",'syn','flu']:          
                train_mae=mean_absolute_error(y_true, y_hat)
                train_rmse = np.sqrt(mean_squared_error(y_true, y_hat))
                if args.log:
                    writer.add_scalar('mae/Train', train_mae, epoch)
                    writer.add_scalar('rmse/Train', train_rmse, epoch)
                print(( f"epoch[{epoch:d}] train_loss : {train_loss:.3f} train_mae : {train_mae:.3f} train_rmse : {train_rmse:.3f}" ))
                if args.model in ['GFusion']: 
                    if args.fidelity_train==True:   
                        print(f"fidelity weight: {fidelity_weight[0]:.3f}, {fidelity_weight[1]:.3f}")
                    print(f"weighted_fidelity_weight: {weighted_fidelity_weight[0]:.3f}, {weighted_fidelity_weight[1]:.3f}")
            if epoch % args.test_per_round == 0:
                if args.model in ['GFusion']:
                    val_loss, yhat_val, ytrue_val = test(val_Loader,GFusion_model,fidelity_weight)
                    test_loss, yhat_test, ytrue_test = test(test_Loader,GFusion_model,fidelity_weight)
                if args.log:
                    writer.add_scalar('loss/val', val_loss, epoch)
                    writer.add_scalar('loss/test', test_loss, epoch)
                if args.dataset in ['south',"north",'syn','flu']:          
                    val_mae=mean_absolute_error(ytrue_val, yhat_val)
                    val_rmse = np.sqrt(mean_squared_error(ytrue_val, yhat_val))
                    if args.log:
                        writer.add_scalar('mae/val', val_mae, epoch)
                        writer.add_scalar('rmse/val', val_rmse, epoch)
                    print(blue( f"epoch[{epoch:d}] val_mae : {val_mae:.3f} val_rmse : {val_rmse:.3f}" ))        
                    test_mae = mean_absolute_error(ytrue_test, yhat_test)
                    test_rmse = np.sqrt(mean_squared_error(ytrue_test, yhat_test))
                    test_var=explained_variance_score(ytrue_test,yhat_test)
                    test_coefOfDetermination=r2_score(ytrue_test,yhat_test)
                    test_Pearsonr=measure_Pearsonr(np.array(yhat_test).reshape(-1, 1),np.array(ytrue_test).reshape(-1))[0]
                    if args.log:
                        writer.add_scalar('mae/test', test_mae, epoch)
                        writer.add_scalar('rmse/test', test_rmse, epoch)
                    print(blue( f"epoch[{epoch:d}] test_mae: {test_mae:.3f} test_rmse: {test_rmse:.3f} test_Pearsonr: {test_Pearsonr:.3f} test_coefOfDetermination: {test_coefOfDetermination:.3f}" ))
                    if args.model in ['GFusion']: 
                        if args.fidelity_train==True:   
                            print(f"fidelity weight: {fidelity_weight[0]:.3f}, {fidelity_weight[1]:.3f}")
                        print(f"weighted_fidelity_weight: {weighted_fidelity_weight[0]:.3f}, {weighted_fidelity_weight[1]:.3f}")
                    val_trigger=val_mae                  
                if val_trigger < best_val_trigger:
                    best_val_trigger = val_trigger
                    best_GFusion_model = copy.deepcopy(GFusion_model)
                    best_fidelity=copy.deepcopy(fidelity_weight)
                    best_info=[epoch,val_trigger]
            """ 
            update lr when epoch≥30
            """
            if epoch >= 30:
                lr = scheduler.optimizer.param_groups[0]['lr']
                if old_lr!=lr:
                    print(red('lr'), epoch, (lr), sep=', ')
                    old_lr=lr
                scheduler.step(val_trigger)
                if args.fidelity_train:
                    scheduler2.step(val_trigger)             
    val_loss, yhat_val, ytrue_val = test(val_Loader,best_GFusion_model,best_fidelity)
    test_loss, yhat_test, ytrue_test = test(test_Loader,best_GFusion_model,best_fidelity)
    if args.dataset in ['south',"north",'syn','flu']:          
        val_mae = mean_absolute_error(ytrue_val, yhat_val)
        val_rmse=np.sqrt(mean_squared_error(ytrue_val,yhat_val))
        val_var=explained_variance_score(ytrue_val,yhat_val)
        print(blue( f"best_val  val_mae: {val_mae:.3f} val_rmse: {val_rmse:.3f} val_var: {val_var:.3f}" ))
        
        test_mae=mean_absolute_error(ytrue_test,yhat_test)
        test_rmse=np.sqrt(mean_squared_error(ytrue_test,yhat_test))
        test_var=explained_variance_score(ytrue_test,yhat_test)
        test_coefOfDetermination=r2_score(ytrue_test,yhat_test)
        test_Pearsonr=measure_Pearsonr(np.array(yhat_test).reshape(-1, 1),np.array(ytrue_test).reshape(-1))[0]
        print(blue( f"best_test test_mae: {test_mae:.3f} test_rmse: {test_rmse:.3f} test_var: {test_var:.3f}" ))
    if not args.loadmodel:
        """
        save training info and best result 
        """
        result_file=os.path.join(info_dir, suffix)
        with open(result_file, 'w') as f:
            print(args.num_neighbors,args.nepoch,sep=' ',file=f)
            print(f"fidelity weight: {best_fidelity[0]:.3f}, {best_fidelity[1]:.3f}",file=f)
            print("Random Seed: ", Seed,file=f)
            if args.dataset in ['south',"north",'syn','flu']:          
                print(f"MAE  val : {val_mae:.3f}, Test : {test_mae:.3f}", file=f)
                print(f"rmse val : {val_rmse:.3f}, Test : {test_rmse:.3f}", file=f)
                print(f"var  val : {val_var:.3f}, Test : {test_var:.3f}", file=f)        
                print(f"test_coefOfDetermination: {test_coefOfDetermination:.3f}, test_Pearsonr : {test_Pearsonr:.3f}", file=f)
            print(f"Best info: {best_info}", file=f)
            for i in [[a,getattr(args, a)] for a in args.__dict__]:
                print(i,sep='\n',file=f)
        with open(os.path.join(model_dir,'best_f_weight'+"_"+suffix+".pkl"), 'wb') as handle:
            pickle.dump(fidelity_weight, handle)
        torch.save(best_GFusion_model.state_dict(), os.path.join(model_dir,'best_GFusion_model'+"_"+suffix+'.pth') )
    print("done")
    
if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    tensorboard_dir=os.path.join(args.save_dir,'log')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir,exist_ok=True)
    model_dir=os.path.join(args.save_dir,'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)    
    info_dir=os.path.join(args.save_dir,'info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir,exist_ok=True)      
    Seed = args.man_seed if args.manualSeed else random.randint(1, 10000)
    print("Random Seed: ", Seed)
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)
    flag=0
    if args.dataset in ['south',"north",'syn',"flu"]:
        graphs1,graphs2=dataset.load_point(args.dataset,args.num_neighbors,[False,200,500])
        np.random.shuffle(graphs1)
        val_test_split = int(np.around( 2 / 10 * len(graphs1) ))
        train_val_split = int(len(graphs1)-2*val_test_split)
        if args.single_high:
            train_graphs = graphs1[:train_val_split]
        else:
            train_graphs = graphs1[:train_val_split]+graphs2
        val_graphs = graphs1[train_val_split:train_val_split+val_test_split]
        test_graphs = graphs1[train_val_split+val_test_split:]
        
        np.random.shuffle(train_graphs)        
        train_Loader=DataLoader(train_graphs, batch_size=args.batchSize)
        val_Loader=DataLoader(val_graphs, batch_size=args.batchSize)
        test_Loader=DataLoader(test_graphs, batch_size=args.batchSize)
        print(f"train_pair_num: {len(train_graphs)}, val_pair_num: {len(val_graphs)}, test_pair_num: {len(test_graphs)}")
    else:
        raise Exception('Dataset not recognized.')    
    main(args,train_Loader,val_Loader,test_Loader)
    