import torch
from torch import nn
import torch.nn.functional as F
import math
class ShiftedSoftplus(nn.Module):           
    def forward(self, x):                   
        return F.softplus(x) - math.log(2.0)


    
class GNN_Model(nn.Module):
    def __init__(self,rbf, max_eln,embedd_dim,n_modules,gamma,target_std,target_mean,device):
        super().__init__()
        self.embedding = nn.Embedding(max_eln, embedd_dim)      ### max_eln must be atomic number of heaviest element in dataset
        self.cfconv_m = nn.ModuleList([nn.Sequential(                               ###  nn part of continuous-filter convolutional layer                                 
                                           nn.Linear(300, embedd_dim),              ###  input - dim of rbf (300)                             
                                           ShiftedSoftplus(),                       ###                                         
                                           nn.Linear(embedd_dim, embedd_dim),       ###  output in dimension of "atom"                                      
                                           ShiftedSoftplus()                        ###                                       
                                        ) for i in range(n_modules)])                                    
        
        self.inter_m = nn.ModuleList([nn.Sequential(            
                                          nn.Linear(embedd_dim, embedd_dim),
                                          ShiftedSoftplus(),                
                                          nn.Linear(embedd_dim, embedd_dim),
                                        ) for i in range(n_modules)])
        self.atom_m = nn.ModuleList([nn.Sequential(
                                            nn.Linear(embedd_dim, embedd_dim)         
                                        )  for i in range(n_modules)])
        
        self.end = nn.Sequential(                               ### end layer                                                    
            nn.Linear(embedd_dim, 32),                                  ### produces 1-dim per atom energy contribution       
            ShiftedSoftplus(),                                  ###                    
            nn.Linear(32, 1)                                    ###                   

        )
        self.rbf = rbf
        self.n_modules = n_modules
        self.gamma = gamma
        self.target_std = target_std
        self.target_mean = target_mean
        self.device = device
        
    def forward(self,Node_train ,Edge_train,Pools, Node_crd_train):
        def Dist_f(Edge_train, Node_crd_train):                ###    Manual calculation of interatom distances                                                               
            a = Edge_train[:,0]                                ###    Nodes                      
            b = Edge_train[:,1]                                ###    Neighbors of nodes                      
            Dist = Node_crd_train[a] - Node_crd_train[b]       ###    simplify later ->                                                 
            Dist = torch.square(Dist)                          ###    .                              
            Dist = torch.sum(Dist,1)                           ###    .                              
            Dist = torch.sqrt(Dist)                            ###    -> all in one function                          
            return Dist
        def interaction(atom,inter,dt,Node_emb,Edge_train):              ###   Interaction block                                                                                                          
            p_at = atom(Node_emb)                         ###   processing embedded atoms with atom layer                                                                                                                           
            n_j = p_at[Edge_train[:,1]]                         ###   replacing atom numbers with processed embedings                                                                                              
            prod = dt * n_j                                    ###   el-wise multiplication of filter tensors (dt) and feature representation of neighbors                                                                                   
                                                               ###                                                         
            n_agg = torch.zeros_like(Node_emb)               ###                                                                                                             
            n_agg.index_add_(0,Edge_train[:,0],prod)         ###   Node-wise aggregation of the above                        
            inter_res = inter(n_agg)                       ###   resulting convolution processed by 2 atom-wise layers
            return inter_res                                       ###   resulting residual
        Node_crd_train.requires_grad_(True)                             ###  Records operations on tensor representing coords, to later use torch.autograd.grad (derivative) of energy w.r.t. positions -> forces          
        Dist = Dist_f(Edge_train,Node_crd_train)                                                                                                     
        Dist = self.rbf(Dist,self.gamma)                                                                       
        Node_emb  = self.embedding(Node_train)                                                                       
        for i in range(self.n_modules):
            if i == 0:
                dt = self.cfconv_m[i](Dist)                                                           ### calculating filter value from relative distances                               
                cycle = Node_emb + interaction(self.atom_m[i],self.inter_m[i],dt,Node_emb,Edge_train)   ### calculating next layer features from preavious (embeddings) + residuals     
            else:
                dt = self.cfconv_m[i](Dist)                                                                       
                cycle = cycle + interaction(self.atom_m[i],self.inter_m[i],dt,cycle,Edge_train)
                                                                 
        end = self.end(cycle).squeeze(-1)                               ###  producing 1-dim per atom energy                                        
        batch_idx = torch.repeat_interleave(                            ###  repeats elements of the tensor torch.arange - Pools times                                                                
            torch.arange(len(Pools),device=self.device),                     ###  creates linear space with number of molecules as entries (1,2...,n_mol)                                                             
            Pools                                                       ###  Pools contains number of atoms for each molecule       
        )                                                               ###  result - tensor (n_atoms) and each entree is n_mol , that this atom belongs to
        energy = torch.zeros_like(Pools,dtype=end.dtype)
        energy.index_add_(0,batch_idx,end)                              ###  summing normalized energy contributions from atoms by above indeces to create tensor (n_mol,)
        energy = energy / Pools                                         ###  averaging normalized energy per n_atoms in each molecule
        energy_for_forces =  ((energy * self.target_std) + self.target_mean) * Pools   ###  denormalizing energy prediction for forces
        forces = -torch.autograd.grad(                                  ###                                                                                      
            outputs=energy_for_forces,                                  ###  Computing grad with respect to energy - f(x)                                      
            inputs=Node_crd_train,                                      ###  variables (arguments) of energy prediction - x                                
            grad_outputs=torch.ones_like(energy_for_forces),            ###  output 1st dim - n_mol                                                            
            create_graph=True                                           ###                              
        )[0]                                                            ###  retaining derivative graph so f_loss can be used in training              
        return energy,forces