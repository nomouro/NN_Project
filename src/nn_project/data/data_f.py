from ase.db import connect
from ase.neighborlist import neighbor_list
import numpy as np
import torch


def DATA(train_idx,bs,b_num,data,device,cutoff):                                                                                                              
    with data:                                                                                                                             
        list_of_edges = []                                                                                                                 
        list_of_nodes = []                                                                                                                 
        list_of_energies = []                                                                                                              
        list_of_coord = []                                                                                                                 
        list_of_forces = []                                                                                                                
        list_of_dist = []                                                                                                                  
        last = 0                                                                                                                           
        for idx in train_idx[(b_num*bs):(b_num*bs+bs)]:                                                                                    
            row = data.get(int(idx) + 1)                                                                                                   
            numbers = row["numbers"]                                                                                                       
            list_of_forces.append(row.data.forces)                                                                                         
            atoms = row.toatoms()                                                                                                          
            list_of_coord.append(atoms.get_positions())                                                                                    
            i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff,self_interaction=False)     ### calculating neigbor_list with ase          
            list_of_edges.append(np.dstack((np.array(i),np.array(j)))[0]+last)                 ### adding last to avoid overlap of indexes 
            list_of_dist.append(np.array(d))                                                                                               
            list_of_nodes.append(np.array(numbers))                                                                                        
            list_of_energies.append(row.data["energy"])                                                                                    
            last += len(numbers)         

    Node_crd_train = [torch.tensor(coord) for coord in list_of_coord]                                                                      
    Node_train  = [torch.tensor(nodes) for nodes in list_of_nodes]                                                                         
    Pools = torch.tensor([len(nodes) for nodes in list_of_nodes])                                                                          
    Edge_train  = [torch.tensor(edges) for edges in list_of_edges]                                                                         
    En_train  = [torch.tensor(energies,dtype=torch.float32) for energies in list_of_energies]                                              
    Force_ref =  [torch.tensor(forces) for forces in list_of_forces]                                                                       
    Dist_train  = [torch.tensor(dist) for dist in list_of_dist]          

    y_train = torch.stack(En_train)                                                                                                        
    y_train = torch.squeeze(y_train)     

    Edge_train = torch.cat(Edge_train).to(device)                     ###                                                                  
    Dist_train = torch.cat(Dist_train).to(device)                     ###                                                                  
    Node_train = torch.cat(Node_train).to(device)                     ###                                                                  
    Node_crd_train = torch.cat(Node_crd_train).to(device)             ###  All tensors on same device                                      
    Force_ref = torch.cat(Force_ref).to(device)                       ###                                                                  
    y_train = y_train.to(device)                                      ###                                                                  
    Pools = Pools.to(device)                                          ###        
                                                              
    return Edge_train, Force_ref, y_train, Node_train, Pools, Node_crd_train                                                               