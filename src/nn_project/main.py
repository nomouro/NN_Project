def main():


    import math
    import os
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ase.db import connect
    from ase.neighborlist import neighbor_list
    from omegaconf import OmegaConf
    from torch import nn

    from .models import schnet
    from .data import data_f
    from .training import fit
    from .training import test
    from .utils import paths

    np.random.seed(42)

    cwd = os.getcwd()

    data = connect(f'{cwd}/DATA/new_dataset.db')  ### Database 

    N_frames = len(data)

    config_dir = paths.CONFIG_PATH   ### Root config file 

    conf_gen = OmegaConf.load(config_dir)
    print("General config",conf_gen)

    conf_cwd = OmegaConf.load("config.yaml") ### Local config 
    print("Local config",conf_cwd)

    conf = OmegaConf.merge(conf_gen,conf_cwd) ### Final config 
    print("config",conf)


    device = torch.device(conf.device)


                                                                ### Config allocation ###

    run = conf.run                                              # routine, training or testing               
    #run_type = conf.run_type                                   # type of routine - with or without forces             

    cutoff = conf.defaults.cutoff 
    n_modules = conf.defaults.n_modules
    epochs = conf.defaults.epochs                               # max number of epochs 
    bs = conf.defaults.bs                                       # batch size                         
    max_eln = conf.defaults.max_eln                             # maximal number of diverse chemical elements - the element num of the heaviest elmnt + 1
    lr = conf.defaults.lr                                       # learning rate
    embedd_dim = conf.defaults.embedd_dim                       # embedding dimension
    gamma = conf.defaults.gamma                                 # gamma factor used in radial basis function calculation
    force_rho = 0.01                                            # weight of force loss in training
                                                                ### Config allocation ###
    idx = np.random.permutation(N_frames)         
    
    print(idx)


    if (conf.idx == "default"):                                ###                                                                              
        train_stop = int(N_frames*0.8)                         ###                                                     
        val_start = int(N_frames*0.8)                          ###    Standard dataset split                                                
        val_stop = int(N_frames*0.9)                           ###                                                     
        test_start = int(N_frames*0.9)                         ###                                                     
        test_stop = int(N_frames)                              ###                                                 
    else :                                                                              
        train_stop = conf.idx.train_stop                       ###                                                         
        val_start = conf.idx.val_start                         ###                                                      
        val_stop = conf.idx.val_stop                           ###    Config dataset split                                             
        test_start = conf.idx.test_start                       ###                                                         
        test_stop = conf.idx.test_stop                         ###                                                     



    train_idx = idx[:train_stop]                                                           
    val_idx   = idx[val_start:val_stop]
    test_idx  = idx[test_start:test_stop]


    def rbf(Dist_train,gamma):                                                     ###    Gaussian RBF calculation                                                                                             
        ctr = torch.linspace(0,30,300,device=Dist_train.device)                    ###    300 points 0 - 30 Angstrom (A)                                                               
                                                                                   ###    separated by 0.1 A
        return torch.exp(-((Dist_train[:,None]-ctr[None,:])**2)*gamma).float()     ###    distances vector -> vector of vectors with rbf vals.                                                                                   


    total = []

    for idx in train_idx:
        row = data.get(int(idx) + 1)                             ###   Calculating total energy per atom  
        energy = row.data["energy"]                              ###    
        n_atoms = len(row.toatoms())                             ###     

        total.append(energy / n_atoms)
    total = np.array(total)
    target_std = np.std(total)                                   ###  Energy standard deviation
    target_mean = np.sum(total) / len(train_idx)                 ###  Energy mean                 

    target_mean = torch.tensor(
        target_mean,
        dtype=torch.float32,
        device=device
    )
    target_std = torch.tensor(
        target_std,
        dtype=torch.float32,
        device=device
    )
    print("target_mean:",target_mean,"target_std:",target_std)

    

    loss_func = torch.nn.functional.mse_loss
    loss_func_test = torch.nn.L1Loss()

    def get_model():
        model = schnet.GNN_Model(rbf, max_eln,embedd_dim,n_modules,gamma,target_std,target_mean,device).to(device)
        return model, torch.optim.Adam(model.parameters(), lr=lr)



    def loss_batch(model, loss_func, Edge_train,Force_ref, Node_train, Pools,Node_crd_train, e_train, opt):
        e_target = ((e_train / Pools) - target_mean)/target_std             ###  Normalizing target energies

        e_pred, f_pred = model(
            Node_train,
            Edge_train,
            Pools,
            Node_crd_train,
        )

        loss_e = loss_func(e_pred, e_target)
        loss_f = loss_func(f_pred, Force_ref)
        loss = loss_e + force_rho * loss_f                                  ###  force_rho - relative weight of force loss
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss_e.item(), loss_f.item(),1
    
    def loss_batch_test(model, loss_func, Edge_train,Force_ref, Node_train, Pools,Node_crd_train, e_train, opt):
        e_target = e_train                                                  ###  Target energies as they are in dataset
        pred_per_atom, f_pred = model(
            Node_train,
            Edge_train,
            Pools,
            Node_crd_train,
        )
        e_pred = ((pred_per_atom * target_std) + target_mean) * Pools       ### Denormalizing predicted energies
        loss_e = loss_func(e_pred, e_target)
        loss_f = loss_func(f_pred, Force_ref)

        return loss_e.item(), loss_f.item(),1

    
    min_val_loss = float("inf")
    print("Run type",run)
    if run == "train":

        if os.path.exists("best_model.pth"):          
            best_state_dict = torch.load("best_model.pth",map_location=device,weights_only=False)
            min_val_loss = best_state_dict["min_val_loss"]        ### only loading minimal validation loss from best model
            if 'train_loss_f' in best_state_dict:
                print("forces present")
                print("old best_model DATA:",best_state_dict["epoch"],best_state_dict["train_loss"],min_val_loss)
                print("old best_model NORMALIZATION DATA:",best_state_dict["target_mean"],best_state_dict["target_std"])
                print("Force train mse:", best_state_dict["train_loss_f"],"Force val mse:", best_state_dict["val_loss_f"])
            else:
                print("old best_model DATA:",best_state_dict["epoch"],best_state_dict["train_loss"],min_val_loss)
                print("old best_model NORMALIZATION DATA:",best_state_dict["target_mean"],best_state_dict["target_std"])



        if os.path.exists("checkpoint.pth"):                                                    ### loading all other parameters from last checkpoint
            state_dict = torch.load("checkpoint.pth",map_location=device,weights_only=False)
            with torch.device(device):
                model,opt = get_model()
            model.load_state_dict(state_dict['model_state_dict'])
            opt.load_state_dict(state_dict['optimizer_state_dict'])
            last_epoch = state_dict["epoch"]
            last_train_loss = state_dict["train_loss"]
            last_val_loss = state_dict["val_loss"]
            target_mean = state_dict["target_mean"]
            target_std = state_dict["target_std"]
            for g in opt.param_groups:
                g['lr'] = lr                                                                                    ###  overwriting lr from past training
            print("last epoch DATA:",last_epoch,last_train_loss,last_val_loss)
            print("last model NORMALIZATION DATA:",state_dict["target_mean"],state_dict["target_std"])
        else: 
            last_epoch = 0
            model,opt = get_model()

        fit.fit(data_f.DATA,loss_batch,last_epoch, epochs, model, loss_func, opt,train_idx,val_idx,bs,min_val_loss,data,device,cutoff,target_std,target_mean)


    elif run == "test":
        if os.path.exists("best_model.pth"):
            best_state_dict = torch.load("best_model.pth",map_location=device,weights_only=False)

            with torch.device(device):
                model,opt = get_model()
            model.load_state_dict(best_state_dict['model_state_dict'])
            min_val_loss = best_state_dict["min_val_loss"]
            last_epoch = best_state_dict["epoch"]
            target_mean = best_state_dict["target_mean"]
            target_std = best_state_dict["target_std"]
            print("old best_model DATA:",best_state_dict["epoch"],best_state_dict["train_loss"],min_val_loss)
            print("old best_model NORMALIZATION DATA:",best_state_dict["target_mean"],best_state_dict["target_std"])
            print("Force train mse:", best_state_dict["train_loss_f"],"Force val mse:", best_state_dict["val_loss_f"])
        else: 
            print("No best_model detected, exiting...")
            exit(0)



        test.test(data_f.DATA,loss_batch_test,last_epoch, model, loss_func_test, test_idx, bs,data,device,cutoff,opt)     





if __name__ == "__main__":
    main()   