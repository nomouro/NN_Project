import math
import torch
import numpy as np

def fit(DATA,loss_batch,last_epoch, epochs, model, loss_func, opt,train_idx,val_idx,bs,min_val_loss,data,device,cutoff,target_std,target_mean):
    for epoch in range(epochs):
        epoch += last_epoch + 1
        all_loss_e = 0
        all_loss_f = 0
        all_num = 0
        model.train()

        for batch in range(math.ceil(len(train_idx) / bs)):
            Edge_train, Force_ref, y_train, Node_train, Pools, Node_crd_train  = DATA(train_idx,bs,batch,data,device,cutoff)
            
            losses_e, losses_f, nums = loss_batch(model, loss_func, Edge_train,Force_ref, Node_train, Pools,Node_crd_train , y_train, opt)
            all_loss_e += np.sum(np.multiply(losses_e, nums))
            all_loss_f += np.sum(np.multiply(losses_f, nums))
            
            all_num += np.sum(nums)
        train_loss_e = all_loss_e / all_num
        train_loss_f = all_loss_f / all_num
        all_loss_e = 0
        all_loss_f = 0
        all_num = 0
        model.eval()
        for batch in range(math.ceil(len(val_idx) / bs)):
            Edge_val, Force_ref, y_val, Node_val, Pools,Node_crd_train  = DATA(val_idx,bs,batch,data,device,cutoff)
            
            losses_e, losses_f, nums = loss_batch(model, loss_func, Edge_val,Force_ref, Node_val, Pools,Node_crd_train , y_val, opt=None)
            all_loss_e += np.sum(np.multiply(losses_e, nums))
            all_loss_f += np.sum(np.multiply(losses_f, nums))
            all_num += np.sum(nums)
        val_loss_e = all_loss_e / all_num
        val_loss_f = all_loss_f / all_num

        if (val_loss_e < min_val_loss):
            print("Saving NEW best model")
            min_val_loss = val_loss_e
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': train_loss_e,
                'train_loss_f': train_loss_f,
                'min_val_loss': min_val_loss,
                'val_loss_f': val_loss_f,
                'target_mean': target_mean,
                'target_std': target_std,
            }, 'best_model.pth')               
       
        print(epoch, "Train_e_RMSE:", math.sqrt(train_loss_e),"Val_e_RMSE:",math.sqrt(val_loss_e), "Train_f_RMSE:", math.sqrt(train_loss_f),"Val_f_RMSE:",math.sqrt(val_loss_f))
        torch.save({  
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss_e,
            'train_loss_f': train_loss_f,
            'val_loss': val_loss_e,
            'val_loss_f': val_loss_f,
            'min_val_loss': min_val_loss,
            'target_mean': target_mean,
            'target_std': target_std,
        }, 'checkpoint.pth')