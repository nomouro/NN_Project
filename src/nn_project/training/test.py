import math
import torch
import numpy as np

def test(DATA,loss_batch_test,last_epoch, model, loss_func, test_idx, bs,data,device,cutoff,opt):
    all_loss_e = 0
    all_loss_f = 0
    all_num = 0
    model.eval()
    for batch in range(math.ceil(len(test_idx) / bs)):
        Edge_train, Force_ref, y_train, Node_train, Pools, Node_crd_train  = DATA(test_idx,bs,batch,data,device,cutoff)
        losses_e, losses_f, nums = loss_batch_test(model, loss_func, Edge_train,Force_ref, Node_train, Pools,Node_crd_train , y_train, opt)
        all_loss_e += np.sum(np.multiply(losses_e, nums))
        all_loss_f += np.sum(np.multiply(losses_f, nums))
        all_num += np.sum(nums)
    
    test_loss_e = all_loss_e / all_num
    test_loss_f = all_loss_f / all_num
    
    print("last epoch:", last_epoch, "test_e_MAE:",test_loss_e, "test_f_MAE:", test_loss_f)