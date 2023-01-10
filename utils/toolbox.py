

import torch

def Name_Checkpoint(y_plus,var,target,EPOCH):
    import os
    from datetime import datetime
    """
    Give a name to model checkpoint according to the feature,and epoch
    input:
        y_plus: y+ value
        var: list of input feature
        target: list of target
        EPOCH: number of epoch for training
    output:
        name of modelcheck point, format as .pt
    """
    root_dir  = "/storage3/yuning/thesis/models"

    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="|".join(target_sort)
    denotes = 'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"_EPOCH="+str(EPOCH)+".pt"
    
    date_loc=os.path.join(root_dir,str(datetime.now().date()))
    if os.path.exists(date_loc) is False:
        os.mkdir(date_loc)
        print(f"{date_loc} maded!")
    model_dir = os.path.join(date_loc,denotes)
     
    return  model_dir
    
def periodic_padding(input:torch.Tensor,padding:int):
    """
    Function for computing perodic padding for each snapshot 
    input: 
        input: torch tensor with shape (BN,Channel,Height,Width)
        padding: how many pixels to pad around each side
    output: 
        torch tensor with shape of (BN,Channel,Height+padding*2,Width+padding*2)
    """
    if len(input.size()) !=4:
        print("The tenor does not fit the size!")
        return 
    else:
        M1 = torch.cat([input[:,:,:,-padding:],input,input[:,:,:,0:padding]],dim=-1)
        M1 = torch.cat([M1[:,:,-padding:,:],M1,M1[:,:,0:padding,:]],dim=-2)
        return M1

class EarlyStopping():
    def __init__(self,patience=5,tol=0):
        """
        input: 
            patience : how many epochs to wait before stopping 
        when the loss is not improving
            tol: the min difference between current loss and best loss

        """
        self.patience = patience
        self.tol = tol
        self.counter = 0 
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.tol:
            self.best_loss = val_loss
            # reset counter if validation loss imporve
            self.counter = 0 
        elif self.best_loss - val_loss < self.tol:
            self.counter +=1
            if self.counter >= self.patience:
                print("INFO: Patience reached, Early Stop Now!")
                self.early_stop = True

class LRScheduler():
    """
    Learning Rate scheduler, if the val_loss does not descrease for the 
    given number of patience epochs, then the lr will decrease by given factor
    """
    def __init__(self,optimizer,patience = 5, min_lr = 1e-5,factor = 0.8) -> None:
        """
        new_lr = old_lr * factor 
        inputs:
            patience = how many epoch to wait before updata lr
            min_lr  = least lr to reduce
            factor = decrease factor 
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            self.optimizer,
                                                            mode="min",
                                                            patience=self.patience,
                                                            min_lr= self.min_lr,
                                                            factor=self.factor,
                                                            verbose= True
                                                            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

