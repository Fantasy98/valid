import torch 

def fit(model,optimizer,loss_fn,train_dl,device):
    from tqdm import tqdm
    """
    Training loop for single epoch
        input: 
            model: torch model 
            optimizer: torch.optim 
            loss_fn : loss function
            device: cuda device
            train_dl : dataloader for training
            val_dl : dataloader for validation
        output:
            train_loss: loss.item() of train at this epoch 
            val_loss: loss.item() of validation at this epoch 
    """
    
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_loss = 0.0
    val_running_loss = 0.0
    val_loss = 0.0
    counter = 0 
    total = 0 
    bn_size = train_dl.batch_size
    for batch in tqdm(train_dl):
        counter += 1
        x,y = batch
        x = x.float().to(device); y  = y.double().to(device)
        optimizer.zero_grad()
        pred = model(x).double()
        loss= loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        train_running_loss +=loss.item()/bn_size

    train_loss = train_running_loss/counter
    
    return train_loss

def validation(model,loss_fn,val_dl,device):
    from tqdm import tqdm
    """
    Validation loop for each epoch using validation dataset
    input:
        model: torch model
        loss_fn: same metrics used in training
        val_dl: validation dataloader
        device: cuda device
    """
    model.eval()
    print("Validating")
    val_running_loss = 0.0
    val_loss = 0.0
    counter = 0
    bn_size = val_dl.batch_size
    with torch.no_grad():
        for valid_batch in tqdm(val_dl):
            counter +=1 
            x_val,y_val = valid_batch
            pred_val = model(x_val.float().to(device)).float()
            loss_val = loss_fn(pred_val,y_val.float().to(device))
            val_running_loss += loss_val.item()/bn_size
    
    val_loss = val_running_loss/counter
    return val_loss


    