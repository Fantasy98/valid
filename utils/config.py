from utils.newnets import FCN_Pad_Xaiver_CBAM
from utils.networks import Init_Conv
from torch.optim import Adam
from torch import nn
class train:
    Padding = 8
    H = 256
    W = 256
    C = 4 
    KNSIZE = 3 

    SEED = 1024
    model = FCN_Pad_Xaiver_CBAM(H,W,C,KNSIZE,Padding) 
    init = Init_Conv
    adam_eps = 1e-7
    BATCH_SIZE = 6
    EPOCH = 100
    LR = 1e-3
    loss_fn = nn.MSELoss()
    

