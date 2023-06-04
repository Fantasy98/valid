import torch
from torch.utils.data import DataLoader
from torch import nn 
from utils.networks import Init_Conv
from utils.newnets import FCN_Pad_Xaiver_CBAM
from utils.toolbox import periodic_padding
from utils.Vision_Transformer import ViT, ViTBackbone
from utils.vit import VisionTransformer
from tqdm import tqdm
torch.manual_seed(100)

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

HEIGHT = 256;WIDTH = 256; CHANNELS = 4; KNSIZE=3;padding = 8
batch_size = 1;
EPOCH = 10
model = FCN_Pad_Xaiver_CBAM(HEIGHT,WIDTH,CHANNELS,KNSIZE,8)
model_name = f"y_plus_30_pr0025_FCN_CBAM_{EPOCH}epoch_{batch_size}batch"
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-7,betas=(0.9,0.999),weight_decay=2e-2)


train_dl = DataLoader(torch.load("./test1.pt"),batch_size=batch_size, shuffle=True)
model.apply(Init_Conv)
model.to(device)
model.train(True)

loss_hist = []
for epoch in range(EPOCH):
    train_loss = 0
    i = 0
    print(f"Epoch {epoch}",flush=True)
    for batch in tqdm(train_dl):
                i += 1
                x,y  = batch
                
                x = x.float().to(device); y = y.double().to(device)

                pred = model(x).double()
                optimizer.zero_grad()
                loss = loss_fn(pred,y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # train_loss = loss.item()
    
    loss_hist.append(train_loss/i)
                
    print(f"Loss:{loss_hist[-1]}",flush=True)
                    # break