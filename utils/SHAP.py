
import torch.nn as nn
from transformer.sampler import SpaceToDepth
from transformer.cct import CCT
from transformer.ConvBlock import ConvBlock,ConvBlockCat
import math
import torch
from utils.toolbox import periodic_padding
class FullSkip_FCN_CCT_Shap(nn.Module):
    def __init__(self,
                 img_size=256,
                 embedding_dim=256*4,
                 n_input_channels=4,
                 n_conv_layers=2,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.01,
                 stochastic_depth=0.01,
                 num_layers=3,
                 num_heads=8,
                 mlp_ratio=4.0,
                 positional_embedding='learnable',
                 *args, **kwargs):


        super(FullSkip_FCN_CCT_Shap,self).__init__()

        self.CCTEncoder = CCT(img_size=256,
                                embedding_dim=256*4,
                                n_input_channels=4,
                                n_conv_layers=2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                pooling_kernel_size=3,
                                pooling_stride=2,
                                pooling_padding=1,
                                dropout=0.,
                                attention_dropout=0.01,
                                stochastic_depth=0.01,
                                num_layers=3,
                                num_heads=8,
                                mlp_ratio=4.0,
                                positional_embedding='learnable',
                                *args, **kwargs)
        
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.times = embedding_dim//img_size
        # self.Inconv = ConvBlockCat(in_channel=2*embedding_dim)

        self.Downsamplers= nn.ModuleList([ SpaceToDepth(block_size=2**(i+1))
                                          for i in range(self.times) ])
        
        self.Upblocks = nn.ModuleList([ConvBlockCat(in_channel= int(embedding_dim*(0.25**(i)) ) )
                                        for i in range(self.times)])
        self.out = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,padding="same")
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
        self.BN = nn.BatchNorm2d(n_input_channels,eps=0.01,momentum=0.99)
    def forward(self,src):
        enc_x = self.CCTEncoder(src)
        src = self.BN(src)
        B,P2,H_dim = enc_x.shape
        P = int(math.sqrt(P2))
        x = enc_x.reshape(B,H_dim,P,P)
        for i in range(0,self.times):
            if i == 0:
                x_ = self.Downsamplers[-1](src)
            else:
                x_ = self.Downsamplers[-(i+1)](src)
            x = self.Upblocks[i](torch.cat([x,x_],dim=1))
    
        x = self.out(torch.cat([x,src],dim=1))
        x = x.squeeze(1)
        print(x.shape)
        # x = torch.mean(x)
        # x = torch.mean(x,dim=1)
        # x  = x.unsqueeze(0)
        # x  = x.unsqueeze(0)
        x  = x[:,128,128]
        print(x.shape)
        return x.unsqueeze(0)
    

class FCN_Pad_Xaiver_Shap(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_Shap,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.out = nn.ConvTranspose2d(in_channels=64+self.channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-4,momentum=0.01)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-4,momentum=0.01)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-4,momentum=0.01)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-4,momentum=0.01)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-4,momentum=0.01)  
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-4,momentum=0.01)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-4,momentum=0.01)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-4,momentum=0.01)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-4,momentum=0.01)  

        self.elu = nn.ELU()

    def initial(self):
        torch.manual_seed(0)
        gain = nn.init.calculate_gain("leaky_relu",0.2)
        nn.init.xavier_uniform_(self.conv1.weight);nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.Tconv1.weight);nn.init.zeros_(self.Tconv1.bias)

        nn.init.xavier_uniform_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.Tconv2.weight);nn.init.zeros_(self.Tconv2.bias)

        nn.init.xavier_uniform_(self.conv3.weight);nn.init.zeros_(self.conv3.bias)
        nn.init.xavier_uniform_(self.Tconv3.weight);nn.init.zeros_(self.Tconv3.bias)

        nn.init.xavier_uniform_(self.conv4.weight);nn.init.zeros_(self.conv4.bias)
        nn.init.xavier_uniform_(self.Tconv4.weight);nn.init.zeros_(self.Tconv4.bias)

        nn.init.xavier_uniform_(self.out.weight);nn.init.zeros_(self.out.bias)
        print("All layer has been correctly initialized")

    def forward(self, inputs):

        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = self.elu(self.conv1(batch1))     
        batch2 = self.bn1(cnn1)


        cnn2 = self.elu(self.conv2(batch2))
        batch3= self.bn2(cnn2)

        cnn3 = self.elu(self.conv3(batch3))
        batch4 = self.bn3(cnn3)
        
        cnn4 = self.elu(self.conv4(batch4))
        batch5 = self.bn4(cnn4)



        tconv1 = self.elu(self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.tbn1(tconv1)
        
        tconv2 = self.elu(self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.tbn2(tconv2)
        
        tconv3 = self.elu(self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.tbn3(tconv3)
        
        tconv4 = self.elu(self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.tbn4(tconv4)

        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        print(x.shape)
        x = out.squeeze(1)
        print(x.shape)
        # x = torch.mean(x)
        # x = torch.mean(x,dim=1)
        # x  = x.unsqueeze(0)
        # x  = x.unsqueeze(0)
        x  = x[:,128,128]
        print(x.shape)
        return x.unsqueeze(0).unsqueeze(0)
    
