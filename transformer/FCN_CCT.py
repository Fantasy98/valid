
import torch.nn as nn
from transformer.cct import CCT
from transformer.ConvBlock import ConvBlock,ConvBlockCat, ConvBlockMul, ConvBlockOrg
import math
from transformer.sampler import SpaceToDepth
import torch
import torch.nn.functional as F
class FCN_CCT(nn.Module):
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


        super(FCN_CCT,self).__init__()

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
        # self.convin = ConvBlock(in_channel=embedding_dim)
        self.Upblocks = nn.ModuleList([ConvBlock(in_channel= int( embedding_dim*(0.25**(i)) )  )
                                        for i in range(self.times)])
        self.out = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=1,padding="same")
        nn.init.xavier_uniform_(self.out.weight)
        
    
    def forward(self,x):
        enc_x = self.CCTEncoder(x)
        
        B,P2,H_dim = enc_x.shape
        P = int(math.sqrt(P2))
        x = enc_x.reshape(B,H_dim,P,P)
        
        for block in self.Upblocks:
            x = block(x)
    
            
        x = self.out(x)

        return x

        
class Skip_FCN_CCT(nn.Module):
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


        super(Skip_FCN_CCT,self).__init__()

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
        # self.convin = ConvBlock(in_channel=embedding_dim)
        self.Upblocks = nn.ModuleList([ConvBlock(in_channel= int( embedding_dim*(0.25**(i)) )  )
                                        for i in range(self.times)])
        self.out = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,padding="same")
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
        #eps = 0.01
        self.BN = nn.BatchNorm2d(n_input_channels,eps=0.01,momentum=0.99)
    def forward(self,src):
        enc_x = self.CCTEncoder(src)
        
        B,P2,H_dim = enc_x.shape
        P = int(math.sqrt(P2))
        x = enc_x.reshape(B,H_dim,P,P)
        
        for block in self.Upblocks:
            x = block(x)
    
            
        x = self.out(torch.cat([x,self.BN(src)],dim=1))

        return x


class FullSkip_FCN_CCT(nn.Module):
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


        super(FullSkip_FCN_CCT,self).__init__()

        self.CCTEncoder = CCT(img_size=img_size,
                                embedding_dim=embedding_dim,
                                n_input_channels=n_input_channels,
                                n_conv_layers=n_conv_layers,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                pooling_kernel_size=pooling_kernel_size,
                                pooling_stride=pooling_stride,
                                pooling_padding=pooling_padding,
                                dropout=dropout,
                                attention_dropout=attention_dropout,
                                stochastic_depth=stochastic_depth,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                positional_embedding=positional_embedding,
                                *args, **kwargs)
        
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.times = embedding_dim//img_size
        # self.Inconv = ConvBlockCat(in_channel=2*embedding_dim)

        self.Downsamplers= nn.ModuleList([ 
                                          
                                            SpaceToDepth(block_size=2**(i+1))
                                        #  
                                          
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

        return x


class FullSkip_Mul_FCN_CCT(nn.Module):
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


        super(FullSkip_Mul_FCN_CCT,self).__init__()

        self.CCTEncoder = CCT(img_size=img_size,
                                embedding_dim=embedding_dim,
                                n_input_channels=n_input_channels,
                                n_conv_layers=n_conv_layers,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                pooling_kernel_size=pooling_kernel_size,
                                pooling_stride=pooling_stride,
                                pooling_padding=pooling_padding,
                                dropout=dropout,
                                attention_dropout=attention_dropout,
                                stochastic_depth=stochastic_depth,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                positional_embedding=positional_embedding,
                                *args, **kwargs)
        
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.times = embedding_dim//img_size
        downlist = [
                    ConvBlockOrg(in_channel=int(img_size*(0.25**(i))))
                    # ConvBlockOrg(in_channel=int((4**(i))))
                    for i in range(self.times)
                     ]
        downlist.reverse()
        self.DownBlocks = nn.ModuleList(downlist)
        print(self.DownBlocks)
        self.Upblocks = nn.ModuleList([ConvBlockMul(in_channel= int(embedding_dim*(0.25**(i)) ) )
                                        for i in range(self.times)])
        # self.out = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,padding="same")
        self.out = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,padding="same")
        self.out2 = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=1,padding="same")
        nn.init.xavier_uniform_(self.out.weight)
        # nn.init.xavier_uniform_(self.out2.weight)
        nn.init.zeros_(self.out.bias)
        # nn.init.zeros_(self.out2.bias)
        
        
        self.BN = nn.BatchNorm2d(n_input_channels,eps=0.01,momentum=0.99)
        self.BN_out = nn.BatchNorm2d(4,eps=0.01,momentum=0.99)
        self.act_out = nn.ELU()
        self.fact = nn.Softmax2d()
    def forward(self,src):
        src = self.BN(src)
        enc_x = self.CCTEncoder(src)
        B,P2,H_dim = enc_x.shape
        P = int(math.sqrt(P2))
        x_up = enc_x.reshape(B,H_dim,P,P)
        # print(f"In model, encoder output {x_up.shape}")
        downs = []
        x_down = src
        for i in range(self.times):
            x_down = self.DownBlocks[i](x_down)
            # print(x_down.shape)
            downs.append(x_down)
        downs.reverse()


        for i in range(self.times):
            x_up =  self.Upblocks[i]( torch.cat([ downs[i],x_up  ],dim=1))  

        x = self.out(torch.cat([x_up,src],dim=1))
        # x = self.act_out(self.BN_out(x))
        x = self.out2(x)
    

        return x
    
