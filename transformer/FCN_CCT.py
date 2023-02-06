
import torch.nn as nn
from transformer.sampler import SpaceToDepth
from transformer.cct import CCT
from transformer.ConvBlock import ConvBlock,ConvBlockCat
import math
import torch

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

        return x

