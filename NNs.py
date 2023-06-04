from torch import nn
import torch
from transformer.FCN_CCT import FullSkip_FCN_CCT,FullSkip_Mul_FCN_CCT
from utils.networks import Init_Conv, FCN_Pad_Xaiver_gain
from utils.newnets import FCN_Pad_Xaiver_CBAM2, FCN_4
from utils.CBAM import CBAM
from torch import nn
import torch


class HeatFormer_mut(nn.Module):
    def __init__(self,
                 img_size=256,
                 embedding_dim=256,
                #  n_input_channels=1,
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
        
        super(HeatFormer_mut,self).__init__()

        self.skipvit = FullSkip_Mul_FCN_CCT(img_size=img_size,
                                        embedding_dim=embedding_dim,
                                        # n_input_channels=n_input_channels,
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
        
        self.fcn  = FCN_Pad_Xaiver_gain(img_size,img_size,3,3,8)
        # self.actf = nn.Sigmoid()
        # self.actf2 = nn.Softmax2d()
        self.vit_bn = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99)
        
        self.out_conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1)
        self.bn = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99)
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.fcn.apply(Init_Conv)
    



    def forward(self,x): 
                  
            x_pr = self.vit_bn(self.skipvit(x[:,2:3,:,:]))
     
            x_vel = self.fcn(x[:,:3,:,:])
            

            x =  self.out_conv(  torch.cat([x_pr,x_vel],dim=1)  )
            return x_vel

class HeatFormer_passive(nn.Module):
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
        
        super(HeatFormer_passive,self).__init__()

        self.skipvit = FullSkip_FCN_CCT(img_size=img_size,
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
        
        self.fcn  = FCN_Pad_Xaiver_CBAM2(img_size,img_size,3,3,8)
        self.actf = nn.ELU()

        self.cbbn = nn.BatchNorm2d(3,eps=1e-3,momentum=0.99)


        self.vit_bn = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99)
        
        self.out_conv = FCN_4(img_size,img_size,2,3,8)
        self.bn = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99)
        self.act = nn.PReLU()
        
        self.out_conv.apply(Init_Conv)
        self.fcn.apply(Init_Conv)
    



    def forward(self,x): 
                  
            x_pr = self.actf(self.vit_bn(self.skipvit(x)))
          
            x_vel = self.fcn(torch.cat([x[:,:3,:,:]],dim=1))
            
            x =  self.out_conv( torch.cat([x_vel,x_pr], dim = 1) )
            return x