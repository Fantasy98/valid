import torch 
from torch import nn 
from transformer.sampler import DepthToSpace
class ConvBlock(nn.Module):
    def __init__(self,in_channel,up_factor=2):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                                out_channels= in_channel,
                                kernel_size=3,padding="same")

        self.act = nn.PReLU(init=0)
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.upsampler = DepthToSpace(block_size=2)
        nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self,x):
        x = self.conv1(x)
        x = self.act(self.BN(x))
        x = self.upsampler(x)
        return x 
    
class ConvBlockCat(nn.Module):
    def __init__(self,in_channel,up_factor=2):
        super(ConvBlockCat,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*in_channel,
                                out_channels= in_channel,
                                kernel_size=3,padding="same")

        self.act = nn.PReLU(init=0)
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.upsampler = DepthToSpace(block_size=2)
        nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self,x):
        x = self.conv1(x)
        x = self.act(self.BN(x))
        x = self.upsampler(x)
        return x 
    