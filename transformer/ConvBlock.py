import torch 
from torch import nn 
from utils.networks import Init_Conv
from transformer.sampler import DepthToSpace, SpaceToDepth
from utils.CBAM import CBAM
class ConvBlockOrg(nn.Module):
    def __init__(self,in_channel):
        super(ConvBlockOrg,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                                out_channels= in_channel,
                                kernel_size=3,padding="same")

        # self.act = nn.PReLU(init=0)
        self.cbam = CBAM(in_channel,1,3)
        self.act = nn.ELU()
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.downsampler =SpaceToDepth(block_size=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.cbam.apply(Init_Conv)

    def forward(self,x):
        x = self.conv1(x)
        # x = self.act(self.BN(x))
        x = self.cbam(x)
        x = self.act(self.BN(x))
        x = self.downsampler(x)
        return x 

class ConvBlock(nn.Module):
    def __init__(self,in_channel,up_factor=2):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                                out_channels= in_channel,
                                kernel_size=3,padding="same")

        self.act = nn.PReLU(init=0)
        # self.act = nn.ELU()
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.upsampler = DepthToSpace(block_size=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

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
        # self.act = nn.ELU()
        # eps default = 1e-2
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.upsampler = DepthToSpace(block_size=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.act(self.BN(x))
        x = self.upsampler(x)
        return x 

class ConvBlockMul(nn.Module):
    def __init__(self,in_channel,up_factor=2):
        super(ConvBlockMul,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*in_channel,
                                out_channels= in_channel,
                                kernel_size=3,padding="same")

        self.act = nn.ELU()
        # self.cbam = CBAM(in_channel,1,3)
        # eps default = 1e-2
        self.BN = nn.BatchNorm2d(in_channel,eps=0.01,momentum=0.99)
        self.upsampler = DepthToSpace(block_size=2)
        # self.upsampler = nn.Upsample(scale_factor=2,mode="bicubic")
        # nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        # self.cbam.apply(Init_Conv)

    def forward(self,x):
        x = self.conv1(x)
        # x = self.cbam(x)
        x = self.act(self.BN(x))
        # x = self.cbam(x)
        # x = self.act(self.BN(x))
        x= self.upsampler(x)
        return x 