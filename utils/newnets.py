import torch 
from torch import nn
import torch.nn.functional as F
from utils.toolbox import periodic_padding
from utils.CBAM import CBAM
from utils.partialConv2d import PartialConv2d
class FCN_4(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_4,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize = knsize
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5,padding="same")
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize,padding="same")
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize,padding="same")
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=self.knsize)
        

        self.Tconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=self.knsize) 
        # self.Tconv2 = nn.ConvTranspose2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.Tconv3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.Tconv4 = nn.ConvTranspose2d(in_channels=256,out_channels=64,kernel_size=5)
        self.out = nn.ConvTranspose2d(in_channels=256,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(512,eps=1e-3,momentum=0.99)  
        
        self.tbn1 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99) 
    
        self.elu =nn.ELU()
    def forward(self,input):
        # input1 = input[:,1,:,:].unsqueeze(1)
        # input2 = input[:,-1,:,:].unsqueeze(1)
        # input = torch.cat([input1,input2],dim=1)
        x = periodic_padding(input,self.padding)
        x = self.initial_norm(x)
        
        x = self.conv1(x)
        x =F.elu(self.bn1(x))

        x = self.conv2(x)
        x =F.elu(self.bn2(x))

        
        x = self.conv3(x)
        x =F.elu(self.bn3(x))
     
        
        x = self.conv4(x)
        x =F.elu(self.bn4(x))

        x = self.Tconv1(x)
        x = self.out(x)
        
        return x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding
                ]

class FCN_4_Partial(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_4_Partial,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize = knsize
        self.padding = padding

        self.conv1 = PartialConv2d(in_channels=self.channels,out_channels=64,kernel_size=5,padding=1,stride=1)
        self.conv2 = PartialConv2d(in_channels=64,out_channels=128,kernel_size=self.knsize,padding=1,stride=1)
        self.conv3 = PartialConv2d(in_channels=128,out_channels=256,kernel_size=self.knsize,padding=1,stride=1)
        self.conv4 = PartialConv2d(in_channels=256,out_channels=512,kernel_size=self.knsize,padding=1)
        

        self.Tconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=self.knsize) 
        # self.Tconv2 = nn.ConvTranspose2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.Tconv3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.Tconv4 = nn.ConvTranspose2d(in_channels=256,out_channels=64,kernel_size=5)
        self.out = nn.ConvTranspose2d(in_channels=256,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(512,eps=1e-3,momentum=0.99)  
        
        self.tbn1 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99) 
    
        self.elu =nn.ELU()
    def forward(self,x):
        # x = periodic_padding(x,self.padding)
        x = self.initial_norm(x)
        
        x = self.conv1(x)
        x =F.elu(self.bn1(x))

        x = self.conv2(x)
        x =F.elu(self.bn2(x))

        
        x = self.conv3(x)
        x =F.elu(self.bn3(x))
     
        
        x = self.conv4(x)
        x =F.elu(self.bn4(x))

        x = self.Tconv1(x)
        x = self.out(x)
        return x  
        # return x[:,
        #         :,
        #         self.padding-1:-self.padding-1,
        #         self.padding-1:-self.padding-1
        #         ]


class Block(nn.Module):
    def __init__(self,in_channel,knsize,activate=True) -> None:
        super(Block,self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channel,in_channel,knsize,padding="same"),
                                        nn.BatchNorm2d(in_channel,eps=1e-3,momentum=0.99),
                                        nn.ELU())
        self.conv_block2 = nn.Sequential(
                                            nn.Conv2d(in_channel,in_channel,knsize,padding="same"),
                                            nn.BatchNorm2d(in_channel,eps=1e-3,momentum=0.99),
                                        )
        self.elu = nn.ELU()
        self.activate = activate
    def forward(self,x):
        res = x 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + res
        if self.activate:
            return self.elu(x)
        else: 
            return x        


class Res4(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(Res4,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize = knsize
        self.padding = padding

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.conv0 = nn.Conv2d(channels,64,kernel_size=5,stride=2,padding=5)
        self.block1 = Block(64,knsize=5,activate=True)

        self.conv1 = nn.Conv2d(64,128,stride=2,kernel_size=knsize,padding=knsize)
        
        self.block2 = Block(128,knsize=self.knsize,activate=True)
        self.conv2 = nn.Conv2d(128,256,stride=2,kernel_size=knsize,padding=knsize)
        
        self.block3 = Block(256,knsize=self.knsize,activate=True)
        # self.conv3 = nn.Conv2d(256,256,stride=2,kernel_size=knsize)
        
        
        # self.tconv1 = nn.ConvTranspose2d(256,256,knsize,stride=2)
        self.tconv2 = nn.ConvTranspose2d(256,128,knsize,stride=2,padding=knsize)
        self.tconv3 = nn.ConvTranspose2d(128,64,knsize,stride=2,padding=knsize-1)
        self.tconv4 = nn.ConvTranspose2d(64,4,5,stride=2,padding=5-1)
        self.tconv5 = nn.ConvTranspose2d(4,1,1,padding=0)
        

        self.bn0 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99) 
        self.tbn1 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99) 
        self.tbn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99) 
        self.tbn3 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99) 
        self.tbn4 = nn.BatchNorm2d(4,eps=1e-3,momentum=0.99) 
        self.tbn5 = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99) 
    def forward(self,x):
        import torch.nn.functional as F
        x = self.initial_norm(periodic_padding(x,self.padding))
        # x = self.initial_norm(x)
        x =  F.elu(self.bn0(self.conv0(x)))
        x = self.block1(x)
        
        x =  F.elu(self.bn1(self.conv1(x)))
        x = self.block2(x)
        
        x =  F.elu(self.bn2(self.conv2(x)))
        x = self.block3(x)



        # x =  F.elu(self.tbn1(self.tconv1(x)))
        # print(x.size())
        
        x =  F.elu(self.tbn2(self.tconv2(x)))
        
        x =  F.elu(self.tbn3(self.tconv3(x)))
        
        x =  F.elu(self.tbn4(self.tconv4(x)))
        
        x =  F.elu(self.tbn5(self.tconv5(x)))
        
        

        
        # return x
        return x[:,
                :,
                self.padding:-self.padding-1,
                self.padding:-self.padding-1
                ]



class PBlock(nn.Module):
    def __init__(self,in_channel,knsize,activate=True) -> None:
        super(PBlock,self).__init__()
        self.conv_block1 = nn.Sequential(PartialConv2d(in_channel,in_channel,knsize,padding="same"),
                                        nn.BatchNorm2d(in_channel,eps=1e-3,momentum=0.99),
                                        nn.ELU())
        self.conv_block2 = nn.Sequential(
                                            PartialConv2d(in_channel,in_channel,knsize,padding="same"),
                                            nn.BatchNorm2d(in_channel,eps=1e-3,momentum=0.99),
                                        )
        self.elu = nn.ELU()
        self.activate = activate
    def forward(self,x):
        res = x 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + res
        if self.activate:
            return self.elu(x)
        else: 
            return x 

class Res4_Partial(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(Res4_Partial,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize = knsize
        self.padding = padding

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.conv0 = PartialConv2d(channels,64,kernel_size=5,stride=2,padding=5)
        self.block1 = PBlock(64,knsize=5,activate=True)

        self.conv1 = PartialConv2d(64,128,stride=2,kernel_size=knsize,padding=knsize)
        
        self.block2 = PBlock(128,knsize=self.knsize,activate=True)
        self.conv2 = PartialConv2d(128,256,stride=2,kernel_size=knsize,padding=knsize)
        
        self.block3 = PBlock(256,knsize=self.knsize,activate=True)
        # self.conv3 = nn.Conv2d(256,256,stride=2,kernel_size=knsize)
        
        
        # self.tconv1 = nn.ConvTranspose2d(256,256,knsize,stride=2)
        self.tconv2 = nn.ConvTranspose2d(256+256,128,knsize,stride=2,padding=knsize-1)
        self.tconv3 = nn.ConvTranspose2d(128+128,64,knsize,stride=2,padding=knsize-1)
        self.tconv4 = nn.ConvTranspose2d(64+64,4,5,stride=2,padding=4)
        self.tconv5 = nn.ConvTranspose2d(4,1,1,padding=0)
        

        self.bn0 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        # self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99) 
        self.tbn1 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99) 
        self.tbn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99) 
        self.tbn3 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99) 
        self.tbn4 = nn.BatchNorm2d(4,eps=1e-3,momentum=0.99) 
        self.tbn5 = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99) 
    def forward(self,inputs):
        import torch.nn.functional as F
        inputs = inputs[:,-1,:,:].unsqueeze(1)
        print(inputs.size())
        inputs = self.initial_norm(periodic_padding(inputs,self.padding))
        batch1 = self.initial_norm(inputs)
        cnn1 =  F.elu(self.bn0(self.conv0(batch1)))
        x = self.block1(cnn1)
        # print(x.size())
        cnn2 =  F.elu(self.bn1(self.conv1(x)))
        x = self.block2(cnn2)
        # print(x.size())
        
        cnn3 =  F.elu(self.bn2(self.conv2(x)))
        x = self.block3(cnn3)
        # print(x.size())
        


        # x =  F.elu(self.tbn1(self.tconv1(x)))
        # print(x.size())
        
        x =  F.elu(self.tbn2(self.tconv2(torch.cat([cnn3,x],dim=1))))
        
        x =  F.elu(self.tbn3(self.tconv3(torch.cat([cnn2,x[:,:,1:,1:]],dim=1))))
        
        x =  F.elu(self.tbn4(self.tconv4(torch.cat([cnn1,x[:,:,1:-1,1:-1]],dim=1))))
        
        x =  F.elu(self.tbn5(self.tconv5(x)))
        
        

        
        # return x
        return x[:,
                 :,
                self.padding+1:-self.padding,
                self.padding+1:-self.padding
                ]








class FCN_Pad_Xaiver_CBAM(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out

from utils.CBAM import SpatialAttention,ChannelAttention
class FCN_Pad_SA(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_SA,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = SpatialAttention(knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = SpatialAttention(knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out

class FCN_Pad_CA(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_CA,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = ChannelAttention(256,1)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = ChannelAttention(128,1)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out


class FCN_Pad_Xaiver_CBAM_Batch(nn.Module):
    """
    Attention on Batch5 only
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_Batch,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        # inputs = inputs[:,-1,:,:].unsqueeze(1)
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        # cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        # batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out

class FCN_Pad_Xaiver_CBAM_Batch_Res(nn.Module):
    """
    Attention on Batch5  and Tconv1
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_Batch_Res,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        # cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)+cnn4
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out


class FCN_Pad_Xaiver_CBAM_Batch_Tconv(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_Batch_Tconv,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out


class FCN_Pad_Xaiver_CBAM_Tconv2(nn.Module):
    """
     tcb(tconv1) and tcb(cnn3)
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_Tconv2,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)
        cnn3 = self.tcb2(cnn3)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out

class FCN_Pad_Xaiver_CBAM_Out(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_Out,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        # cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        # batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.outcb(self.out(torch.concat([padded,batch9],dim=1)))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out


class FCN_Pad_Xaiver_CBAM_In(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_In,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAM(4,1,1)
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        inputs = self.incab(inputs)
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        # cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        # batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out


class FCN_Pad_Xaiver_CBAM_InAft(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_InAft,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAM(4,1,1)
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        
        padded = periodic_padding(inputs,self.padding)
        padded = self.incab(padded)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        
        batch5 = self.elu(self.bn4(cnn4))


        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out
class FCN_Pad_Xaiver_CBAM_InOut(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_InOut,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAM(4,1,1)
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        
        padded = periodic_padding(inputs,self.padding)
        padded = self.incab(padded)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        
        batch5 = self.elu(self.bn4(cnn4))


        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        tconv4 = self.tcb4(tconv4)
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out

class FCN_Pad_Xaiver_CBAM_InOut2(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM_InOut2,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAM(4,1,1)
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        
        padded = periodic_padding(inputs,self.padding)
        padded1 = self.incab(padded)
        batch1 = self.initial_norm(padded1)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        
        batch5 = self.elu(self.bn4(cnn4))


        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        # tconv4 = self.tcb4(tconv4)
        batch9 = self.elu(self.tbn4(tconv4))
        batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out
    



class FCN_Pad_Partial(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Partial,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = PartialConv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = PartialConv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = PartialConv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = PartialConv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, padded):
        # padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))

        batch5 = self.elu(self.bn4(cnn4))
        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))
        
        return x



class FCN_Pad_Partial_CBAM(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Partial_CBAM,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = PartialConv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = PartialConv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = PartialConv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = PartialConv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, padded):
        # padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        batch5 = self.cb4(batch5)
        
        
        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))


        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))
        
        return x


class FCN_Pad_Partial_CBAM_InOut2(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Partial_CBAM_InOut2,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAM(4,1,1)
        self.conv1 = PartialConv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = PartialConv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = PartialConv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = PartialConv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        
        # padded = periodic_padding(inputs,self.padding)
        padded1 = self.incab(inputs)
        batch1 = self.initial_norm(padded1)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        
        batch5 = self.elu(self.bn4(cnn4))


        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        # tconv4 = self.tcb4(tconv4)
        batch9 = self.elu(self.tbn4(tconv4))
        batch9 = self.tcb4(batch9)
        out = self.out(torch.concat([inputs,batch9],dim=1))

        #Corp the padding
        # out = x[:,
        #         :,
        #         self.padding:-self.padding,
        #         self.padding:-self.padding]
        
        return out
    


from utils.CBAM import CBAMRes
class FCN_Pad_CBAM_Res(nn.Module):
    """
    Add BN after tcb(tconv1) 
    """
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_CBAM_Res,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.incab = CBAMRes(4,1,1)
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        # self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        # self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        # self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        # self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        self.tcb4 = CBAMRes(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)
        # self.outcb = CBAM(1,1,1)
        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        
        
        # self.tcbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        
        # padded = periodic_padding(inputs,self.padding)
        padded1 = self.incab(inputs)
        batch1 = self.initial_norm(padded1)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        # cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        
        batch5 = self.elu(self.bn4(cnn4))


        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        batch6 = self.elu(self.tbn1(tconv1))
        # batch6 = self.tcbn1(self.tcb1(batch6))

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        batch7 =self.elu(self.tbn2(tconv2))
        # batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        # tconv4 = self.tcb4(tconv4)
        batch9 = self.elu(self.tbn4(tconv4))
        batch9 = self.tcb4(batch9)
        out = self.out(torch.concat([inputs,batch9],dim=1))

        #Corp the padding
        # out = x[:,
        #         :,
        #         self.padding:-self.padding,
        #         self.padding:-self.padding]
        
        return out
    