import torch 
from torch import nn 
import matplotlib.pyplot as plt 

class FCN(nn.Module):
    def __init__(self, height,width,channels,knsize) -> None:
        super(FCN,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=self.knsize)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=256,kernel_size=self.knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256*2,out_channels=256,kernel_size=self.knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=256+128,out_channels=128,kernel_size=self.knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=128+64,out_channels=64,kernel_size=self.knsize)
        self.out = nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=1)
        
        # self.elu = nn.ELU()
        # for i in range(len(self.features)):
        #     setattr(self,f"BN{i+1}",nn.Sequential([nn.BatchNorm2d(self.features[i]),self.elu]))
        
        self.bn1 = nn.Sequential( nn.BatchNorm2d(64),nn.ELU()  )
        self.bn2 = nn.Sequential( nn.BatchNorm2d(128),nn.ELU()  )
        self.bn3 = nn.Sequential( nn.BatchNorm2d(256),nn.ELU()  )
        self.bn4 = nn.Sequential( nn.BatchNorm2d(256),nn.ELU()  )
    
    def forward(self, inputs):
        cnn1 = self.conv1(inputs)
        
        x = self.bn1(cnn1)
        cnn2 = self.conv2(x)
        
        x = self.bn2(cnn2)
        cnn3 = self.conv3(x)
        
        x = self.bn3(cnn3)
        cnn4 = self.conv4(x)
        
        x = self.bn4(cnn4)
        

                
        tconv1 = self.Tconv1(torch.concat([x,cnn4],dim=1))
        
        x = self.bn4(tconv1)
        tconv2 = self.Tconv2(torch.concat([x,cnn3],dim=1))
        
        x =self.bn3(tconv2)
        tconv3 = self.Tconv3(torch.concat([x,cnn2],dim =1 ))
        x = self.bn2(tconv3)
        
        tconv4 = self.Tconv4(torch.concat([x,cnn1],dim=1))
        
        x = self.bn1(tconv4)

        x = self.out(x)
        
        #Corp the padding
        out = x[:,:,1:-1,1:-1]
        return out

from utils.toolbox import periodic_padding
class FCN_pad(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_pad,self).__init__()
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

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-5,momentum=0.001)
        self.bn1 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(64,eps=1e-5,momentum=0.001)  ) 
        self.bn2 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(128,eps=1e-5,momentum=0.001)  )
        self.bn3 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(256,eps=1e-5,momentum=0.001)  )
        self.bn4 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(256,eps=1e-5,momentum=0.001)  )
        
        self.tbn1 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(128,eps=1e-5,momentum=0.001)  )
        self.tbn2 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(256,eps=1e-5,momentum=0.001)  )
        self.tbn3 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(256,eps=1e-5,momentum=0.001)  )
        self.tbn4 = nn.Sequential( nn.ELU(),nn.BatchNorm2d(64,eps=1e-5,momentum=0.001)  )
    
    def forward(self, inputs):

        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = self.conv1(batch1)     
        batch2 = self.bn1(cnn1)


        cnn2 = self.conv2(batch2)
        batch3= self.bn2(cnn2)

        cnn3 = self.conv3(batch3)
        batch4 = self.bn3(cnn3)
        
        cnn4 = self.conv4(batch4)
        batch5 = self.bn4(cnn4)



        tconv1 = self.Tconv1(torch.concat([cnn4,batch5],dim=1))
        batch6 = self.tbn1(tconv1)
        
        tconv2 = self.Tconv2(torch.concat([cnn3,batch6],dim=1))
        batch7 =self.tbn2(tconv2)
        
        tconv3 = self.Tconv3(torch.concat([cnn2,batch7],dim =1 ))
        batch8 = self.tbn3(tconv3)
        
        tconv4 = self.Tconv4(torch.concat([cnn1,batch8],dim=1))
        batch9 = self.tbn4(tconv4)

        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out
    
