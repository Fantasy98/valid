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
        self.out = nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=self.knsize)
        
        # self.elu = nn.ELU()
        # for i in range(len(self.features)):
        #     setattr(self,f"BN{i+1}",nn.Sequential([nn.BatchNorm2d(self.features[i]),self.elu]))
        
        self.bn1 = nn.Sequential( nn.BatchNorm2d(64),nn.ELU()  )
        self.bn2 = nn.Sequential( nn.BatchNorm2d(128),nn.ELU()  )
        self.bn3 = nn.Sequential( nn.BatchNorm2d(256),nn.ELU()  )
        self.bn4 = nn.Sequential( nn.BatchNorm2d(256),nn.ELU()  )
    
    def forward(self, inputs):
        cnn1 = self.conv1(inputs)
        # print(cnn1.size())
        x = self.bn1(cnn1)
        cnn2 = self.conv2(x)
        # print(cnn2.size())
        x = self.bn2(cnn2)
        cnn3 = self.conv3(x)
        # print(cnn3.size())
        x = self.bn3(cnn3)
        cnn4 = self.conv4(x)
        # print(cnn4.size())
        x = self.bn4(cnn4)
        

        # print("Upsample")
        
        tconv1 = self.Tconv1(torch.concat([x,cnn4],dim=1))
        # print(tconv1.size())
        x = self.bn4(tconv1)
        tconv2 = self.Tconv2(torch.concat([x,cnn3],dim=1))
        # print(tconv2.size())
        x =self.bn3(tconv2)
        tconv3 = self.Tconv3(torch.concat([x,cnn2],dim =1 ))
        x = self.bn2(tconv3)
        # print(tconv3.size())
        
        tconv4 = self.Tconv4(torch.concat([x,cnn1],dim=1))
        # print(tconv4.size())
        
        x = self.bn1(tconv4)

        x = self.out(x)
        
        #Corp the padding
        out = x[:,:,1:-1,1:-1]
        return out