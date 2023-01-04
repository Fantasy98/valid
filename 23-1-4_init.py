import torch 
from torch import nn 

conv= nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
print(conv.weight)
print(conv.bias)
nn.init.xavier_uniform_(conv.weight)
nn.init.zeros_(conv.bias)
print(conv.weight)
print(conv.bias)

# nn.init.kaiming_normal_(conv.weight)
# nn.init.zeros_(conv.bias)
# print(conv.weight)
# print(conv.bias)

bn = nn.BatchNorm2d(num_features=1)
print(bn.bias)
print(bn.weight)
