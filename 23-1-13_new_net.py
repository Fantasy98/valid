import torch 
from utils.newnets import Res4_Partial
from utils.networks import Init_Conv
model = Res4_Partial(256,256,1,3,8)
model.apply(Init_Conv)
input = torch.zeros(size=(1,4,256,256))
out = model(input)
print(out.size())