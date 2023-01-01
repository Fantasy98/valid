from utils.networks import FCN_pad
import torch

height = 256;width = 256;channels=4;knsize =3;padding = 8

model = FCN_pad(height,width,channels,knsize,padding)


print(model.eval())

input = torch.randint(0,2,(1,channels,height,width)).float()

with torch.no_grad():
    out = model(input)

print(out.size())