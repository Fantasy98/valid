import torch 
from torch import nn 

class model(nn.Module):
    def __init__(self,input_size,inner_size,output_size) -> None:
        super(model,self).__init__()
        self.input_size = input_size
        self.inner_size = inner_size
        self.output_size = output_size
        relu = nn.ReLU()
        mlp1 = nn.Linear(in_features=self.input_size,out_features=self.inner_size)
        mlp2 = nn.Linear(in_features=self.inner_size,out_features=self.output_size)
        setattr(self,"relu",relu)
        setattr(self,"mlp1",mlp1)
        setattr(self,"mlp2",mlp2)


    def forward(self,inputs):
        linear1 = getattr(self,"mlp1")(inputs)
        relu1 = getattr(self,"relu")(linear1)
        linear2 = getattr(self,"mlp2")(relu1)
        output = getattr(self,"relu")(linear2)
        return output


m = model(16,32,16)
print(m.eval())
inputs = torch.ones(size=(1,16))
out = m(inputs)
print(out)
print(out.size())

# Check the a parameters in modules
for name in m.state_dict():
    print(name)

# A way to get parameters of weight of a layer
x = m.get_parameter("mlp1.weight")
print("The weight of mlp1")
print(x)
nn.init.zeros_(m.get_parameter("mlp1.weight"))
print("After initialized by 0 ")
x = m.get_parameter("mlp1.weight")
print(x)
optimizer = torch.optim.Adam(params=m.parameters())
# So now we understand that the model.eval will only evaluate the element in initial, not in forward function. 
# If you want to summary the model as in the 