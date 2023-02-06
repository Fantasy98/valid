import torch 
from torch import nn 
from utils.CBAM import CBAM
from utils.networks import Init_Conv
import torch.nn.functional as F
def gumbel_rsample(shape):
    import torch 
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    gumbel = torch.distributions.gumbel.Gumbel(zero,one).rsample
    return gumbel(shape)


class Dense_Gate(nn.Module):
    def __init__(self, expert_list:nn.ModuleList, n_expert,device) -> None:

        super(Dense_Gate,self).__init__()
        self.w_gate = nn.Conv2d(in_channels=4,out_channels=n_expert,kernel_size=1)
        self.cbam = CBAM(n_expert,1,1)

        torch.nn.init.xavier_uniform_(self.w_gate.weight)
        torch.nn.init.zeros_(self.w_gate.bias)
        self.cbam.apply(Init_Conv)
        self.expert_list = expert_list
        self.device = device
        self.n_expert = n_expert
    def forward(self,x):
        pred_final = 0
        gate_weight = []
        pred_list = []

        ge = self.cbam(self.w_gate(x))
        gumble_noise = gumbel_rsample(ge.size()).to(self.device)

        masks = F.gumbel_softmax( ge+gumble_noise,tau=1,dim=1)
        for i in range(self.n_expert):
            pred = self.expert_list[i](x)
            pred_list.append(pred)
            out_mask = masks[:,i,:,:]
            gate_weight.append(out_mask)
        
        for idx,items in enumerate(gate_weight):
   
            pred_final += pred_list[idx]*gate_weight[idx] 

        return pred_final
