import torch
def periodic_padding(input:torch.Tensor,padding:int):
    if len(input.size()) !=4:
        print("The tenor does not fit the size!")
        return 
    else:
        M1 = torch.cat([input[:,:,:, -padding: ],input,input[:,:,:, 0:padding ]],dim=-1)
        M1 = torch.cat([M1[:,:, -padding: ,:],M1,M1[:,:, 0:padding ,:]],dim=-2)
        return M1
