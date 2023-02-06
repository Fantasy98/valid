def gumbel_rsample(shape):
    import torch 
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    gumbel = torch.distributions.gumbel.Gumbel(zero,one).rsample
    return gumbel(shape)

x = gumbel_rsample(shape=(1,4,256,256))

print(x.size())
print(x)