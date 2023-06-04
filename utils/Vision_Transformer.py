import torch 
from torch import nn
import math 
import torch.nn.functional as F 


class Transpose(nn.Module):
    def __init__(self,d0,d1) -> None:
        super(Transpose,self).__init__()

        self.d0 = d0; self.d1 = d1
    def forward(self,x):
        return x.transpose(self.d0,self.d1)
        

def attention(q,k,v,mask= None):
    """
    Implemention of attention and self-attention
    Args:
        q: tensor of shape [B, T_Q,D_K]
        k: tensor of shape [B, T_V,D_K]
        v: tensor of shape [B, T_V,D_V]

    Output:
        out tensor of shape [B,T_Q,D_V]
    """

    # batch_size
    B = q.shape[0]
    scale = math.sqrt(k.shape[2])
    att = torch.bmm(q,k.transpose(1,2))/scale
    if mask is not None:
        mask = mask.unsequeeze(0).repeat(B,1,1)
        att = torch.where(mask == 0, att, -math.inf*torch.ones_like(att))
    
    att = F.softmax(att,2)
    out = torch.bmm(att,v)

    return out

def create_causal_mask(size1,size2):
    mask = torch.ones(size1,size2)
    mask = torch.triu(mask,diagonal=0)

class Head(nn.Module):
    def __init__(self,h_dim, head_out_dim) -> None:
        super(Head,self).__init__()
        self.q_lin = nn.Linear(h_dim,head_out_dim,bias=False)
        self.k_lin = nn.Linear(h_dim,head_out_dim,bias=False)
        self.v_lin = nn.Linear(h_dim,head_out_dim,bias=False)
    
    def forward(self,q,k= None,v=None,mask = None):
        if k is None:
            k = q
        if v is None:
            v = k
        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)

        x = attention(q,k,v,mask=mask)

        return x 
    
class MultiHeadAttention(nn.Module):
    def __init__(self,h_dim,num_heads) -> None:
        super(MultiHeadAttention,self).__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
                Head(h_dim, h_dim//num_heads) for _ in range(num_heads)
                                    ])
        self.linear = nn.Linear((h_dim//num_heads)*num_heads, h_dim,bias=False)

    def forward(self,q,k=None,v = None, mask = None):
        x = [head(q,k,v,mask=mask) for head in self.heads]
        
        # [B,T,h_dim*num_heads]
        x = torch.cat(x,-1)
        x = self.linear(x)

        return x
    

class ViTransformerEncoderLayer(nn.Module):

    """ 
    An Image is Worth 16x16 words
    """
    def __init__(self,h_dim, num_heads, d_ff = 2048,dropout =0.0) -> None:
        super(ViTransformerEncoderLayer,self).__init__()
        
        self.norm0 = nn.LayerNorm(h_dim)
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim,num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
                                nn.Linear(h_dim,d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff,h_dim)
                                )


    def forward(self,x,mask=None):
        x_ = self.norm0(x)
        x = self.mha(x_,x_,x_,mask) + x
        x = self.norm1(x)+x
        x = self.ffn(x)
        x_ = self.norm2(x)
        
        return x + x_


class Patchlize(nn.Module):
    def __init__(self,patch_size,n_channel,h_dim) -> None:
        super(Patchlize,self).__init__()
        self.proc = nn.Sequential(
                                    nn.Unfold((patch_size,patch_size),
                                                stride=(patch_size,patch_size)),
                                    Transpose(1,2),
                                    nn.Linear(n_channel*patch_size*patch_size, h_dim),
                                    nn.LayerNorm(h_dim)
                                )
    def forward(self,x):
        x = self.proc(x)
        return x 



class ViTransformerEncoder(nn.Module):

    def __init__(self,num_layers,h_dim,num_heads,d_ff = 2048,
                 max_seq_steps = None, use_clf_token = False, dropout = 0.0, drop_out_emb = 0.0) -> None:
        super(ViTransformerEncoder,self).__init__()

        self.layers = nn.ModuleList([
                                        ViTransformerEncoderLayer(h_dim,num_heads,d_ff,dropout )
                                            for _ in range(num_layers)
                                    ])
        
        self.pos_emb = nn.Embedding(max_seq_steps,h_dim)
        self.use_clf_token = use_clf_token
        if self.use_clf_token:
            self.clf_token = nn.Parameter(torch.randn(1,h_dim))
        self.dropout_emb = nn.Dropout(drop_out_emb)

    def forward(self,x,mask=None):

        if self.use_clf_token:
            clf_token = self.clf_token.unsqueeze(0).repeat(x.shape[0],1,1)
            print(clf_token.shape)
            x = torch.cat([clf_token,x],1)
            if mask is not None:
                raise Exception("Error, clf_token with mask is not supported")
        
        embs = self.pos_emb.weight[:x.shape[1]]

        x +=embs

        x = self.dropout_emb(x)

        for layer in self.layers:
            x = layer(x,mask=mask)
        
        return x


class ViT(nn.Module):

    def __init__(self,patch_size,num_layers,h_dim,num_heads,n_channel,
                 d_ff = 2048, max_seq_length= None, use_clf_token= False, dropout= 0.0,dropout_emb = 0.0) -> None:
        
        super(ViT,self).__init__()
        self.n_channel = n_channel
        self.h_dim = h_dim
        self.proc = Patchlize(patch_size,n_channel,h_dim)

        self.enc = ViTransformerEncoder(num_layers, h_dim, num_heads,
                                        d_ff=d_ff,
                                        max_seq_steps=max_seq_length,
                                        use_clf_token=use_clf_token,drop_out_emb=dropout_emb,dropout=dropout)
        
        # self.out = nn.Conv1d((h_dim**2),h_dim**2,1)
        self.out = nn.Conv2d(1,1,1)
    def forward(self,x):
        # Gain batch size
        
        x_ = self.proc(x)
        x = self.enc(x_)
    
        # x = x.transpose(1,2)
        BS,l_dim,h_dim2 = x.shape
        x +=x_

        # if patch = 8
        # x = x.reshape(BS,1,l_dim//self.n_channel,l_dim//self.n_channel)
        # if patch = 16
        x = x.reshape((BS,1,h_dim2,h_dim2))
        x = self.out(x)
        return x

class ViTBackbone(nn.Module):
    def __init__(self,patch_size,num_layers,h_dim,num_heads,n_channel,
                 d_ff = 4096, max_seq_length= None, use_clf_token= False, dropout= 0.0,dropout_emb = 0.0) -> None:
        
        super(ViTBackbone,self).__init__()

        self.ps = patch_size
        self.proc = Patchlize(patch_size,n_channel,h_dim)

        self.enc = ViTransformerEncoder(num_layers, h_dim, num_heads,
                                        d_ff=d_ff,
                                        max_seq_steps=max_seq_length,
                                        use_clf_token=use_clf_token,drop_out_emb=dropout_emb,dropout=dropout)
        


        self.up = nn.Upsample(scale_factor=2,mode="bicubic")
        self.tconv0 = nn.Conv2d(256,256,kernel_size=3,padding="same") 
        self.tconv1 = nn.Conv2d(256,128,kernel_size=3,padding="same") 
        
        self.tconv2 = nn.Conv2d(128,64,3,padding="same")        
        self.tconv3 = nn.Conv2d(64,n_channel,3,padding="same")        
        self.tconv4 = nn.Conv2d(n_channel,1,1)    

        self.bn0 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)    
        self.bn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)    
        self.bn2 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)    
        self.bn3 = nn.BatchNorm2d(n_channel,eps=1e-3,momentum=0.99)    
        self.bn4 = nn.BatchNorm2d(1,eps=1e-3,momentum=0.99)    
    def forward(self,x):


        x = self.proc(x)
        x = self.enc(x)
        print(x.shape)
        BS, PS2 ,L_dim = x.shape
       
        x = x.reshape(BS, L_dim , self.ps , self.ps)

        print(x.shape)
        x_ = self.tconv0(x)
        x = F.elu(self.bn0(x_))
        x = self.up(x+x_)

        x_ = self.tconv1(x)
        x = F.elu(self.bn1(x_))
        x = self.up(x+x_)
        # print(x.shape)

        x_ = self.tconv2(x)
        x = F.elu(self.bn2(x_))
        x = self.up(x+x_)

        x_ = self.tconv3(x)
        x = F.elu(self.bn3(x_))
        x = self.up(x+x_)
        
        x = self.tconv4(x)
        x = F.elu(self.bn4(x))

        return x




_MODELS_CONFIG = {
    'vit-base': {'num_layers': 12, 'h_dim': 768, 'd_ff': 3072, 'num_heads': 12},
    'vit-large': {'num_layers': 24, 'h_dim': 1024, 'd_ff': 4096, 'num_heads': 16},
    'vit-huge': {'num_layers': 32, 'h_dim': 1280, 'd_ff': 5120, 'num_heads': 16},
                }





