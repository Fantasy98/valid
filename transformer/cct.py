
import torch.nn as nn
from transformer.transformer import Transformer
from transformer.tokenizer import Tokenizer
import math
class CCT(nn.Module):
    def __init__(self,
                 img_size=256,
                 embedding_dim=256*4,
                 n_input_channels=4,
                 n_conv_layers=2,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.01,
                 stochastic_depth=0.01,
                 num_layers=3,
                 num_heads=8,
                 mlp_ratio=4.0,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.PReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.encoder = Transformer(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        # print(f"Token size{x.shape}")
        return self.encoder(x)

