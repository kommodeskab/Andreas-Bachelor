# all credits to: https://github.com/johndpope/Simplified-Diffusion-Schrodinger-Bridge

import torch

class AdaLayerNorm(torch.nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2, -1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class LayerNorm(torch.nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x

class ResBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, n_cond=1000):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = torch.nn.Linear(self.dim_in, self.dim_out, bias=bias)
        if n_cond > 0:
            self.norm = AdaLayerNorm(self.dim_out, n_cond)
        else:
            self.norm = LayerNorm(self.dim_out, n_cond)
        self.activation = torch.nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = torch.nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x, t):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out, t)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = torch.nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x, t):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    


class ResMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=ResBlock, n_cond=1000):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias, n_cond=n_cond))
            else:
                net.append(torch.nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = torch.nn.ModuleList(net)
        
    
    def forward(self, x, t):

        for l in range(self.num_layers - 1):
            x = self.net[l](x, t)
        x = self.net[-1](x)
            
        return x
    
if __name__ == "__main__":
    net = ResMLP(2, 2, 128, 3, n_cond=100)
    x = torch.randn(10, 2)
    t = torch.randint(0, 100, (10, ))
    out = net(x, t)
    print(out.shape)