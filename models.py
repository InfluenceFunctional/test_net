import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class MultiLayerPerceptron(nn.Module):
    def __init__(self,config, data_dimensions):
        super(MultiLayerPerceptron,self).__init__()
        # initialize constants and layers

        self.activations = nn.ModuleList(
            [Activation(config.model_activation, config.model_filters)
             for _ in range(config.model_layers)
             ]
        )

        self.linears = nn.ModuleList(
            [nn.Linear(config.model_filters, config.model_filters, bias = True)
             for _ in range(config.model_layers)
             ]
        )
        self.norms = nn.ModuleList(
            [Normalization(config.model_norm, config.model_filters)
             for _ in range(config.model_layers)
             ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.model_dropout)
             for _ in range(config.model_layers)
             ]
        )

        self.init_linear = nn.Linear(data_dimensions['dimension'], config.model_filters)
        self.init_activation = Activation(config.model_activation, config.model_filters)
        self.output_linear = nn.Linear(config.model_filters, 1, bias=False)


    def forward(self, x):
        x = self.init_activation(self.init_linear(x))
        for linear, activation, norm, dropout in zip(self.linears, self.activations, self.norms, self.dropouts):
            x = x + activation(norm(dropout(linear(x))))

        return self.output_linear(x)




class kernelActivation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        gamma = 1/(6*(self.dict[-1]-self.dict[-2])**2) # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma',torch.ones(1) * gamma) #

        #self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)


    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        return torch.exp(-self.gamma*(x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()

    def forward(self, input):
        return self.norm(input)



class transformer(nn.Module):
    def __init__(self,config, data_dimensions):
        super(transformer,self).__init__()

        self.activations = nn.ModuleList(
            [Activation(config.model_activation, config.model_filters)
             for _ in range(config.model_layers)
             ]
        )

        heads = max(config.model_filters // 16, 1)
        self.self_attentions = nn.ModuleList(
            [RelativeGlobalAttention(config.model_filters, heads, dropout = config.model_dropout, max_len = config.dataset_dimension)
             for _ in range(config.model_layers)
             ]
        )

        self.linears = nn.ModuleList(
            [nn.Linear(config.model_filters, config.model_filters, bias = True)
             for _ in range(config.model_layers)
             ]
        )
        self.norms = nn.ModuleList(
            [Normalization(config.model_norm, config.model_filters)
             for _ in range(config.model_layers)
             ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.model_dropout)
             for _ in range(config.model_layers)
             ]
        )

        self.init_linear = nn.Linear(1, config.model_filters)
        self.init_activation = Activation(config.model_activation, config.model_filters)
        self.output_linear = nn.Linear(config.model_filters, 1, bias=False)


    def forward(self,x):
        x = self.init_activation(self.init_linear(x.unsqueeze(-1)))
        for attention, linear, activation, norm, dropout in zip(self.self_attentions, self.linears, self.activations, self.norms, self.dropouts):
            x = x + activation(norm(dropout(linear(attention(x)))))

        x = x.sum(dim=-2)

        return self.output_linear(x)




class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
                .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel