import torch.nn as nn
import torch
from NegativeBinomial import NegBin as NB

## For matrix-like (two-dimensional) input data
class NowcastPNNDaily(nn.Module):
    def __init__(self, past_units = 30, max_delay = 40, hidden_units = [16, 8], conv_channels = [10, 1, 1]):
        super().__init__()
        self.past_units = past_units
        self.max_delay = max_delay
        self.final_dim = past_units# * (2**len(conv_channels))
        self.conv1 = nn.Conv1d(self.max_delay, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        #self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=7, padding="same")
        #self.conv4 = nn.Conv1d(conv_channels[2], conv_channels[3], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.past_units, self.past_units)#, nn.Linear(self.past_units, self.past_units)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])
        #self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[-1], 2)
        #self.fcpoi = nn.Linear(hidden_units[1], 1)
        self.const = 10000

        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.max_delay), nn.BatchNorm1d(num_features=conv_channels[0])#, nn.BatchNorm1d(num_features=conv_channels[1])#, nn.BatchNorm1d(num_features=conv_channels[2])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])#, nn.BatchNorm1d(num_features=hidden_units[2])
        #self.bnorm7 = nn.BatchNorm1d(num_features=hidden_units[1])
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.past_units for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.max_delay, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(0.1), nn.Dropout(0.1)#, nn.Dropout(0.1) # 0.4, 0.4, 0.2 great performance, but too wide CIs
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
    
    def forward(self, x): ## Feed forward function, takes input of shape [batch, past_units, max_delay]
        #x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = x.float() # maybe uncomment
        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        #x = self.act(self.conv3(self.bnorm3(x)))
        #x = self.act(self.conv4(self.bnorm4(x)))
        x = torch.squeeze(x, dim = 1) # only squeeze max_delay dimension in case a single obs (batch size of 1) is passed

        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        #x = self.drop3(x)
        #x = self.act(self.fc5(self.bnorm7(x)))
        #x = self.drop3(x)
        x = self.fcnb(self.bnorm_final(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = (self.const**2)*self.softplus(x[:, 1])+1e-5)
        #x = self.fcpoi(self.bnorm7(x))
        #dist = torch.distributions.Poisson(rate=self.const*self.softplus(x))
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)

## For summed (one-dimensional) input data
class PNNSumDaily(nn.Module):
    def __init__(self, past_units = 45, max_delay = 45, n_layers = 3, hidden_units = [64, 32, 16]):
        super().__init__()
        self.past_units = past_units
        self.max_delay = max_delay
        self.attfc1 = nn.Linear(self.past_units, self.past_units)
        self.attfc2 = nn.Linear(self.past_units, self.past_units)
        self.attfc3 = nn.Linear(self.past_units, self.past_units)
        self.attfc4 = nn.Linear(self.past_units, self.past_units)
        # Should iterate over n_layers for more robust solution and make ModuleList
        self.fc3 = nn.Linear(past_units, hidden_units[0])
        self.fc4 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcpoi = nn.Linear(hidden_units[2], 1)
        self.fcnb = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.lnorm1, self.lnorm2, self.lnorm3, self.lnorm4 = nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1])
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.attn4 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.2), nn.Dropout(0.4), nn.Dropout(0.2)
        self.softplus = nn.Softplus()
        self.relu, self.silu = nn.ReLU(), nn.SiLU()
    
    def forward(self, x):
        #print(x.size())
        #x = x + self.pos_embed(x)
        x = torch.unsqueeze(x, -1)#.permute(0, 2, 1)
        #print(f"Before att layers: {x.size()}")
        x_add = x.clone()
        x = self.lnorm1(x)
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.attfc1(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x + x_add
        """x_add = x.clone()
        x = self.lnorm2(x)
        x = self.attn2(x, x, x, need_weights = False)[0]
        x = self.attfc2(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x+x_add
        x_add = x.clone()
        x = self.lnorm3(x)
        x = self.attn3(x, x, x, need_weights = False)[0]
        x = self.attfc3(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x+x_add
        x_add = x.clone()
        x = self.lnorm4(x)
        x = self.attn4(x, x, x, need_weights = False)[0]
        x = self.attfc4(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x+x_add"""
        x = x.permute(0, 2, 1) # [batch, past_units, 1] -> [batch, 1, past_units], so can take past_units
        x = torch.squeeze(x)
        x = self.silu(self.fc3(self.bnorm1(x)))
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcnb(self.bnorm4(x))
        #dist = torch.distributions.Poisson(rate=1000*self.softplus(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)

nowcast_pnn_dow = None # run once and save
## Let it train once, then keep embedding and freeze them, from pre-trained, save somewhere
torch.save(nowcast_pnn_dow.embed.weight, f"./weights/embedding_weights")