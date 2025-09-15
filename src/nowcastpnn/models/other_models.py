# LSTM and standard MLP as comparison models
import torch
import torch.nn as nn
from nowcastpnn.distributions.NegativeBinomial import NegBin as NB

class NowcastLSTM(nn.Module):
    """
    Causal LSTM over daily delay-profiles.
    Input:
      - x: [B, T=past_units, D=max_delay] counts
      - mask: optional [B, T, D] (1=observed, 0=unobserved). If provided and use_mask=True,
              we append a per-step observed-ratio channel to x.
    """
    def __init__(
        self,
        past_units: int = 40,
        max_delay: int = 40,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout_lstm: float = 0.0,
        use_mask: bool = True,
        head_hidden: tuple[int, int] = (64, 32),
        dropout_head: tuple[float, float] = (0.15, 0.10),
        const: float = 10_000.0,
    ):
        super().__init__()
        in_size = max_delay + (1 if use_mask else 0)
        self.use_mask = use_mask
        self.const = const

        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # causal
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        self.bnorm = nn.BatchNorm1d(hidden_size)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_head[0])
        self.drop2 = nn.Dropout(dropout_head[1])
        self.fc1 = nn.Linear(hidden_size, head_hidden[0])
        self.fc2 = nn.Linear(head_hidden[0], head_hidden[1])
        self.fc_out = nn.Linear(head_hidden[1], 2)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, T, D]; mask: [B, T, D]
        if self.use_mask:
            if mask is None:
                # assume zeros mean unobserved; ratio= (x>0).mean over delay
                obs_ratio = (x > 0).float().mean(dim=-1, keepdim=True)
            else:
                obs_ratio = mask.float().mean(dim=-1, keepdim=True)
            x_in = torch.cat([x.float(), obs_ratio], dim=-1)  # [B, T, D+1]
        else:
            x_in = x.float()

        _, (h_n, _) = self.lstm(x_in)          # h_n: [num_layers, B, H]
        feat = h_n[-1]                          # last layer, last time step: [B, H]
        feat = self.act(self.bnorm(feat))
        feat = self.drop1(self.act(self.fc1(feat)))
        feat = self.drop2(self.act(self.fc2(feat)))
        params = self.fc_out(feat)              # [B, 2]

        lbda = self.const * self.softplus(params[:, 0])
        phi = (self.const ** 2) * self.softplus(params[:, 1]) + 1e-5
        dist = NB(lbda=lbda, phi=phi)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)


class NowcastMLP(nn.Module):
    """
    MLP over flattened triangle.
    Input:
      - x: [B, T, D] -> flattened to [B, T*D]
      - mask: optional [B, T, D], concatenated as a second flattened channel if use_mask=True.
    """
    def __init__(
        self,
        past_units: int = 40,
        max_delay: int = 40,
        hidden: tuple[int, int, int] = (512, 256, 128),
        use_mask: bool = True,
        dropout: float = 0.1,
        const: float = 10_000.0,
    ):
        super().__init__()
        in_dim = past_units * max_delay * (2 if use_mask else 1)
        self.use_mask = use_mask
        self.const = const

        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 2)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, t, d = x.shape
        x_flat = x.reshape(b, t * d).float()
        if self.use_mask:
            if mask is None:
                mask_flat = (x > 0).float().reshape(b, t * d)
            else:
                mask_flat = mask.float().reshape(b, t * d)
            x_in = torch.cat([x_flat, mask_flat], dim=1)
        else:
            x_in = x_flat

        params = self.net(x_in)                 # [B, 2]
        lbda = self.const * self.softplus(params[:, 0])
        phi = (self.const ** 2) * self.softplus(params[:, 1]) + 1e-5
        dist = NB(lbda=lbda, phi=phi)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
