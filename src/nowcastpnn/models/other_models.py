# LSTM and standard MLP as comparison models
import os
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
        dropout_lstm: float = 0.0,
        num_layers: int = 1,
        hidden_units: tuple[int, int] = (64, 32),
        dropout_probs: tuple[float, float] = (0.15, 0.10),
        const: float = 10_000.0,
    ):
        super().__init__()
        in_size = max_delay
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
        self.drop1 = nn.Dropout(dropout_probs[0])
        self.drop2 = nn.Dropout(dropout_probs[1])
        self.fc1 = nn.Linear(hidden_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_out = nn.Linear(hidden_units[1], 2)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, dow: torch.Tensor | None = None):
        # x: [B, T, D]; mask: [B, T, D]
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


class NowcastLSTMDOW(nn.Module):
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
        dropout_lstm: float = 0.0,
        num_layers: int = 1,
        hidden_units: tuple[int, int] = (64, 32),
        dropout_probs: tuple[float, float] = (0.15, 0.10),
        embedding_dim = 10,
        load_embed = True,
        const: float = 10_000.0,
    ):
        super().__init__()
        in_size = max_delay
        self.const = const
        self.embedding_dim = embedding_dim

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
        self.fc_embed1 = nn.Linear(embedding_dim, 2*embedding_dim)
        self.fc_embed2 = nn.Linear(2*embedding_dim, hidden_size)
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.drop1 = nn.Dropout(dropout_probs[0])
        self.drop2 = nn.Dropout(dropout_probs[1])
        self.fc1 = nn.Linear(hidden_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_out = nn.Linear(hidden_units[1], 2)
        self.softplus = nn.Softplus()

        if load_embed:
            if os.path.exists(f"../../weights/embedding_weights_{embedding_dim}"):
                self.embed = nn.Embedding.from_pretrained(torch.load(f"../../weights/embedding_weights_{embedding_dim}").detach())
            elif os.path.exists(f"src/weights/embedding_weights_{embedding_dim}"):
                self.embed = nn.Embedding.from_pretrained(torch.load(f"src/weights/embedding_weights_{embedding_dim}").detach())
            else:
                raise ValueError(f"Saved embeddings under 'src/weights/embedding_weights_{embedding_dim}' not found.")

        else:
            self.embed = nn.Embedding(7, embedding_dim)

    def forward(self, x: torch.Tensor, dow: torch.Tensor | None = None):
        # x: [B, T, D]; mask: [B, T, D]
        x_in = x.float()

        _, (h_n, _) = self.lstm(x_in)          # h_n: [num_layers, B, H]
        feat = h_n[-1]                          # last layer, last time step: [B, H]
        feat = self.act(self.bnorm(feat))

        # Addition of day of week before fully connected block
        if dow is not None:
            if len(dow.size()) == 0:
                dow = torch.unsqueeze(dow, 0)
            embedded = self.embed(dow)
            feat = feat + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
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
        hidden_units: tuple = (256, 128),
        dropout_probs: tuple = (0.15, .1),
        process_in: str = 'sum',
        const: float = 10_000.0,
    ):
        super().__init__()
        in_dim = past_units * (1 if process_in == 'sum' else max_delay)
        self.process_in = process_in
        self.const = const
        assert len(dropout_probs) == len(hidden_units), f"Architecture requires number of hidden layers ({len(hidden_units)}) to be equal to number of dropout probs ({len(dropout_probs)})"
        layers: list[nn.Module] = []
        last = in_dim
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout_probs[i])]
            last = h
        layers += [nn.Linear(last, 2)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def _process_in(self, x: torch.Tensor, dow: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        match self.process_in:
            case 'sum':
                x = x.sum(dim=2)
                assert x.shape == torch.Size((b, t)), f'Final shape after summing is {x.shape} but expected to be batch x t ({b, t})'
                return x.float()
            case 'stack':
                return x.reshape(b, t * d).float()
            case _:
                raise ValueError(f"Init variable '{self.process_in =}' has to be in ['sum', 'stack']")

    def forward(self, x: torch.Tensor, dow: torch.Tensor | None = None):
        x = self._process_in(x)

        params = self.net(x)              # [B, 2]

        lbda = self.const * self.softplus(params[:, 0])
        phi = (self.const ** 2) * self.softplus(params[:, 1]) + 1e-5
        dist = NB(lbda=lbda, phi=phi)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
