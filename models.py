from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, n_layers: int,
                 lr: float = 1e-4, dropout: float = False, vectors: Tensor = None) -> None:
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)
        if vectors is not None:
            self.emb.weight.data.copy_(vectors)
            self.emb.weight.data.requires_grad = False

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.optim = self.optim = torch.optim.Adam(self.learnable_parameters(), lr)

    def forward(self, x: Tensor, hidden: Tensor = None) -> Tensor:
        x = self.emb(x)
        o, (h, c) = self.rnn(x, hidden)

        (_, batch_size, hidden_size) = c.shape  # [2*n_layers, batch_size, hidden_size]
        c = c.view(-1, 2, batch_size, hidden_size)  # [n_layers, 2, batch_size, hidden_size]
        c = c[-1]  # take only last layer
        c = torch.cat((c[0], c[1]), 1)  # concatenate ontput from both directions to single vector

        return c

    def learnable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, prior_size: int, n_layers: int,
                 lr: float = 1e-4, dropout: float = False, vectors: Tensor = None) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        if vectors is not None:
            self.emb.weight.data.copy_(vectors)
            self.emb.weight.data.requires_grad = False

        self.rnn = nn.LSTM(emb_size + prior_size, hidden_size, n_layers, dropout=dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.optim = self.optim = torch.optim.Adam(self.learnable_parameters(), lr)

    def forward(self, x: Tensor, z: Tensor, hidden: Tensor = None, y: Tensor = None) -> Tuple[Tensor, Tensor]:
        x = self.emb(x)

        (seq_len, batch_size, emb_size) = x.shape
        z = z[None, :, :]
        z = z.repeat(seq_len, 1, 1)
        x = torch.cat((x, z), 2)

        if y is not None:
            y = y[None, :, :]
            y = y.repeat(seq_len, 1, 1)
            x = torch.cat([x, y], dim=2)

        o, h = self.rnn(x, hidden)

        (seq_len, batch_size, hidden_size) = o.shape

        x = o.view(-1, hidden_size)
        x = F.log_softmax(self.fc(x), dim=1)
        return x, h

    def learnable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class Discriminator(nn.Module):
    def __init__(self, sizes, dropout: float = False, lr: float = 1e-4, bn: bool = False,
                 activation_fn: nn.Module = nn.Tanh(), last_fn: nn.Module = None, first_fn: nn.Module = None) -> None:
        super().__init__()

        layers = []

        if first_fn is not None:
            layers.append(first_fn)
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(activation_fn)  # нам не нужен дропаут и фнкция активации в последнем слое
        else:
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if last_fn is not None:
            layers.append(last_fn)
        self.model = nn.Sequential(*layers)

        self.optim = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        if y is not None:
            x = torch.cat([x, y], dim=1)
        return self.model(x)
