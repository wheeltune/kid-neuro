import torch.nn as nn

from .norm_layer import NormLayer

__all__ = ["KeystrokesEncoder"]

#===============================================================================

class KeystrokesEncoder(nn.Module):

    #---------------------------------------------------------------------------
    def __init__(self, d_codes, d_hidden, n_layers, p_rnn_dropout=0.2, dropout=0.5):
        super().__init__()

        self.d_codes = d_codes
        self.d_times = 4
        self.d_model = self.d_codes + self.d_times

        self.d_hidden = d_hidden
        self.p_dropout = dropout
        self.p_rnn_dropout = p_rnn_dropout

        self.batch_norm_1 = NormLayer(self.d_model)

        self.rnn_1 = nn.LSTM(
            self.d_model,
            self.d_hidden,
            num_layers=n_layers,
            dropout=self.p_rnn_dropout,
            batch_first=True
        )

        self.batch_norm_2 = NormLayer(self.d_hidden)
        self.dropout = nn.Dropout(self.p_dropout)

        self.rnn_2 = nn.LSTM(
            self.d_hidden,
            self.d_hidden,
            num_layers=n_layers,
            dropout=self.p_rnn_dropout,
            batch_first=True,
        )

    #---------------------------------------------------------------------------
    def forward(self, x):
        times = self.batch_norm_1(x)

        x, _ = self.rnn_1(x)

        x = self.batch_norm_2(x)
        x = self.dropout(x)

        _, (ht, _) = self.rnn_2(x)
        x = ht[-1]

        return x

#===============================================================================