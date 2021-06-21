import torch
import torch.nn as nn

from .norm_layer import NormLayer

__all__ = ["KeystrokesGenerator"]

#===============================================================================

class KeystrokesGenerator(nn.Module):

    #---------------------------------------------------------------------------
    def __init__(self, d_codes, d_hidden, n_layers, p_rnn_dropout=0.2, dropout=0.5):
        super().__init__()

        self.d_codes = d_codes
        self.d_times = 4
        self.d_model = self.d_codes + self.d_times
        self.n_layers = n_layers

        self.d_hidden = d_hidden
        self.p_dropout = dropout
        self.p_rnn_dropout = p_rnn_dropout

        self.batch_norm_0 = NormLayer(self.d_model)

        self.rnn_1 = nn.LSTM(
            self.d_model,
            self.d_hidden,
            num_layers=n_layers,
            dropout=self.p_rnn_dropout,
            batch_first=True
        )
        # self.batch_norm_1 = NormLayer(self.d_hidden)
        # self.dropout_1 = nn.Dropout()

        # self.rnn_2 = nn.LSTM(
        #     self.d_hidden,
        #     self.d_hidden,
        #     num_layers=n_layers,
        #     dropout=self.p_rnn_dropout,
        #     batch_first=True
        # )

        self.linear_3 = nn.Linear(self.d_hidden, 2)
        # self.linear_3 = nn.Linear(self.d_hidden, self.d_hidden * 2)
        # self.dropout_3 = nn.Dropout(self.p_dropout)
        # self.activation_3 = nn.LeakyReLU(inplace=True)

        # self.linear_4 = nn.Linear(self.d_hidden * 2, 2)

    #---------------------------------------------------------------------------
    def forward(self, x, hidden):
        x = self.batch_norm_0(x)

        x, hidden_1 = self.rnn_1(x, hidden)
        # x, hidden_1 = self.rnn_1(x, hidden[0])
        # x = self.batch_norm_1(x)
        # x = self.dropout_1(x)

        # x, hidden_2 = self.rnn_2(x, hidden[1])

        x = self.linear_3(x)
        # x = self.dropout_3(x)
        # x = self.activation_3(x)

        # x = self.linear_4(x)

        # return x, (hidden_1, hidden_2)
        return x, hidden_1

    #---------------------------------------------------------------------------
    # def init_hidden(self, batch_size):
    #     return ((torch.zeros(self.n_layers, batch_size, self.d_hidden),
    #              torch.zeros(self.n_layers, batch_size, self.d_hidden)),
    #             (torch.zeros(self.n_layers, batch_size, self.d_hidden),
    #              torch.zeros(self.n_layers, batch_size, self.d_hidden)))

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.d_hidden),
                torch.zeros(self.n_layers, batch_size, self.d_hidden))

#===============================================================================
