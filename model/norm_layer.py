import torch.nn as nn

__all__ = ["NormLayer"]

#===============================================================================

class NormLayer(nn.Module):

    #---------------------------------------------------------------------------
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.batchnorm = nn.BatchNorm1d(self.d_model)

    #---------------------------------------------------------------------------
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.transpose(1, 2)
        return x

#===============================================================================
