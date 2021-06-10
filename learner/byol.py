import copy
import random
from   functools import wraps

import pytorch_lightning as pl
import torch
from   torch import nn
import torch.nn.functional as F

__all__ = ["KeystrokesBYOLLearner"]

#===============================================================================

def _get_model_device(model):
    return next(model.parameters()).device

#===============================================================================

def _set_requires_grad(model, is_requires):
    for p in model.parameters():
        p.requires_grad = is_requires

#===============================================================================

class BYOLLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    EPS = 1e-9

    #---------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    #---------------------------------------------------------------------------
    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

#===============================================================================

class EMA:
    """Exponential moving average"""

    #---------------------------------------------------------------------------
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    #---------------------------------------------------------------------------
    def __call__(self, old, new):
        if old is None:
            return new

        return old * self.beta + (1 - self.beta) * new

#===============================================================================

class MLP(nn.Module):

    #---------------------------------------------------------------------------
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_output)
        )

    #---------------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

#===============================================================================

class KeystrokesBYOLLearner(pl.LightningModule):

    #---------------------------------------------------------------------------
    def __init__(self, model, d_view,
                 d_projection = None, d_hidden = 4096,
                 momentum = 0.95, lr=1e-3):

        super().__init__()

        self._lr = lr
        self._d_view = d_view
        self._d_hidden = d_hidden
        self._d_projection = d_projection if d_projection is not None else self._d_view
        self._ema = EMA(momentum)

        self.loss = BYOLLoss()

        self.online_model = model
        self.online_projector = MLP(self._d_view, self._d_projection, d_hidden)
        self.online_predictor = MLP(self._d_projection, self._d_projection, d_hidden)

        self.target_model = copy.deepcopy(self.online_model)
        self.target_projector = copy.deepcopy(self.online_projector)

        # # get device of network and make wrapper same device
        # device = _get_model_device(model)
        # self.to(device)

    #---------------------------------------------------------------------------
    def _update_model_moving_average(self, target, online):
        for online_param, target_param in zip(target.parameters(), online.parameters()):
            online_weight = online_param.data
            target_weight = target_param.data
            target_weight.data = self._ema(target_weight, online_weight)

    #---------------------------------------------------------------------------
    def _update_moving_average(self):
        self._update_model_moving_average(self.target_model, self.online_model)
        self._update_model_moving_average(self.target_projector, self.online_projector)

    #---------------------------------------------------------------------------
    def forward(self, x_1, x_2):
        online_y_1 = self.online_predictor(self.online_projector(self.online_model(x_1)))
        online_y_2 = self.online_predictor(self.online_projector(self.online_model(x_2)))

        with torch.no_grad():
            target_y_1 = self.target_projector(self.target_model(x_1))
            target_y_2 = self.target_projector(self.target_model(x_2))

            target_y_1.detach_()
            target_y_2.detach_()

        return target_y_1.detach(), target_y_2.detach(), online_y_1, online_y_2

    #---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_1, x_2, y = batch
        t_1, t_2, o_1, o_2 = self.forward(x_1, x_2)
        loss = (self.loss(t_1, o_2) + self.loss(t_2, o_1)).mean()
        return {'loss': loss}

    #---------------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    #---------------------------------------------------------------------------
    def on_before_zero_grad(self, _):
        self._update_moving_average()

#===============================================================================
