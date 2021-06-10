import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KeystrokesPairwiseLearner"]

#===============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    EPS = 1e-9

    #---------------------------------------------------------------------------
    def __init__(self, margin=1.0, p=2, reduction='mean'):
        super().__init__()
        self._margin = margin
        self._reduction = reduction
        self._p = p

    #---------------------------------------------------------------------------
    def forward(self, x_1, x_2, target, size_average=True):
        distances = self._distance(x_1, x_2)
        losses = 0.5 * (target.float() * distances.pow(2) +
                        (1 + -1 * target).float() * F.relu(self._margin - distances).pow(2))

        if self._reduction == 'mean':
            return losses.mean()
        elif self._reduction == 'sum':
            return losses.sum()

    #---------------------------------------------------------------------------
    def _distance(self, x_1, x_2):
        if self._p == 1:
            return (x_1 - x_2).abs().sum(axis=1)
        elif self._p == 2:
            return ((x_1 - x_2).pow(2).sum(axis=1) + self.EPS).sqrt()
        else:
            raise NotImplemented


#===============================================================================

class KeystrokesPairwiseLearner(pl.LightningModule):

    #---------------------------------------------------------------------------
    def __init__(self, encoder, margin=1.0, p=2, lr=1e-3):
        super().__init__()

        self._lr = lr
        self.encoder = encoder
        self.loss = ContrastiveLoss(margin=margin, p=p)

    #---------------------------------------------------------------------------
    def forward(self,x_1, x_2):
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        return x_1, x_2

    #---------------------------------------------------------------------------
    def step(self, batch, batch_idx, validation=False):
        x_1, x_2, y = batch
        y_pred = self(x_1, x_2)

        loss = self.loss(y_pred[0], y_pred[1], y)
        logs = {
            "loss": loss,
        }

        if not validation:
            logs.update({
                "lr": self.optimizers().param_groups[0]["lr"]
            })

        return loss, logs

    #---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    #---------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, validation=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    #---------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.5,
                patience=25,
                min_lr=1e-8),
            'monitor': 'test_loss'
        }
        return [optimizer], [scheduler]

#===============================================================================
