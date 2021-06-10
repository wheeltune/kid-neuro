import pytorch_lightning as pl
from   sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

__all__ = ["KeystrokesSoftmaxLearner"]

#===============================================================================

def exist_accuracy(y_true, y_pred):
    return ((y_true.unsqueeze(1) == y_pred).any(dim=1).sum().item() / len(y_true))

#===============================================================================

class MLP(nn.Module):

    #---------------------------------------------------------------------------
    def __init__(self, d_input, d_output, d_hidden, p_dropout=0.5):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(p_dropout),
            nn.Tanh(),
            nn.Linear(d_hidden, d_output)
        )

    #---------------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

#===============================================================================

class KeystrokesSoftmaxLearner(pl.LightningModule):

    #---------------------------------------------------------------------------
    def __init__(self, encoder, d_view, d_classes, projector=None, lr=1e-3):
        super().__init__()

        self._lr = lr
        self._encoder = encoder

        self._projector = projector
        self._predictor = nn.Linear(d_view, d_classes)

        self._loss = nn.CrossEntropyLoss()

    #---------------------------------------------------------------------------
    def forward(self, x):
        x = self._encoder(x)

        if self._projector is not None:
            x = self._projector(x)

        x = self._predictor(x)
        return x

    #---------------------------------------------------------------------------
    def step(self, batch, batch_idx, validation=False):
        x, y = batch

        y_pred = self(x)
        y_pred_classes = y_pred.argmax(dim=1)

        loss = self._loss(y_pred, y)
        logs = {
            "loss": loss,
            "acc": accuracy_score(y.cpu(), y_pred_classes.cpu())
        }

        if validation:
            y_pred_sort = y_pred.detach().argsort()

            acc_5 = exist_accuracy(y, y_pred_sort[:, -5:])
            acc_50 = exist_accuracy(y, y_pred_sort[:, -50:])
            acc_100 = exist_accuracy(y, y_pred_sort[:, -100:])

            logs.update({
                "acc_5": acc_5,
                "acc_50": acc_50,
                "acc_100": acc_100,
            })
        else:
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
                factor=0.9,
                patience=25,
                min_lr=1e-8),
            'monitor': 'test_loss'
        }
        return [optimizer], [scheduler]

#===============================================================================