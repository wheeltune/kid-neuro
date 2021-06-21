import pytorch_lightning as pl
import torch
import torch.nn as nn

__all__ = ["KeystrokesTripletLearner"]

#===============================================================================

class KeystrokesTripletLearner(pl.LightningModule):

    #---------------------------------------------------------------------------
    def __init__(self, encoder, margin=1.0, p=2, lr=1e-3):
        super().__init__()

        self._margin = margin
        self._p = p
        self._lr = lr

        self.encoder = encoder
        self.loss = nn.TripletMarginLoss(margin=self._margin, p=self._p)

    #---------------------------------------------------------------------------
    def forward(self, x_1, x_2, x_3):
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x_3 = self.encoder(x_3)
        return x_1, x_2, x_3

    #---------------------------------------------------------------------------
    def step(self, batch, batch_idx, validation=False):
        x_1, x_2, x_3 = batch
        y_1, y_2, y_3 = self(x_1, x_2, x_3)

        loss = self.loss(y_1, y_2, y_3)
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
        return [optimizer]
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         'min',
        #         factor=0.5,
        #         patience=25,
        #         min_lr=1e-4),
        #     'monitor': 'test_loss'
        # }
        # return [optimizer], [scheduler]

#===============================================================================