import pytorch_lightning as pl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KeystrokesGeneratorLearner"]

#===============================================================================

class KeystrokesGeneratorLearner(pl.LightningModule):

    #---------------------------------------------------------------------------
    def __init__(self, generator, p_teacher=0.5, lr=1e-3):
        super().__init__()

        self._p_teacher_forcing = p_teacher
        self._lr = lr

        self.generator = generator
        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()

    #---------------------------------------------------------------------------
    def forward(self, x, y_true=None):

        if self.training:
            use_teacher_forcing = (random.random() < self._p_teacher_forcing)
        else:
            use_teacher_forcing = False

        batch_size = x.shape[0]
        hidden = self.generator.init_hidden(batch_size)

        if use_teacher_forcing:
            x_help = nn.functional.pad(y_true[:, :-1], (0, 0, 1, 0, 0, 0))
            x = torch.cat((x, x_help), axis=2)
            y, _ = self.generator(x, hidden)

        else:
            y = [torch.zeros((x.shape[0], 1, 4))]

            for i in range(x.shape[1]):
                x_i = x[:, i : i + 1]

                if i == 0:
                    hide = torch.zeros(batch_size)
                    x_i = torch.cat((
                            x_i,
                            torch.zeros(batch_size, 1, 4)
                        ), axis=2)

                else:
                    hide = (x[:, i-1:i, 0:1] != 0)
                    x_i = torch.cat((
                            x_i,
                            y[-1],
                            (y[-1][:, :, 0:1] - y[-2][:, :, 1:2]) * hide,
                            (y[-1][:, :, 0:1] + y[-1][:, :, 1:2] - y[-2][:, :, 1:2]) * hide,
                        ), axis=2)

                y_i, hidden = self.generator(x_i, hidden)
                y.append(y_i)

            y = torch.cat(y[1:], axis=1)

        return y

    #---------------------------------------------------------------------------
    def step(self, batch, batch_idx, validation=False):
        x, _ = batch
        x, y = x[:, :, :-4], x[:, :, -4:]

        y_pred = self(x, y)
        loss = self.loss(y_pred, y[:, :, 0:2])
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
        #         factor=0.9,
        #         patience=25,
        #         min_lr=1e-4),
        #     'monitor': 'test_loss'
        # }
        # return [optimizer], [scheduler]

#===============================================================================
