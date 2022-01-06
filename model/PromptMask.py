
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_transformers import RobertaForMaskedLM
from pytorch_transformers.modeling_bert import BertLayerNorm
from config.cfg import cfg, path, hyper_roberta
import time
from torch.autograd import Variable
import pytorch_lightning as pl
import os


class PromptMask(pl.LightningModule):
    def __init__(self, answer_map):
        super(PromptMask, self).__init__()
        self.answer_map = answer_map

        self.roberta = RobertaForMaskedLM.from_pretrained(path['roberta_path'])


    def forward(self, input_x):
        mask0 = (input_x == 50264)
        mask1 = (input_x != 1).type(torch.long)

        input_x = self.roberta(input_x, attention_mask=mask1)
        x = input_x[0]
        x = x[mask0]

        return x

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x, batch_y = torch.tensor(batch_x).long(), torch.tensor(batch_y).long()

        output = self(batch_x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch_y)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log("my_loss", outputs.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x, batch_y = torch.tensor(batch_x).long(), torch.tensor(batch_y).long()

        output = self(batch_x)

        _, pred = torch.max(output, dim=1)

        pred = pred.cpu().detach().numpy()
        batch_y = batch_y.cpu().detach().numpy()

        # for j in range(pred.shape[0]):
        #     label_out.append(pred[j])
        #     label_y.append(batch_y[j])
        pos_cc, neg_cc = 0, 0
        for j in range(cfg['K']):
            if pred[j] == self.answer_map[1]:
                pos_cc += 1
        for j in range(cfg['K'], pred.shape[0]):
            if pred[j] == self.answer_map[1]:
                neg_cc += 1

        label_out, label_y = -1, -1
        if pos_cc >= neg_cc:
            label_out = 1
        else:
            label_out = 0

        if batch_y[0] == self.answer_map[1]:
            label_y = 1
        else:
            label_y = 0

        return label_out, label_y

    def test_epoch_end(self, outputs):
        out, y = [], []
        print(outputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=cfg['learning_rate'])
        return optimizer







