import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch import nn
import matplotlib.pyplot as plt

from .loss import ELRloss
from .base import AbstractModel


class Clothing1MModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.model = self.model_selection()
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, args.num_output)
        """ct = 0
        for child in model_ft.children():
            ct += 1
            if ct < 7:
            for param in child.parameters():
                    param.requires_grad = False"""

        self.loss_func = nn.CrossEntropyLoss()
        if args.loss_type == "elr":
            self.loss_func = ELRloss(args)
        self.val_loss = nn.CrossEntropyLoss()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def model_selection(self):
        if self.args.network_name == 'resnet50':
            return models.resnet50(pretrained=True)
        elif self.args.network_name == 'resnet34':
            return models.resnet34(pretrained=True)

    @classmethod
    def code(cls):
        return 'clothing1m'

    def forward(self, x):
        # x [batch, 3, 224, 224]
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # batch : dict of input, label
        inputs = batch['image']
        outputs = self(inputs)
        labels = batch['label'].squeeze()
        indexes = batch['index'].squeeze()
        if self.args.loss_type == "elr":
            loss, ce, reg = self.loss_func(indexes, outputs, labels)
            self.log('train_ce', ce)
            self.log('train_reg', reg)
        else:
            loss = self.loss_func(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.train_acc(outputs, labels))
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        outputs = self(inputs)
        labels = batch['label'].squeeze()
        loss = self.val_loss(outputs, labels)
        return {'loss': loss, 'preds': outputs, 'labels': labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(preds,labels), prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs = batch['image']
        outputs = self(inputs)
        labels = batch['label'].squeeze()
        loss = self.val_loss(outputs, labels)
        return {'loss': loss, 'preds': outputs, 'labels': labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(preds, labels), prog_bar=True)

