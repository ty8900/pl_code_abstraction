import pytorch_lightning as pl
import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch import nn
import matplotlib.pyplot as plt

from .loss import ASLloss
from .base import AbstractModel


class MVCModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = EfficientNet.from_pretrained(args.network_name, advprop=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(args.hidden_size, args.num_output),
            nn.Sigmoid()
        )
        self.loss_func = ASLloss(args)
        if args.loss_type == "bce":
            self.loss_func = nn.BCELoss()
        torch.backends.cudnn.benchmark = False

    @classmethod
    def code(cls):
        return 'mvc'

    def forward(self, x):
        # x [batch, 3, 224, 224]
        self.model.eval()
        with torch.no_grad():
            feature = self.model.extract_features(x)
        # f [batch, 1280, 7, 7]
        feature = torch.squeeze(self.pool(feature))
        # f [batch, 1280]
        output = self.classifier(feature)
        return output

    def training_step(self, batch, batch_idx):
        # batch : dict of input, label
        inputs = batch['image']
        outputs = self(inputs)
        labels = batch['label']
        loss = self.loss_func(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        outputs = self(inputs)
        labels = batch['label']
        loss = self.loss_func(outputs, labels)
        return {'loss': loss, 'preds': outputs, 'labels': labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        p, r, f, p_arr, r_arr, f_arr = self.get_metric(
            preds, labels)
        self.plot(p_arr, 'precision', 'val')
        self.plot(r_arr, 'recall', 'val')
        self.plot(f_arr, 'f1_score', 'val')
        self.log('precision', p, prog_bar=True)
        self.log('recall', r, prog_bar=True)
        self.log('f1_score', f, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        p, r, f, p_arr, r_arr, f_arr = self.get_metric(
            preds, labels)
        self.plot(p_arr, 'precision', 'test')
        self.plot(r_arr, 'recall', 'test')
        self.plot(f_arr, 'f1_score', 'test')
        torch.save(p_arr, 'parr.pt')
        torch.save(r_arr, 'rarr.pt')
        torch.save(f_arr, 'farr.pt')
        self.log('precision', p, prog_bar=True)
        self.log('recall', r, prog_bar=True)
        self.log('f1_score', f, prog_bar=True)

    def get_metric(self, preds, labels):
        prec = pl.metrics.Precision(num_classes=1, multilabel=True)
        rec = pl.metrics.Recall(num_classes=1, multilabel=True)
        f1 = pl.metrics.F1(num_classes=1, multilabel=True)
        prec_f = pl.metrics.Precision(num_classes=1, multilabel=False)
        rec_f = pl.metrics.Recall(num_classes=1, multilabel=False)
        f1_f = pl.metrics.F1(num_classes=1, multilabel=False)

        precision = prec(preds, labels)
        recall = rec(preds, labels)
        f1score = f1(preds, labels)
        parr = torch.stack(
            [prec_f(preds[:, i], labels[:, i]) for i in range(len(preds[0]))])
        rarr = torch.stack(
            [rec_f(preds[:, i], labels[:, i]) for i in range(len(preds[0]))])
        farr = torch.stack(
            [f1_f(preds[:, i], labels[:, i]) for i in range(len(preds[0]))])
        return precision, recall, f1score, parr, rarr, farr

    def plot(self, arr, metric, stage):
        # avg_arr = torch.mean(arr, dim=0)
        n = np.arange(len(arr))
        plt.bar(n, arr)
        plt.savefig("figure/" + str(self.current_epoch) +
                    metric + stage + "_fig.png")
        plt.clf()
