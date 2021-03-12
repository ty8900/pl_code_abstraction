import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLloss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.g_pos = args.g_pos
        self.g_neg = args.g_neg
        self.margin = args.margin
        self.eps = args.eps

    def forward(self, x, y):
        """
        x : input logits (already did sigmoid)
        y : labels (multi-label, binary)
        """
        self.target = y
        self.x_pos = x
        self.x_neg = 1 - x

        self.x_neg = (self.x_neg + self.margin).clamp(max=1)

        self.los_pos = self.target * torch.log(self.x_pos.clamp(min=self.eps))
        self.los_neg = ((1 - self.target) *
                        torch.log(self.x_neg.clamp(min=self.eps)))
        self.loss = self.los_pos + self.los_neg

        self.x_pos = self.x_pos * self.target
        self.x_neg = self.x_neg * (1 - self.target)
        self.focus_loss = torch.pow(1 - self.x_pos - self.x_neg,
                                    self.g_pos * self.target +
                                    self.g_neg * (1 - self.target))
        self.loss *= self.focus_loss

        return -self.loss.sum()


class ELRloss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dataset_size = args.dataset_size
        self.output_size = args.num_output
        self.lamb = args.lamb
        self.beta = args.beta

        self.target = torch.zeros(self.dataset_size, self.output_size).cuda()

    def forward(self, i, x, y):
        # i = B / x = B, C(14) / y = B
        y_pred = F.softmax(x, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[i] = self.beta * self.target[i] + (1 - self.beta) * (
            (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(x, y)
        elr_reg = ((1 - (y_pred_ * self.target[i]).sum(dim=1)).log()).mean()
        elr_loss = ce_loss + self.lamb * elr_reg
        return elr_loss, ce_loss, elr_reg