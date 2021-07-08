import torch.nn as nn
import torch

class SoftCrossEntropy(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None, reduction='mean',k=0.8):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(SoftCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim
        self.reduction = reduction
        self.k = k
    
    def reduce_fun(self,loss):

        if self.reduction == 'mean':
            return torch.mean(loss)

        elif self.reduction == 'topk':
            res, _ = torch.topk(loss, int(loss.size(0) * self.k), sorted=False)
            return torch.mean(res)

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return self.reduce_fun(torch.sum(-true_dist * pred, dim=self.dim))  
