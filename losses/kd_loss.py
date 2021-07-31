import torch.nn as nn
import torch.nn.functional as F



class KD_Loss(nn.Module):
    def __init__(self, alpha=0.95, temperature=6):

        super(KD_Loss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kldivloss = nn.KLDivLoss()

    def forward(self, outputs, labels, teacher_outputs):

        kldivloss = self.kldivloss(F.log_softmax(outputs/self.temperature, dim=1),
                             F.softmax(teacher_outputs/self.temperature, dim=1))
        
        celoss =  F.cross_entropy(outputs, labels)

        total_loss = self.alpha*self.temperature**2 * kldivloss + (1. - self.alpha) * celoss

        return total_loss
