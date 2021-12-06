import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    """
    CE(pt)= -log[pt]
    FL(pt)=-(1-pt)**α*log(pt)


    loss_fn = FocalLoss()
    loss = loss_fn(input,label)
    loss.backward()
    """

    def __init__(self,weight=None, reduction="mean",gamma=1,eps=1e-7):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)


    def forward(self,input, target):
        logpt = -self.ce(input,target) # ce =-logpt
        pt = torch.exp(logpt)
        loss = -(1-pt)**self.gamma*logpt
        return loss.mean()
