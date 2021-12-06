"""
标签平滑loss
可以把真实标签平滑集成在loss函数里面，然后计算loss
也可以直接在loss函数外面执行标签平滑，然后计算散度loss
"""
import torch.nn as nn
import torch

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑Loss
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        """

        :param classes: 类别数目
        :param smoothing: 平滑系数
        :param dim: loss计算平均值的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # #torch.mean(torch.sum(-true_dist * pred, dim=self.dim))就是按照公式来计算损失
        # loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        #采用KLDivLoss来计算
        loss = self.loss(pred,true_dist)
        return loss