import torch
import torch.nn as nn
import torch.nn.functional as F

class loss(nn.Module):
    def __init__(self, margin=1): # 当label = 1 时， 距离<margin 则产生loss
        super().__init__()
        self.margin = margin
    def forward(self, output, edgeindex):
        sum = 0
        for edge in edgeindex.H:
            euclidean_distance = F.pairwise_distance(output[edge[0]], output[edge[1]], keepdim=True)
            sum = sum + max(self.margin-euclidean_distance,torch.tensor(0.0,requires_grad=True))
        pred = output.argmax(dim=1)
        count  = torch.bincount(pred).tolist()
        print(count)
        sum = sum -torch.log(0.1*torch.prod(count))
        # loss = sum.clone().detach().requires_grad_(True)
        # loss = loss.detach()
        # loss.requires_grad_(True)
        # print(loss)
        # print(loss.grad)
        # print(loss.is_leaf)
        return sum