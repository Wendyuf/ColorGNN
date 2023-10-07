import torch
from torch.nn import Linear
from torch.nn import Softmax
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 16)
        self.conv6 = GCNConv(16, 8)
        self.conv7 = GCNConv(8, 4)
        self.conv8 = GCNConv(4, 4)
        self.conv9 = GCNConv(4, 4)
        self.conv10 = GCNConv(4, 4)
        self.classifier = Softmax(1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()
        h = self.conv4(h, edge_index)
        h = h.relu()
        h = self.conv5(h, edge_index)
        h = h.relu()
        h = self.conv6(h, edge_index)
        h = h.relu()
        h = self.conv7(h, edge_index)
        h = h.relu()
        h = self.conv8(h, edge_index)
        h = h.relu()
        h = self.conv9(h, edge_index)
        h = h.relu()
        h = self.conv10(h, edge_index)
        h = h.relu()


        # Apply a final (linear) classifier.
        out = self.classifier(h)
        # mask = (out == out.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        # out = torch.mul(mask, out)
        # out = (out == out.max(dim=1, keepdim=True)[0]).float()

        return out, h

if __name__=="__main__":
    model = GCN()
    print(model)