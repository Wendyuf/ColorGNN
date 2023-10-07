from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import Model
import time
import loss
import torch
import  dataset
from Visualization.GraphVisiualization import visualize

def test(data):
    modelp = Model.GCN()
    modelp.load_state_dict(torch.load("TrainModel/300.pt"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    modelp.eval()
    out = modelp(data.x, data.edge_index)[0]
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # pred = (out == out.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)# Use the class with highest probability
    visualize(data,pred)

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out, data.edge_index)  # Compute the loss solely based on the training nodes.
    # loss = loss.detach()
    # loss.requires_grad_(True)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


writer = SummaryWriter('logs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Dataset = dataset.LayoutDataset("MyData")
# list=[]
# for i in range(Dataset.len()):
#     list.append(Dataset.get(i))
model = Model.GCN()
criterion = loss.loss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define optimizer.
# loader = DataLoader(list, batch_size=1, shuffle=True)

for epoch in range(10000):
    # for data in loader:
        # loss = test(data)
    for i in range(Dataset.len()):
        # list.append(Dataset.get(i))
        data = Dataset.get(i)
        if data.x.size(0)>1:
            loss = train(data)
    print("Epoch{}_loss:{}".format(epoch, loss.item()))
    writer.add_scalar("loss", loss.item(), epoch, walltime=None)
    if epoch % 1 == 0:
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
    if epoch % 10 == 0:  # 42
        torch.save(model.state_dict(), "TrainModel/{}.pt".format(epoch))




