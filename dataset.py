import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


def load_node_csv(path):
    df = pd.read_csv(path,names=['id','xl','yl','xh','yh'], header=None)
    x_num = len(df["id"])
    # mapping = {index: i for i, index in range(1000), range(1000)}
    # mapping = {i: i for i in range(1000)}
    # x = None
    # if encoders is not None:
    #     xs = [encoder(df[col]) for col, encoder in encoders.items()]
    #     x = torch.cat(xs, dim=-1)
    x1 = np.random.rand(x_num, 4)
    # 将每行的元素进行归一化，使其和为1，赋初始概率值
    x = x1 / np.sum(x1, axis=1, keepdims=True)
    feature = torch.tensor(x, dtype=torch.float)
    return feature


def load_edge_csv(path):
    df = pd.read_csv(path)

    # src = [src_mapping[index] for index in df[src_index_col]]
    # dst = [dst_mapping[index] for index in df[dst_index_col]]
    src = df["source"]
    dst = df["target"]
    weight = df["weight"]
    edge = torch.tensor([src, dst],dtype=torch.int64)
    # edge_index = torch.tensor(df[['source', 'target']].values, dtype=torch.long).t().contiguous()
    return edge


class LayoutDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        list = os.listdir("MyData/processed")
        return list


    def process(self):
        idx = 0
        EdgeFolderPath = "../../OPENMPL/sub_graph/edge"
        NodeFolderPath = "../../OPENMPL/sub_graph/node"
        for Edge_path in os.listdir(EdgeFolderPath):
            # Read data from `raw_path`.
            edge = load_edge_csv(EdgeFolderPath+'/'+Edge_path)
            x = load_node_csv(NodeFolderPath+'/'+Edge_path)
            # Y 可以是通过ILP算法计算得出的color方案,如果没有Y，则是无监督模型
            data = Data(x=x, edge_index=edge)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == "__main__":
    b = LayoutDataset("MyData")

