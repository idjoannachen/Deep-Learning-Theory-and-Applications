# GCNs.py - First created by Kincaid MacDonald in Spring 2021.
# Deep Learning Theory and Applications - Assignment 4.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool, global_mean_pool

# This is the standard GCN Convolutional layer from Kipf & Welling. You will use it in your experiments,
# but the assignment asks you to adapt this into a more powerful GNN, using the skeleton class below
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# TODO: Adapt the GCNConv class with the modifications suggested by the pset. Refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html for details on the MessagePassing class
class BetterGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BetterGCNConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, in_channels),
                       ReLU(),
                       Linear(in_channels, out_channels))
        

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.mlp(x)

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        # print(tmp.shape())
        return self.mlp(tmp)




# These are the network classes: they combine the message-passing layers defined above just as a CNN combines convolutional layers.
class NodeClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_features, num_classes):
        super(NodeClassifier, self).__init__()
        self.conv1 = BetterGCNConv(num_node_features, hidden_features)
        self.conv2 = BetterGCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GraphClassifier, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = BetterGCNConv(num_node_features, hidden_channels)
        self.conv2 = BetterGCNConv(hidden_channels, hidden_channels)
        self.conv3 = BetterGCNConv(hidden_channels, hidden_channels)
        self.conv4 = BetterGCNConv(hidden_channels, hidden_channels)
        self.conv5 = BetterGCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.bn = nn.BatchNorm1d(hidden_channels, affine=False)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv5(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0, training=self.training)
        x = self.lin(x)

        return x

class GraphClassifier2(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GraphClassifier2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.bn = nn.BatchNorm1d(hidden_channels, affine=False)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv5(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0, training=self.training)
        x = self.lin(x)

        return x