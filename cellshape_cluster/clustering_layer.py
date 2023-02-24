import torch
from torch import nn


class ClusteringLayer(nn.Module):
    def __init__(self, num_features=10, num_clusters=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.Tensor(self.num_clusters, self.num_features)
        )
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        z = z.cuda()
        diff = torch.sum((z.unsqueeze(1) - self.weight) ** 2, 2)
        numerator = 1.0 / (1.0 + (diff / self.alpha))
        power = (self.alpha + 1.0) / 2
        numerator = numerator**power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        # x = x.unsqueeze(1) - self.weight
        # x = torch.mul(x, x)
        # x = torch.sum(x, dim=2)
        # x = 1.0 + (x / self.alpha)
        # x = 1.0 / x
        # x = x ** ((self.alpha + 1.0) / 2.0)
        # x = torch.t(x) / torch.sum(x, dim=1)
        # x = torch.t(x)
        return q

    def extra_repr(self):
        return "in_features={}, out_features={}, alpha={}".format(
            self.num_features, self.num_clusters, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor.clone(), requires_grad=True)


class ClusteringLayer1(nn.Module):
    def __init__(self, num_features=10, num_clusters=10, alpha=1.0):
        super(ClusteringLayer1, self).__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.Tensor(self.num_clusters, self.num_features)
        )
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        z = z.cuda()
        diff = torch.sum((z.unsqueeze(1) - self.weight) ** 2, 2)
        numerator = 1.0 / (1.0 + (diff / self.alpha))
        power = (self.alpha + 1.0) / 2
        numerator = numerator**power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        # x = x.unsqueeze(1) - self.weight
        # x = torch.mul(x, x)
        # x = torch.sum(x, dim=2)
        # x = 1.0 + (x / self.alpha)
        # x = 1.0 / x
        # x = x ** ((self.alpha + 1.0) / 2.0)
        # x = torch.t(x) / torch.sum(x, dim=1)
        # x = torch.t(x)
        return q

    def extra_repr(self):
        return "in_features={}, out_features={}, alpha={}".format(
            self.num_features, self.num_clusters, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor.clone(), requires_grad=True)
