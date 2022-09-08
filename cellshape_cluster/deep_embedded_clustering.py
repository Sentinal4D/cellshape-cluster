from torch import nn
from clustering_layer import ClusteringLayer


class DeepEmbeddedClustering(nn.Module):
    def __init__(self, autoencoder, num_clusters):
        super(DeepEmbeddedClustering, self).__init__()
        self.autoencoder = autoencoder
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer(
            num_features=self.autoencoder.encoder.num_features,
            num_clusters=self.num_clusters,
        )

    def forward(self, x):
        features = self.autoencoder.encoder(x)
        clusters = self.clustering_layer(features)
        output = self.autoencoder.decoder(features)
        return output, features, clusters
