import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .clustering_layer import ClusteringLayer
import numpy as np
from sklearn.cluster import KMeans
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss
from tqdm import tqdm


class DeepEmbeddedClusteringPL(pl.LightningModule):
    def __init__(self, autoencoder, num_clusters, dataset, args):
        super(DeepEmbeddedClusteringPL, self).__init__()
        self.autoencoder = autoencoder
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer(
            num_features=self.autoencoder.model.num_features,
            num_clusters=self.num_clusters,
        )
        self.dataset = dataset
        self.args = args
        self.reconstruction_criterion = ChamferLoss()
        self.cluster_criterion = torch.nn.KLDivLoss(reduction="sum")
        self.lr = args.learning_rate_clustering

    def load_model_autoencoder(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        model_dict = (
            self.model.state_dict()
        )  # load parameters from pre-trained FoldingNet

        for k in checkpoint["model_state_dict"]:
            if k in model_dict:
                model_dict[k] = checkpoint["model_state_dict"][k]
                print("    Found weight: " + k)
            elif k.replace("folding1", "folding") in model_dict:
                model_dict[k.replace("folding1", "folding")] = checkpoint[
                    "model_state_dict"
                ][k]
                print("    Found weight: " + k)
        print("Done loading encoder")

        self.model.load_state_dict(model_dict)

    def load_lightning_autoencoder(self, path):
        print("Loading lightning autoencoder.")
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        self.autoencoder.load_state_dict(checkpoint["state_dict"])
        print("Done.")

    def load_lightning_dec(self, path):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        self.load_state_dict(checkpoint["state_dict"])

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=False
        )
        # it is very important that shuffle=False here!

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        return optimizer

    def encode(self, x):
        z = self.autoencoder.model.encoder(x)
        return z

    def cluster(self, z):
        q = self.clustering_layer(z)
        return q

    def decode(self, z):
        out = self.autoencoder.model.decoder(z)
        return out

    def _initialise_centroid(self):
        print("Initialising cluster centroids...")
        km = KMeans(n_clusters=self.num_clusters, n_init=20)
        dataloader = self.val_dataloader()
        feature_array = None
        self.autoencoder.model.encoder.eval()
        for batch in dataloader:
            data = batch[0]
            data = data.to(self.device)
            features = self.encode(data)
            if feature_array is not None:
                feature_array = np.concatenate(
                    (feature_array, features.cpu().detach().numpy()), 0
                )
            else:
                feature_array = features.cpu().detach().numpy()

        km.fit_predict(feature_array)
        weights = torch.from_numpy(km.cluster_centers_, requires_grad=True)
        self.clustering_layer.set_weight(weights.to(self.device))
        self.autoencoder.model.encoder.train()

    def _get_target_distribution(self, out_distribution):
        numerator = (out_distribution**2) / torch.sum(
            out_distribution, axis=0
        )
        p = (numerator.t() / torch.sum(numerator, axis=1)).t()
        return p

    def _get_distributions(self, dataloader):

        cluster_distribution = None
        self.autoencoder.model.encoder.eval()
        for data in tqdm(dataloader):
            inputs = data[0]
            inputs = inputs.to(self.device)
            z = self.encode(inputs)
            clusters = self.clustering_layer(z)
            if cluster_distribution is not None:
                cluster_distribution = np.concatenate(
                    (cluster_distribution, clusters.cpu().detach().numpy()), 0
                )
            else:
                cluster_distribution = clusters.cpu().detach().numpy()

        predictions = np.argmax(cluster_distribution.data, axis=1)
        self.autoencoder.model.encoder.train()
        return torch.from_numpy(cluster_distribution), predictions

    def training_step(self, batch, batch_idx):
        batch_num = batch_idx + 1
        if (self.current_epoch == 0) or (
            self.current_epoch % self.args.update_interval == 0
        ):
            print("Initialising centroids.")
            # self._initialise_centroid()
            print("Initialising target distribution.")
            (
                cluster_distribution,
                previous_cluster_predictions,
            ) = self._get_distributions(self.val_dataloader())
            target_distribution = self._get_target_distribution(
                cluster_distribution
            )

        inputs = batch[0]
        batch_size = inputs.shape[0]
        tar_dist = target_distribution[
            ((batch_num - 1) * batch_size) : (batch_num * batch_size),
            :,
        ].to(self.device)

        features = self.encode(inputs)
        clusters = self.cluster(features)
        outputs = self.decode(features)
        reconstruction_loss = self.reconstruction_criterion(inputs, outputs)
        cluster_loss = self.cluster_criterion(torch.log(clusters), tar_dist)
        loss = reconstruction_loss + (self.args.gamma * cluster_loss)

        self.log_dict(
            {
                "loss": loss,
                "recon_loss": reconstruction_loss,
                "cluster_loss": cluster_loss,
            }
        )

        return loss
