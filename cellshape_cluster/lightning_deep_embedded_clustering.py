import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .clustering_layer import ClusteringLayer1
import numpy as np
from sklearn.cluster import KMeans
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss
import torch.nn.functional as F


class DeepEmbeddedClusteringPL(pl.LightningModule):
    def __init__(self, autoencoder, num_clusters, dataset, args):
        super(DeepEmbeddedClusteringPL, self).__init__()
        self.target_distribution = None
        self.save_hyperparameters(
            ignore=[
                "reconstruction_criterion",
                "autoencoder",
                "cluster_criterion",
                "dataset",
            ]
        )
        self.autoencoder = autoencoder
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer1(
            num_features=self.autoencoder.model.num_features,
            num_clusters=self.num_clusters,
        )
        self.dataset = dataset
        self.args = args
        self.reconstruction_criterion = ChamferLoss()
        self.cluster_criterion = torch.nn.KLDivLoss(reduction="batchmean")
        self.lr = args.learning_rate_clustering
        self.automatic_optimization = False

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
        print("loading c")
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        self.load_state_dict(checkpoint["state_dict"])

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=24,
        )
        # it is very important that shuffle=False here!

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9
            # betas=(0.9, 0.999),
            # weight_decay=1e-4,
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
        print(" \t Initialising cluster centroids...")
        km = KMeans(n_clusters=self.num_clusters, n_init=20)
        feature_array = self._extract_features()
        km.fit_predict(feature_array)
        weights = torch.from_numpy(km.cluster_centers_)
        self.clustering_layer.set_weight(weights.to(self.device))
        self.autoencoder.model.train()
        print("Cluster centres initialised")

    def _get_target_distribution(self, out_distribution, q_power):
        numerator = (out_distribution**q_power) / torch.sum(
            out_distribution, axis=0
        )
        p = (numerator.t() / torch.sum(numerator, axis=1)).t()
        return p

    def _get_distributions(self, dataloader):
        print("Getting target distribution.")
        cluster_distribution = None
        self.eval()
        for data in dataloader:
            inputs = data[0]
            inputs = inputs.to(self.device)
            z = self.encode(inputs)
            clusters = self.clustering_layer(z)
            if cluster_distribution is not None:
                cluster_distribution = np.concatenate(
                    (
                        cluster_distribution,
                        clusters.clone().cpu().detach().numpy(),
                    ),
                    0,
                )
            else:
                cluster_distribution = clusters.clone().cpu().detach().numpy()

        predictions = np.argmax(cluster_distribution.data, axis=1)
        self.train()
        print("Finished getting target distribution.")
        return torch.from_numpy(cluster_distribution), predictions

    def _extract_features(self):
        print("Extracting features.")
        dataloader = self.train_dataloader()
        feature_array = None
        self.eval()
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
        self.train()
        print("Done extracting features.")

        return feature_array

    def on_train_start(self) -> None:
        self._initialise_centroid()

    def training_step(self, batch, batch_idx):
        self.autoencoder.model.encoder.train()
        for param in self.autoencoder.model.encoder.parameters():
            param.requires_grad = True
        self.autoencoder.model.decoder.train()
        self.clustering_layer.train()
        for param in self.clustering_layer.parameters():
            param.requires_grad = True
        opt = self.optimizers()
        # print(list(self.parameters())[0].grad)
        # print(list(self.parameters())[-1].grad)
        # print(list(self.parameters())[-1])
        opt.zero_grad()
        batch_num = batch_idx + 1
        if (
            (self.current_epoch == 0)
            or (self.current_epoch % self.args.update_interval == 0)
        ) and (batch_num == 1):
            (
                cluster_distribution,
                previous_cluster_predictions,
            ) = self._get_distributions(self.train_dataloader())
            self.target_distribution = self._get_target_distribution(
                cluster_distribution, self.args.q_power
            )

        inputs = batch[0]
        batch_size = inputs.shape[0]
        tar_dist = self.target_distribution[
            ((batch_num - 1) * batch_size) : (batch_num * batch_size),
            :,
        ].to(self.device)

        features = self.autoencoder.model.encoder(inputs)
        clusters = self.clustering_layer(features)
        # print(clusters.grad)
        # print(features.grad)
        outputs = self.autoencoder.model.decoder(features)
        # print(outputs.grad)
        reconstruction_loss = self.reconstruction_criterion(inputs, outputs)
        cluster_loss = self.cluster_criterion(
            F.log_softmax(clusters), F.softmax(tar_dist)
        )
        # loss = reconstruction_loss + (self.args.gamma * cluster_loss)
        loss = reconstruction_loss + 100 * cluster_loss
        a = list(self.parameters())[-1].clone()
        c = list(self.parameters())[0].clone()

        self.manual_backward(loss, retain_graph=True)
        opt.step()

        b = list(self.parameters())[-1].clone()
        d = list(self.parameters())[0].clone()

        print(torch.equal(a.data, b.data))
        print(torch.equal(c.data, d.data))

        # print(list(self.parameters())[-1][0]*1000000)
        print(list(self.clustering_layer.parameters())[0][0][0])
        print(list(self.autoencoder.model.encoder.parameters())[0][0][0])

        self.log("loss", loss, prog_bar=True)
        # self.log("recon_loss", reconstruction_loss)
        # self.log("cluster_loss", cluster_loss)
        outputs = {
            "loss": loss,
            "recon_loss": reconstruction_loss,
            "cluster_loss": cluster_loss,
        }
        return outputs
