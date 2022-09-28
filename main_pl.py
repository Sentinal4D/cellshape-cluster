import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cellshape_cloud.lightning_autoencoder import CloudAutoEncoderPL
from cellshape_cloud.pointcloud_dataset import (
    PointCloudDataset,
    SingleCellDataset,
    GefGapDataset,
)
from cellshape_cloud.reports import get_experiment_name
from cellshape_cloud.cloud_autoencoder import CloudAutoEncoder
from cellshape_cluster.lightning_deep_embedded_clustering import (
    DeepEmbeddedClusteringPL,
)
import os
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
from torch import Tensor
from matplotlib import pyplot as plt


def make_umap(pl_module, trainer):
    print("Creating UMAP figure.")
    feature_array = pl_module._extract_features()
    scalar = StandardScaler()
    cluster_centres = pl_module.clustering_layer.weight.detach().cpu().numpy()
    features_and_clus = np.concatenate((feature_array, cluster_centres), 0)
    scaled_features = scalar.fit_transform(features_and_clus)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(scaled_features)
    num_clusters = cluster_centres.shape[0]
    b = np.zeros((len(embedding) - num_clusters, 2))

    b[:, 0] = embedding[:-num_clusters, 0]
    b[:, 1] = embedding[:-num_clusters, 1]

    data = pd.DataFrame(b, columns=["Umap1", "Umap2"])

    facet = sns.lmplot(
        data=data,
        x="Umap1",
        y="Umap2",
        fit_reg=False,
        legend=True,
        scatter_kws={"s": 2},
    )
    plt.scatter(
        x=embedding[-num_clusters:, 0],
        y=embedding[-num_clusters:, 1],
        color="r",
        s=15,
    )

    fig = facet.figure
    tensorboard = pl_module.logger.experiment

    callback_step = trainer.callback_metrics.get("step")
    step = (
        callback_step.int()
        if isinstance(callback_step, Tensor)
        else torch.tensor(trainer.global_step)
    )
    tensorboard.add_figure("UMAP", fig, step)
    print("Done creating UMAP figure.")


class UmapCallback(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        make_umap(pl_module, trainer)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        make_umap(pl_module, trainer)

    def on_epoch_end(self, trainer, pl_module):
        make_umap(pl_module, trainer)


def train_dec_pl(args):
    if args.dataset_type == "SingleCell":
        dataset = SingleCellDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            num_points=args.num_points,
        )

    elif args.dataset_type == "GefGap":
        dataset = GefGapDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            norm_std=args.norm_std,
        )

    else:
        dataset = PointCloudDataset(args.cloud_dataset_path)

    if args.is_pretrained_lightning:
        model = CloudAutoEncoder(
            num_features=args.num_features,
            k=args.k,
            encoder_type=args.encoder_type,
            decoder_type=args.decoder_type,
        )
        autoencoder = CloudAutoEncoderPL(args=args, model=model)
        model = DeepEmbeddedClusteringPL(
            autoencoder=autoencoder,
            num_clusters=args.num_clusters,
            args=args,
            dataset=dataset,
        )
        try:
            model.load_lightning_autoencoder(args.pretrained_path)

        except Exception as e:
            print(f"Cannot load model due to error {e}.")

    else:
        autoencoder = CloudAutoEncoder(
            num_features=args.num_features,
            k=args.k,
            encoder_type=args.encoder_type,
            decoder_type=args.decoder_type,
        )
        model = DeepEmbeddedClusteringPL(
            autoencoder=autoencoder,
            num_clusters=args.num_clusters,
            args=args,
            dataset=dataset,
        )
        try:
            model.load_model_autoencoder(args.pretrained_path)
        except Exception as e:
            print(f"Can't load pretrained network due to error {e}.")
    new_output = args.output_dir + f"/{args.dataset_type}/"
    os.makedirs(new_output, exist_ok=True)

    logging_info = get_experiment_name(
        model=autoencoder.model, output_dir=new_output
    )
    os.makedirs(
        new_output + logging_info[3] + "/lightning_logs/", exist_ok=True
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="loss", every_n_epochs=10
    )
    umap_callback = UmapCallback()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.num_epochs_clustering,
        default_root_dir=new_output + logging_info[3],
        callbacks=[checkpoint_callback, umap_callback],
        # strategy="ddp_find_unused_parameters_false",
    )

    trainer.fit(model)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape-cluster")
    parser.add_argument(
        "--model_type",
        default="cloud",
        type=str,
        choices=["cloud", "voxel"],
        help="Please provide the type of model: [cloud, voxel]",
    )
    parser.add_argument(
        "--cloud_convert",
        default="False",
        type=str2bool,
        help="Do you need to convert 3D images to point clouds?",
    )
    parser.add_argument(
        "--num_points",
        default=2048,
        type=int,
        help="The number of points used in each point cloud.",
    )
    parser.add_argument(
        "--train_type",
        default="DEC",
        type=str,
        choices=["pretrain", "DEC"],
        help="Please provide the type of training mode: [pretrain, full]",
    )
    parser.add_argument(
        "--pretrain",
        default="False",
        type=str2bool,
        help="Please provide whether or not to pretrain the autoencoder",
    )
    parser.add_argument(
        "--tif_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestTiff/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestMesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="/home/mvries/Documents/Datasets/OPM/" "VickyCellshape/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataset_type",
        default="GefGap",
        type=str,
        choices=["SingleCell", "GefGap", "Other"],
        help="Please provide the type of dataset. "
        "If using the one from our paper, then choose 'SingleCell', "
        "otherwise, choose 'Other'.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="/home/mvries/Documents/Datasets/OPM/VickyCellshape/"
        "cn_allFeatures_withGeneNames_updated.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/Testing_output/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=1,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_epochs_clustering",
        default=250,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of features to extract.",
    )
    parser.add_argument(
        "--num_clusters",
        default=5,
        type=int,
        help="Please provide the number of clusters to find.",
    )
    parser.add_argument(
        "--k", default=20, type=int, help="Please provide the value for k."
    )
    parser.add_argument(
        "--encoder_type",
        default="dgcnn",
        type=str,
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnetbasic",
        type=str,
        help="Please provide the type of decoder.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.0001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--learning_rate_clustering",
        default=0.00001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--update_interval",
        default=10,
        type=int,
        help="How often to update the target "
        "distribution for the kl divergence.",
    )
    parser.add_argument(
        "--gamma",
        default=100,
        type=int,
        help="Please provide the value for gamma.",
    )
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="Please provide the value for alpha.",
    )
    parser.add_argument(
        "--divergence_tolerance",
        default=0.01,
        type=float,
        help="Please provide the divergence tolerance.",
    )
    parser.add_argument(
        "--proximal",
        default=2,
        type=int,
        help="Do you want to look at cells distal "
        "or proximal to the coverslip?"
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs"
        "/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI"
        "/DYNCESYS/mvries/ResultsAlma/cellshape-cloud/"
        "epoch=53-step=76356.ckpt",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )
    parser.add_argument(
        "--is_pretrained_lightning",
        default=True,
        type=str2bool,
        help="Is the pretrained model a lightning module?",
    )

    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="The number of gpus to use for training.",
    )
    parser.add_argument(
        "--norm_std",
        default=True,
        type=str2bool,
        help="Standardize by a factor of 20?",
    )
    parser.add_argument(
        "--cell_component",
        default="cell",
        type=str,
        help="Cell or nucleus?",
    )

    arguments = parser.parse_args()
    train_dec_pl(arguments)
