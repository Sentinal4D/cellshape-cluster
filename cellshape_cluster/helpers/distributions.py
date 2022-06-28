import torch
import numpy as np


def get_target_distribution(out_distr):
    tar_dist = out_distr**2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


def get_distributions(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cluster_distribution = None
    model.eval()
    for data in dataloader:
        inputs = data[0]
        inputs = inputs.to(device)
        outputs, features, clusters = model(inputs)
        if cluster_distribution is not None:
            cluster_distribution = np.concatenate(
                (cluster_distribution, clusters.cpu().detach().numpy()), 0
            )
        else:
            cluster_distribution = clusters.cpu().detach().numpy()

    predictions = np.argmax(cluster_distribution.data, axis=1)
    return cluster_distribution, predictions
