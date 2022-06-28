import torch
import numpy as np
from sklearn.cluster import KMeans


def kmeans(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    feature_array = None
    model.eval()
    for data in dataloader:
        inputs = data[0]
        inputs = inputs.to(device)
        output, features, clusters = model(inputs)
        if feature_array is not None:
            feature_array = np.concatenate(
                (feature_array, features.cpu().detach().numpy()), 0
            )
        else:
            feature_array = features.cpu().detach().numpy()

    km.fit_predict(feature_array)
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering_layer.set_weight(weights.to(device))

    return km
