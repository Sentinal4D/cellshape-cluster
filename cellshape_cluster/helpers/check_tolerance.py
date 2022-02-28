import numpy as np


def check_tolerance(cluster_predictions, previous_cluster_predictions):
    delta_label = (
        np.sum(cluster_predictions != previous_cluster_predictions).astype(
            np.float32
        )
        / cluster_predictions.shape[0]
    )
    previous_cluster_predictions = np.copy(cluster_predictions)
    return delta_label, previous_cluster_predictions
