[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Version](https://img.shields.io/pypi/pyversions/cellshape-cluster.svg)](https://pypi.org/project/cellshape-cluster)
[![PyPI](https://img.shields.io/pypi/v/cellshape-cluster.svg)](https://pypi.org/project/cellshape-cluster)
[![Downloads](https://pepy.tech/badge/cellshape-cluster)](https://pepy.tech/project/cellshape-cluster)
[![Wheel](https://img.shields.io/pypi/wheel/cellshape-cluster.svg)](https://pypi.org/project/cellshape-cluster)
[![Development Status](https://img.shields.io/pypi/status/cellshape-cluster.svg)](https://github.com/Sentinal4D/cellshape-cluster)
[![Tests](https://img.shields.io/github/workflow/status/Sentinal4D/cellshape-cluster/tests)](
    https://github.com/Sentinal4D/cellshape-cluster/actions)
[![Coverage Status](https://coveralls.io/repos/github/Sentinal4D/cellshape-cluster/badge.svg?branch=master)](https://coveralls.io/github/Sentinal4D/cellshape-cluster?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# cellshape-cluster
___
Cellshape-cluster is an easy-to-use tool to analyse the cluster cells by their shape using deep learning and, in particular, deep-embedded-clustering. The tool provides the ability to train popular graph-based or convolutional autoencoders on point cloud or voxel data of 3D single cell masks as well as providing pre-trained networks for inference.


## To install
```bash
pip install cellshape-cluster
```

## Usage
```python
import torch
from cellshape_cloud import CloudAutoEncoder
from cellshape_cluster import DeepEmbeddedClustering

autoencoder = CloudAutoEncoder(
    num_features=128, 
    k=20, 
    encoder_type="dgcnn"
)

model = DeepEmbeddedClustering(autoencoder=autoencoder, 
                               num_clusters=10,
                               alpha=1.0)

points = torch.randn(1, 2048, 3)

recon, features, clusters = model(points)
```

## Parameters

- `autoencoder`: CloudAutoEncoder or VoxelAutoEncoder.  
Instance of autoencoder class from cellshape-cloud or cellshape-voxel
- `num_clusters`: int.  
The number of clusters to use in deep embedded clustering algorithm.
- `alpha`: float.  
Degrees of freedom for the Student's t-distribution. Xie et al. (ICML, 2016) let alpha=1 for all experiments.

## For developers
* Fork the repository
* Clone your fork
```bash
git clone https://github.com/USERNAME/cellshape-cluster 
```
* Install an editable version (`-e`) with the development requirements (`dev`)
```bash
cd cellshape-cluster
pip install -e .[dev] 
```
* To install pre-commit hooks to ensure formatting is correct:
```bash
pre-commit install
```

* To release a new version:

Firstly, update the version with bump2version (`bump2version patch`, 
`bump2version minor` or `bump2version major`). This will increment the 
package version (to a release candidate - e.g. `0.0.1rc0`) and tag the 
commit. Push this tag to GitHub to run the deployment workflow:

```bash
git push --follow-tags
```

Once the release candidate has been tested, the release version can be created with:

```bash
bump2version release
```
