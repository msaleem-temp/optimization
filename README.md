# 3D Organelle Segmentation: Class Weight Optimization Pipeline

This repository contains the scripts necessary to download the 3D biological volumes from Janelia and execute a Bayesian hyperparameter sweep (using Optuna) to find the optimal sum-normalized class weights for the `DiceCELoss` metric.

## Part 1: Downloading the Volumetric Data

The dataset relies on maintaining the exact 128-pixel spatial padding applied by the CellMap Segmentation Challenge CLI. 

### Step 1: Install Download Dependencies
Run the following commands in your terminal to install the necessary file system libraries and the Janelia `csc` tool directly from the source repository:

```bash
pip install s3fs pydantic-zarr xarray-ome-ngff tqdm
git clone [https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git](https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git)
pip install -e ./cellmap-segmentation-challenge
