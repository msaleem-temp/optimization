# CellMap Segmentation Challenge: Setup and Optimization Guide

This guide provides a complete, step-by-step workflow for downloading the required datasets and running the optimization pipeline for the CellMap Segmentation Challenge.

## Part 1: Downloading the Data

### Step 1: Install Required Packages
First, install the necessary dependencies and clone the official repository:
```bash
pip install s3fs pydantic-zarr xarray-ome-ngff tqdm
git clone https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git
pip install -e ./cellmap-segmentation-challenge
```

### Step 2: Run the Download Script
Execute the data download script:
1. Run the `data-download.py` script.
2. **Important:** Make sure to change the directory path at the end of the script to your desired location:
   ```python
   download_directory = "/your_directory_path/"
   ```

### Step 3: Verify the Download
Once the script finishes, you will see a `raw_data` folder in your specified directory. It should contain all 22 downloaded datasets.

---

## Part 2: Running Optimization

### Step 1: Install Optimization Packages
Install the packages required for model training and optimization:
```bash
pip install pydantic-zarr xarray-ome-ngff monai optuna
```

### Step 2: Configure File Paths
Set up your file paths in your script for JSON configurations, datasets, and databases. Update the placeholder directories (`/your directory/`) with your actual system paths:

```python
# Centroids configuration
json_path = "/your directory/all_centroids.json"
save_path = "/your directory/targets_classes.json" # Stores only the 13 target classes

# Patch JSONs setup
input_json = "/your directory/targets_classes.json" # Path where target_classes centroids are saved
output_dir = "/your directory/patch_json/" # Directory where train.json and val.json will be saved

# Data paths
data_root = "/your directory/raw_data" # Address where all 22 datasets are downloaded
ZARR_MAP = build_zarr_map_modal_direct()

# Training and validation JSONs
train_json_path = "/mnt/voxelcell_vol1/patch_json/train.json"
val_json_path = "/mnt/voxelcell_vol1/patch_json/val.json"

# Optuna Database configurations
working_db = "/root/optuna_class_weights_run.db"
previous_db_path = "/mnt/voxelcell_vol1/backup/optuna_class_weights_run.db" # Note: First time this is executed, this db will not be present
```

### Step 3: Optuna Checkpointing
Ensure your script is configured to store the Optuna data database (`working_db`). This allows saving and resuming trials for multiple checkpoints seamlessly across runs.
