
import os
import json
import zarr
import numpy as np
import torch
import optuna
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import shutil
import random
import torch.nn as nn
from torch.optim import AdamW
from monai.transforms import Compose, RandFlipd, RandRotate90d
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

import collections



# ===================================================================================
# Extracting targeted organelle classes from ofline sampling json
"""
====== Note ======
There are 47 classes in cellMap dataset, I extracted centroids/ anchors of all classes. 
But we are working on 13 classes (target_classes). 

"""

json_path = "/mnt/voxelcell_vol1/patch_json/all_centroids.json" # path of centroid.json that have all centroids/anchors

save_path = "/mnt/voxelcell_vol1/patch_json/targets_classes.json" # path to store target_classes.json.

target_classes = {
    'endo', 'ld', 'lyso', 'mito', 'mt', 'np', 'nuc', 
    'perox', 'ves', 'vim', 'golgi', 'er', 'eres'
}

with open(json_path, 'r') as f:
    blueprint = json.load(f)

target_patches = [patch for patch in blueprint if patch.get("class") in target_classes]

print(f"Saving filtered targets to {save_path}...")
with open(save_path, 'w') as f:
    json.dump(target_patches, f, indent=4)

print(f"Total original patches: {len(blueprint)}")
print(f"Total target patches extracted: {len(target_patches)}")



# ===================================================================================
# Split into Train and Validation

input_json = "/mnt/voxelcell_vol1/patch_json/targets_classes.json" # path where target_classes centroids are saved
output_dir = "/mnt/voxelcell_vol1/patch_json/" # folder/directory where train.json and val.json will be saved

# Load patches
with open(input_json, 'r') as f:
    patches = json.load(f)

# Shuffle reproducibly
random.seed(42)
random.shuffle(patches)

# Total number of patches
total_patches = len(patches)

# Split: 90% Train, 10% Validation
train_end = int(total_patches * 0.90)

train_patches = patches[:train_end]
val_patches = patches[train_end:]

# Output paths
train_path = os.path.join(output_dir, "train.json")
val_path = os.path.join(output_dir, "val.json")

# Save Train JSON
with open(train_path, 'w') as f:
    json.dump(train_patches, f, indent=4)

# Save Validation JSON
with open(val_path, 'w') as f:
    json.dump(val_patches, f, indent=4)

# Print results
print(f"Total Patches: {total_patches}")
print(f"-> Train Dataset (90%): {len(train_patches)}")
print(f"-> Val Dataset   (10%): {len(val_patches)}")

# train and val paths will be suc as
# train_path = "/mnt/voxelcell_vol1/patch_json/train.json"
# val_path = "/mnt/voxelcell_vol1/patch_json/val.json"



# ===================================================================================
# This is very important function it aligns Label volume to EM volume

def extract_safe(zarr_arr, start_coords, patch_shape, pad_value=0, out_dtype=None):
    arr_shape = zarr_arr.shape
    z_min, z_max = max(0, start_coords[0]), min(arr_shape[0], start_coords[0] + patch_shape[0])
    y_min, y_max = max(0, start_coords[1]), min(arr_shape[1], start_coords[1] + patch_shape[1])
    x_min, x_max = max(0, start_coords[2]), min(arr_shape[2], start_coords[2] + patch_shape[2])

    target_dtype = out_dtype if out_dtype is not None else zarr_arr.dtype
    patch = np.full(patch_shape, fill_value=pad_value, dtype=target_dtype)
    
    pz_min, py_min, px_min = z_min - start_coords[0], y_min - start_coords[1], x_min - start_coords[2]
    
    if z_max > z_min and y_max > y_min and x_max > x_min:
        patch[pz_min:pz_min+(z_max-z_min), py_min:py_min+(y_max-y_min), px_min:px_min+(x_max-x_min)] = \
            zarr_arr[z_min:z_max, y_min:y_max, x_min:x_max]
            
    return patch



# ===================================================================================
#  It maks all dataset in dict so that dataset class can read all dataset. 

def build_zarr_map_modal_direct(data_root):
    zarr_map = {}
    
    # Target exactly two levels deep: data_root / folder / dataset.zarr
    search_pattern = os.path.join(data_root, "*", "*.zarr")
    
    # glob.glob without recursive=True is extremely fast here
    for zarr_path in glob.glob(search_pattern):
        dataset_name = os.path.basename(zarr_path).replace(".zarr", "")
        
        if dataset_name not in zarr_map:
            zarr_map[dataset_name] = []
            
        # Prevent duplicates
        if zarr_path not in zarr_map[dataset_name]:
            zarr_map[dataset_name].append(zarr_path)
            
    return zarr_map


# ===================================================================================
# Dataset class

class Patches(Dataset):
    
    def __init__(self, json_path, zarr_map, patch_dim=128, max_jitter=32):
        self.patch_dim = patch_dim
        self.max_jitter = max_jitter
        self.zarr_map = zarr_map
        
        with open(json_path, 'r') as f:
            raw_patches = json.load(f)
            
        self.zarr_cache = {}

        semantic_to_instance_map = {
            3: 1, 4: 1, 5: 1,                                       # 1. Mitochondria
            8: 2, 9: 2,                                             # 2. Vesicles
            10: 3, 11: 3,                                           # 3. Endosomes
            12: 4, 13: 4,                                           # 4. Lysosomes
            14: 5, 15: 5,                                           # 5. Lipid Droplets
            20: 6, 21: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, # 6. Nucleus
            22: 7, 23: 7,                                           # 7. Nuclear Pores
            30: 8, 36: 8,                                           # 8. Microtubules
            47: 9, 48: 9,                                           # 9. Peroxisomes
            6: 10, 7: 10,                                           # 10. Golgi Apparatus
            16: 11, 17: 11,                                         # 11. Endoplasmic Reticulum
            18: 12, 19: 12,                                         # 12. ER Exit Sites
            38: 13,                                                 # 13. Vimentin
        }
        
        self.label_lookup = np.zeros(256, dtype=np.int64)
        for semantic_id, instance_id in semantic_to_instance_map.items():
            self.label_lookup[semantic_id] = instance_id

        self.patches = []
        missing_count = 0
        
        for patch in raw_patches:
            dataset = patch["dataset"]
            crop_id = patch["crop"]
            em_lvl = str(patch["em_scale"])
            lbl_lvl = str(patch["label_scale"])
            
            crop_found = False
            
            if dataset in self.zarr_map:
                for zarr_path in self.zarr_map[dataset]:
                    base_recon_path = os.path.join(zarr_path, "recon-1")
                    em_path = os.path.join(base_recon_path, "em", "fibsem-uint8", em_lvl)
                    label_path = os.path.join(base_recon_path, "labels", "groundtruth", crop_id, "all", lbl_lvl)
                    
                    if os.path.exists(em_path) and os.path.exists(label_path):
                        crop_found = True
                        break
                        
            if crop_found:
                self.patches.append(patch)
            else:
                missing_count += 1
                
        print(f"Dataset initialized. Retained {len(self.patches)} valid patches. Pruned {missing_count} missing patches.")

    def __len__(self):
        return len(self.patches)

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]
            
        if dataset not in self.zarr_map:
            raise FileNotFoundError(f"Dataset '{dataset}' was not found in the Kaggle directory scan.")
            
        valid_em_path = None
        valid_label_path = None
        
        for zarr_path in self.zarr_map[dataset]:
            base_recon_path = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon_path, "em", "fibsem-uint8", str(em_scale))
            temp_label = os.path.join(base_recon_path, "labels", "groundtruth", crop_id, "all", str(label_scale))
            
            if os.path.exists(temp_label) and os.path.exists(temp_em):
                valid_em_path = temp_em
                valid_label_path = temp_label
                break
                
        if not valid_em_path:
            raise FileNotFoundError(f"Crop {crop_id} for dataset '{dataset}' could not be found in any available parts.")
            
        em_zarr = zarr.open(valid_em_path, mode='r')
        label_zarr = zarr.open(valid_label_path, mode='r')
        
        self.zarr_cache[cache_key] = (em_zarr, label_zarr)
        return em_zarr, label_zarr

    def __getitem__(self, idx):
        patch = self.patches[idx]
        dataset = patch["dataset"]
        crop_id = patch["crop"]
        
        em_lvl = patch["em_scale"]
        lbl_lvl = patch["label_scale"]
        
        # 1. Load exact centers from your JSON
        l_center = np.array(patch["l_center"], dtype=float)
        e_center = np.array(patch["e_center"], dtype=float)
        e_shape = np.array(patch["e_shape"], dtype=int)
        
        # 2. Fetch Cached Zarr Handles
        em_zarr, label_zarr = self._get_zarr_handles(dataset, crop_id, em_lvl, lbl_lvl)
        
        # 3. Base Mathematical Centering
        base_l_start = np.floor(l_center - (self.patch_dim / 2.0)).astype(int)
        base_e_start = np.floor(e_center - (e_shape / 2.0)).astype(int)
        
        # 4. Generate Spatial Jitter using PyTorch RNG
        jitter_vector = torch.randint(-self.max_jitter, self.max_jitter + 1, (3,)).numpy()
        
        # 5. Dynamic Boundary Clamping (Reading directly from the opened Zarr handle)
        lbl_shape = np.array(label_zarr.shape)
        max_l_start = np.maximum(lbl_shape - self.patch_dim, 0)
        
        clamped_l_start = np.clip(base_l_start + jitter_vector, 0, max_l_start)
        effective_jitter = clamped_l_start - base_l_start
        clamped_e_start = base_e_start + effective_jitter
        
        # 6. Extraction
        lbl_np = extract_safe(label_zarr, clamped_l_start, [self.patch_dim, self.patch_dim, self.patch_dim], pad_value=0, out_dtype=np.int64)
        em_np = extract_safe(em_zarr, clamped_e_start, e_shape.tolist(), pad_value=0)
        
        # 7. Semantic Remapping and Tensor Conversion
        remapped_lbl = self.label_lookup[lbl_np]
            
        em_tensor = torch.from_numpy(em_np.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(remapped_lbl)
        
        return em_tensor, lbl_tensor




# ===================================================================================
# WeightedRandonSampler

def create_balanced_sampler(dataset):
    class_list = [patch.get("class", "unknown") for patch in dataset.patches]
    class_counts = collections.Counter(class_list)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[patch.get("class", "unknown")] for patch in dataset.patches]
    
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    return sampler
# --------------------------------------------------


data_root="/mnt/voxelcell_vol1/raw_data" # folder address where all datasets are downloaded. 
ZARR_MAP = build_zarr_map_modal_direct()

train_json_path = "/mnt/voxelcell_vol1/patch_json/train.json" # train.json path such as "/mnt/voxelcell_vol1/patch_json/train.json"
val_json_path = "/mnt/voxelcell_vol1/patch_json/val.json"

train_dataset = Patches(train_path, zarr_map=ZARR_MAP, patch_dim=128, max_jitter=48)
val_dataset = Patches(val_path, zarr_map=ZARR_MAP, patch_dim=128, max_jitter=0) # Static for validation

train_sampler = create_balanced_sampler(train_dataset)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4,
    sampler=train_sampler, 
    num_workers=2,   
    pin_memory=True,
    drop_last=True  
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2,   
    pin_memory=True,
    drop_last=False
)



"""
Output should:

Dataset initialized. Retained 2470 valid patches. Pruned 0 missing patches.
Dataset initialized. Retained 275 valid patches. Pruned 0 missing patches.

"""



# ==============================================================================

# Optimization Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print(f"Training on device: {device} with {num_gpus} GPUs")


# --- OPTUNA DATABASE SETUP ---
working_db = "/root/optuna_class_weights_run.db"


previous_db_path = "/mnt/voxelcell_vol1/backup/optuna_class_weights_run.db" # First time when it will executed this db will not be present.

if os.path.exists(previous_db_path) and not os.path.exists(working_db):
    print("Found existing database. Copying to root directory to resume...")
    shutil.copy(previous_db_path, working_db)
elif os.path.exists(working_db):
    print("Working database already exists in root. Continuing...")
else:
    print("No previous database found. Starting a fresh study...")

db_url = f"sqlite:///{working_db}"



# ==============================================================================
# Optuna Objective Function.

def objective(trial):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Sample raw float ranges for all 14 classes
    raw_weights = [trial.suggest_float(f"w_class_{i}", 0.1, 2.0) for i in range(14)]
    
    # 2. Normalize so they sum to exactly 14.0
    weight_tensor = torch.tensor(raw_weights, dtype=torch.float32)
    normalized_weights = (weight_tensor / weight_tensor.sum()) * 14.0
    dynamic_weights = normalized_weights.to(device)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Applied Normalized Weights: {dynamic_weights.cpu().numpy().round(3)}")

    # 3. Initialize Model and Training Components
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,  
        channels=(64, 128, 256, 512, 1024),
        strides=(1, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        act="PRELU",
        norm="INSTANCE"
    ).to(device)


    if num_gpus > 1:
        model = nn.DataParallel(model)

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=dynamic_weights
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    post_label = AsDiscrete(to_onehot=14)

    num_epochs = 20 
    accumulation_steps = 2
    print_freq = 200
    best_mean_dice = -1.0 # Initialize to negative for maximization

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Trial {trial.number}] === Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} ===")
        
        # --- PHASE 1: TRAINING ---
        model.train()
        train_epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (em_batch, lbl_batch) in enumerate(train_dataloader):
            em_batch = em_batch.to(device)
            if lbl_batch.dim() == 4:
                lbl_batch = lbl_batch.unsqueeze(1)
            lbl_batch = lbl_batch.to(device, dtype=torch.long)
            
            with autocast():
                outputs = model(em_batch)
                loss = criterion(outputs, lbl_batch)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_epoch_loss += (loss.item() * accumulation_steps)
            
            if (step + 1) % print_freq == 0:
                print(f"  [Train] Step {step+1}/{len(train_dataloader)} | Loss: {(loss.item() * accumulation_steps):.4f}")
                
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        
        # --- PHASE 2: VALIDATION (DICE SCORE) ---
        model.eval()
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for step, (em_batch, lbl_batch) in enumerate(val_dataloader):
                em_batch = em_batch.to(device)
                if lbl_batch.dim() == 4:
                    lbl_batch = lbl_batch.unsqueeze(1)
                lbl_batch = lbl_batch.to(device, dtype=torch.long)
                
                with autocast():
                    outputs = model(em_batch)
                    val_loss = criterion(outputs, lbl_batch)
                    
                val_epoch_loss += val_loss.item()
                
                # Apply discretization for metric computation
                val_outputs = [post_pred(i) for i in outputs]
                val_labels = [post_label(i) for i in lbl_batch]
                
                # Accumulate metric for the batch
                dice_metric(y_pred=val_outputs, y=val_labels)
                    
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        
        # Aggregate the final Dice score for the epoch and reset the metric
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        
        # Scheduler now monitors the Dice score
        scheduler.step(mean_dice)
        
        print(f"[Trial {trial.number}] Epoch [{epoch+1}/{num_epochs}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Mean Dice: {mean_dice:.4f}")

        # --- PHASE 3: OPTUNA PRUNING & REPORTING ---
        # Optuna tracks mean_dice instead of loss
        trial.report(mean_dice, epoch)
        if trial.should_prune():
            print(f">>> Trial {trial.number} pruned at epoch {epoch} due to poor Dice score.")
            raise optuna.exceptions.TrialPruned()

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice

    print(f"Trial {trial.number} Finished | Final Best Mean Dice: {best_mean_dice:.4f}")
    return best_mean_dice


# ===================================================================================
#  EXECUTE STUDY 



study = optuna.create_study(
    study_name="unet_14class_weight_sweep",
    storage=db_url,
    load_if_exists=True,
    direction="maximize", 
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
)

study.optimize(objective, n_trials=5) # Adjust number of trails 





print("\n=== Optimization Batch Complete ===")
best_trial = study.best_trial
print(f"  Best Mean Dice: {best_trial.value:.4f}")
print("  Optimal Class Weights (Pre-Normalized Sample):")




# Compute the actual applied normalized weights for the best trial safely:
best_raw_weights = torch.tensor([best_trial.params[f"w_class_{i}"] for i in range(14)], dtype=torch.float32)
best_normalized_weights = (best_raw_weights / best_raw_weights.sum()) * 14.0

for i in range(14):
    key = f"w_class_{i}"
    print(f"    {key}: {best_normalized_weights[i]:.4f}")
