
import os
import subprocess
import shutil
import sys

def get_all_crops(dataset):

    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    base_path = f"janelia-cosem-datasets/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth"
    
    try:
        all_items = fs.ls(base_path)
        crops = [item.split('/')[-1].replace('crop', '') 
                 for item in all_items if item.split('/')[-1].startswith('crop')]
        crops.sort()
  
        return crops 
        
    except FileNotFoundError:
        return []

def fetch_datasets_locally(dataset_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(save_dir)
    
    for dataset in dataset_list:
        print(f"\n--- Processing Dataset: {dataset} ---")
        
        numeric_crops = get_all_crops(dataset)
        if not numeric_crops:
            print(f"Warning: No crops found for {dataset}. Skipping.")
            continue
            
        print(f"Found {len(numeric_crops)} crops for {dataset}. (Dry run limit applied)")
        crop_string = ",".join(numeric_crops)
        
        download_command = f"csc fetch-data -d {dataset} -c {crop_string} --raw-padding 128"
        print(f"Executing: {download_command}")
        
        process = subprocess.Popen(
            download_command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        print(f"Dataset {dataset} tool execution finished with return code: {process.returncode}")

        expected_target_dir = os.path.join(save_dir, dataset)
        redundant_nested_dir = os.path.join(expected_target_dir, dataset)
        
        if os.path.exists(redundant_nested_dir):
            print(f"Automatically correcting double-nested directory for {dataset}...")
            for item in os.listdir(redundant_nested_dir):
                src = os.path.join(redundant_nested_dir, item)
                dst = os.path.join(expected_target_dir, item)
                shutil.move(src, dst)
            os.rmdir(redundant_nested_dir)
            
    os.chdir(original_dir)
    
    print("\n=== DOWNLOAD TEST FINISHED ===")



target_datasets = [
    "jrc_cos7-1a",
    "jrc_cos7-1b",
    "jrc_ctl-id8-1",
    "jrc_fly-mb-1a",
    "jrc_fly-vnc-1",
    "jrc_hela-2",
    "jrc_hela-3",
    "jrc_jurkat-1",
    "jrc_macrophage-2",
    "jrc_mus-heart-1",
    "jrc_mus-kidney",
    "jrc_mus-kidney-3",
    "jrc_mus-kidney-glomerulus-2",
    "jrc_mus-liver",
    "jrc_mus-liver-3",
    "jrc_mus-liver-zon-1",
    "jrc_mus-liver-zon-2",
    "jrc_mus-nacc-1",
    "jrc_sum159-1",
    "jrc_sum159-4",
    "jrc_ut21-1413-003",
    "jrc_zf-cardiac-1"
]

download_directory = "" #directory or folder to save data locally.

fetch_datasets_locally(target_datasets, download_directory)
