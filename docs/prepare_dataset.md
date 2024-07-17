# Dataset Preparation

## Reformatting the Original Dataset

### Waymo Dataset
To reformat Waymo's scenario protocol buffer data, download the dataset and then modify the `src_path` and `dst_path` in the script. Execute the following command to split the data:

```bash
python scripts/preprocess/process_wod_data.py
```

### Nuplan Dataset
For instructions on setting up the Nuplan dataset, refer to [Nuplan Devkit Documentation](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html).

## Caching the Dataset
To train efficiently, it is essential to cache the data first. Follow these steps:

### Cache Dataset v1.0 (slow)

#### Waymo Dataset
Run the following script to cache the Waymo dataset:

```bash
sh ./scripts/preprocess/cache_wod_data.sh
```

#### Nuplan Dataset
Run the following script to cache the Nuplan dataset:

```bash
sh ./scripts/preprocess/cache_nuplan_data.sh
```

### Cache Dataset v1.1 (fast)

#### Waymo Dataset
The v1.1 cache script for the Waymo dataset has not yet been released. Stay tuned for updates!

#### Nuplan Dataset
Run the following script to cache the Nuplan dataset version 1.1:

```bash
sh ./scripts/preprocess/cache_nuplan_data_v1_1.sh
```