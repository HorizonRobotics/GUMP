# Training and Evaluation Guide


## set dataset parameters before training or evaluation
set dataset parameters in `nuplan_extent/planning/script/config/common_cfg/gump.yaml`
To train on Nuplan dataset:
```yaml
    dataset: 'nuplan'
    num_past_steps: 12
    past_time_horizon: 6.0 # training
```
to run simulation on Nuplan Dataset
```yaml
    dataset: 'nuplan'
    num_past_steps: 2
    past_time_horizon: 1.0 # simulation
```
To train or eval on Waymo Dataset
```yaml
    dataset: 'waymo'
    num_past_steps: 2
    past_time_horizon: 1.0
```

## Version 1.0

### Training
To train on the Nuplan Dataset, execute the following command:
```bash
sh scripts/training/train_nuplan_model.sh
```

To train on the Waymo Open Dataset, execute the following command:
```bash
sh scripts/training/train_wod_model.sh
```

## Version 1.1

### Training

To train on the Nuplan Dataset version 1.1, execute the following command:

```bash
sh scripts/training/train_nuplan_model_1_1.sh
```

To train on the Waymo Open Dataset version 1.1, execute the following command:

```bash
sh scripts/training/train_wod_model_1_1.sh
```

## Evaluation

### Waymo Sim Agents Benchmark

Sure, here's the refined format for fixing the bugs in the Waymo Open Dataset package:

### Steps to Fix Bugs in Waymo Open Dataset Package

1. **Install the Waymo Open Dataset Package**
   Make sure you have installed the Waymo Open Dataset package by running:
   ```bash
   pip install waymo-open-dataset-tf-2-12-0==1.6.4
   ```

2. **Modify the File**
   Edit the file located at `/usr/local/lib/python3.9/dist-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py`. Change the code at line 47 from:
   ```python
   config_path = '{pyglib_resource}waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto'.format(pyglib_resource='')
   ```
   to:
   ```python
   import os
   # Get the resource path from an environment variable
   pyglib_resource = os.getenv('PYGLIB_RESOURCE_PATH', '')

   # Construct the full config path
   config_path = f'{pyglib_resource}/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_config.textproto'
   ```

3. **Set Environment Variable**
   Set the environment variable for the resource path by running:
   ```bash
   export PYGLIB_RESOURCE_PATH=/usr/local/lib/python3.9/dist-packages
   ```

By following these steps, you will ensure that the configuration path is correctly set and the package functions properly.



#### Start Evaluation
To evaluate on the Waymo Sim Agents Benchmark, follow these steps:

1. Dump the simulated result offline by uncommenting the following lines in `nuplan_extent/planning/script/experiments/training/training_wod_gump.yaml`:

    ```yaml
    - override /aggregated_metric:
        - sim_agents_metrics 
    ```
    and set your dump path in `nuplan_extent/planning/script/config/training/aggregated_metric/sim_agents_metrics.yaml`:

    ```yaml
    basepath: ${YOUR_PKL_DUMP_PATH}
    ```

2. Modify the configuration in `nuplan_extent/planning/script/config/common_cfg/gump.yaml`:

    ```yaml
    ...
    dataset: 'waymo'
    num_past_steps: 2
    past_time_horizon: 1.0
    ...
    ```

3. Ensure you set the downstream task to 'sim_agents':

    ```yaml
    ...
    downstream_task: 'sim_agents'
    ...
    ```
4. Running the testing script:
    ```bash
    sh ./scripts/testing/test_wod_model.sh
    ```

5. Running the offline evaluation script with multiprocessing
    ```bash
    python ./scripts/testing/offline_sim_agents.py --pkl_dir ${YOUR_PKL_DUMP_PATH} --nprocess 32 --output_dir ${SMOOTHING_OUTPUT_DIR}
    ```


### Visualizing Sim Agents Results on Waymo Open Dataset

After dumping the result pickles, you can visualize the results by running:

```bash
python scripts/visualization/wod_simagents_visualization.py
```

## Checkpoints

We provide several checkpoints for model reproduction. To use a checkpoint, download it and replace the checkpoint path in the bash script:

```bash
+checkpoint.ckpt_path=PATH_TO_CHECKPOINT \
+checkpoint.strict=True \
```

### Checkpoint List
We provided the following CKPT:
| Model         | Version | Dataset | Config                                                                                      | Checkpoint                                                                                   |
|---------------|---------|---------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| GPT2-base     | v1.0    | NuPlan  | [config](../nuplan_extent/planning/script/config/common/model/gump_nuplan_gptbase.yaml)     |     coming soon!                                                                                        |
| GPT2-Medium   | v1.0    | NuPlan  | [config](../nuplan_extent/planning/script/config/common/model/gump_nuplan_gptm.yaml)        |                                                                                     coming soon!        |
| GPT2-Medium   | v1.0    | Waymo   | [config](../nuplan_extent/planning/script/config/common/model/gump_wod_gptm.yaml)           | [Google Cloud](https://storage.googleapis.com/93935945854-us-central1-blueprint-config/gptm_wod.ckpt) |coming soon!
| GPT2-base     | v1.1    | NuPlan  | [config](../nuplan_extent/planning/script/config/common/model/gump_nuplan_gptbase_v1_1.yaml)|   coming soon!                                                                                          |
| Llama3-Small  | v1.1    | NuPlan  | [config](../nuplan_extent/planning/script/config/common/model/gump_nuplan_lamma_sm_v1_1.yaml)| [Google Cloud](https://storage.googleapis.com/93935945854-us-central1-blueprint-config/epoch_llama_sm.ckpt) |

## TensorBoard Visualization

The visualization is automatically handled by TensorBoard. To visualize your training progress, use the following command:

```bash
tensorboard --logdir='${SAVE_DIR}/${EXPERIMENT}'
```