#! /usr/bin/env bash

SAVE_DIR=/mnt/nas25/yihan01.hu/workspace/test/
EXPERIMENT=smart_train_bf16_reduce_radius_more_b8
CACHE_DIR="/home/users/yihan01.hu/data/test" 

export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/third_party/SMART:$PYTHONPATH
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache/trainval"
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
# export TF_NUM_INTRAOP_THREADS=1
# export TF_NUM_INTEROP_THREADS=1
# export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
# export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.force_feature_computation=true \
    cache.use_cache_without_dataset=false \
    cache.versatile_caching=false \
    experiment_name=$EXPERIMENT \
    py_func=train \
    seed=0 \
    +training=training_wod_smart \
    scenario_builder=wod \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=4 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.precision=bf16 \
    lightning.trainer.params.gradient_clip_val=5.0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    +lightning.trainer.params.val_check_interval=0.00000001 \
    lightning.trainer.params.accumulate_grad_batches=1 \
    data_loader.params.batch_size=1\
    data_loader.params.num_workers=16 \
    worker=single_machine_thread_pool \
    model=gump_wod_smart \
    optimizer=adamw \
    optimizer.lr=4e-4 \
    optimizer.weight_decay=0.0 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[3] \
    lr_scheduler.gamma=0.2 \
    lightning.trainer.checkpoint.resume_training=false \
    scenario_filter=all_scenarios \
    +checkpoint.ckpt_path='/mnt/nas25/yihan01.hu/workspace/test/smart_train_bf16/training_world_model/2024.10.08.23.59.12/best_model/last.ckpt' \
    +checkpoint.strict=False \
    +checkpoint.resume=False 

# +checkpoint.ckpt_path="/mnt/nas25/yihan01.hu/workspace/test/nuplan_32x64/training_world_model/2024.09.15.09.28.57/best_model/last.ckpt" \
