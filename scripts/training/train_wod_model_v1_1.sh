#! /usr/bin/env bash

SAVE_DIR=/mnt/nas25/yihan01.hu/workspace/test/
EXPERIMENT=wod_full_32x64_from_scratch_v4_randstart_prob10_expdecay005_prev_action_100m_add_selfffn_tokenv3_decay0_redo_pos_embed_less_crossattn_add_pos_v
CACHE_DIR="/home/users/yihan01.hu/data/cache_wod_data_2HZ_v4" 

# export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache/trainval"
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
# export TF_NUM_INTRAOP_THREADS=1
# export TF_NUM_INTEROP_THREADS=1
# export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
# export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    cache.versatile_caching=false \
    experiment_name=$EXPERIMENT \
    py_func=train \
    seed=0 \
    +training=training_wod_gump_v1_1 \
    scenario_builder=wod \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=6 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.precision=bf16 \
    lightning.trainer.params.gradient_clip_val=5.0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    +lightning.trainer.params.val_check_interval=0.25 \
    lightning.trainer.params.accumulate_grad_batches=4 \
    data_loader.params.batch_size=4\
    data_loader.params.num_workers=16 \
    worker=single_machine_thread_pool \
    model=gump_wod_lamma_100m_v1_1_kigump \
    optimizer=adamw \
    optimizer.lr=2e-4 \
    optimizer.weight_decay=0.0 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[4,5] \
    lr_scheduler.gamma=0.2 \
    lightning.trainer.checkpoint.resume_training=false \
    scenario_filter=all_scenarios \
    +checkpoint.ckpt_path=null \
    +checkpoint.strict=False \
    +checkpoint.resume=False 

# +checkpoint.ckpt_path="/mnt/nas25/yihan01.hu/workspace/test/nuplan_32x64/training_world_model/2024.09.15.09.28.57/best_model/last.ckpt" \
