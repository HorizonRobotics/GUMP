#! /usr/bin/env bash
SAVE_DIR=/mnt/nas25/jiaqi.song/workspace/eval/wod_exp/
EXPERIMENT=wod_test
CACHE_DIR="/mnt/nas25/jiaqi.song/tmp/cache_wod_dir_complete/" 
CACHE_META_PATH=/mnt/nas25/jiaqi.song/tmp/cache_wod_dir_complete/metadata/cache_wod_dir_complete_metadata_node_0.csv

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/third_party:$PYTHONPATH
export CUDA118_ENV_PATH=/mnt/nas25/jiaqi.song/miniconda3/envs/hoplan_cuda118
export PATH=$CUDA118_ENV_PATH/bin:$PATH

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.cache_metadata_path=$CACHE_META_PATH \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    experiment_name=$EXPERIMENT \
    py_func=train \
    seed=0 \
    +training=training_wod_gump_v1_1 \
    scenario_builder=wod \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=15 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.precision=bf16 \
    lightning.trainer.params.gradient_clip_val=5.0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    +lightning.trainer.params.val_check_interval=1.0 \
    lightning.trainer.params.accumulate_grad_batches=2\
    data_loader.params.batch_size=8 \
    data_loader.params.num_workers=8 \
    worker=single_machine_thread_pool \
    model=gump_wod_llama_sm_v1_1 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    optimizer.weight_decay=1e-3 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[8,13] \
    lr_scheduler.gamma=0.2 \
    scenario_filter=training_scenarios \
    +checkpoint.ckpt_path=null \
    +checkpoint.resume=false \
    +checkpoint.strict=false 

    

    