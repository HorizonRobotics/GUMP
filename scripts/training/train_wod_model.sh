#! /usr/bin/env bash
SAVE_DIR=/mnt/nas25/yihan01.hu/tmp/save_dir/
EXPERIMENT=wod_test
CACHE_DIR=/mnt/nas25/yihan01.hu/tmp/cache_wod_dir/

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.force_feature_computation=false \
    cache.use_cache_without_dataset=true \
    cache.versatile_caching=false \
    experiment_name=$EXPERIMENT \
    py_func=train \
    seed=0 \
    +training=training_wod_gump \
    scenario_builder=wod \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=15 \
    lightning.trainer.params.max_time=14:32:00:00\
    lightning.trainer.params.precision=bf16 \
    lightning.trainer.params.gradient_clip_val=5.0 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
    +lightning.trainer.params.val_check_interval=1.0 \
    lightning.trainer.params.accumulate_grad_batches=4\
    data_loader.params.batch_size=1 \
    data_loader.params.num_workers=2 \
    worker=single_machine_thread_pool \
    model=gump_wod_gptm \
    optimizer=adamw \
    optimizer.lr=2e-4 \
    optimizer.weight_decay=1e-3 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[7,13,14] \
    lr_scheduler.gamma=0.2 \
    scenario_filter=training_scenarios \
    +checkpoint.ckpt_path=null \
    +checkpoint.resume=False \
    +checkpoint.strict=True 
    # +checkpoint.ckpt_path='/mnt/nas25/yihan01.hu/gptm_wod.ckpt' \

    