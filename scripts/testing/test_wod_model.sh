#! /usr/bin/env bash
SAVE_DIR=/mnt/nas25/yihan01.hu/tmp/save_dir/
EXPERIMENT=wod_test
CACHE_DIR=/mnt/nas25/yihan01.hu/tmp/cache_wod_dir/

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYGLIB_RESOURCE_PATH=/usr/local/lib/python3.9/dist-packages

python -W ignore $PWD/nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.force_feature_computation=true \
    cache.use_cache_without_dataset=false \
    experiment_name=$EXPERIMENT \
    py_func=test \
    seed=0 \
    +training=training_wod_gump \
    scenario_builder=wod \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.max_epochs=8 \
    lightning.trainer.params.max_time=07:32:00:00\
    lightning.trainer.params.precision=16 \
    data_loader.params.batch_size=2 \
    data_loader.params.num_workers=8 \
    worker=single_machine_thread_pool \
    model=gump_wod_gptm \
    optimizer.lr=0.0002 \
    optimizer.weight_decay=0.0 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones=[8,10] \
    lightning.trainer.checkpoint.resume_training=false \
    scenario_filter=training_scenarios\
    +checkpoint.ckpt_path='/mnt/nas26/yihan01.hu/tmp/gptm_wod.ckpt' \
    +checkpoint.resume=True \
    +checkpoint.strict=True  
