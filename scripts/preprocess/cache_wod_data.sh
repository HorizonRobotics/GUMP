#! /usr/bin/env bash
SAVE_DIR="/mnt/nas25/yihan01.hu/tmp/save_dir/"
EXPERIMENT="cache_wod_data"
CACHE_DIR="/mnt/nas25/yihan01.hu/tmp/cache_wod_dir/"
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/third_party:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

python nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    experiment_name=$EXPERIMENT \
    job_name=caching \
    py_func=cache \
    +training=training_wod_gump \
    scenario_builder=wod \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=32 \
    model=gump_wod_gptm \
    scenario_builder.data_root=/mnt/nas20/zhening.yang/wod/scenario_pkl_v1_2/ \
    scenario_builder.training_token_list_path=/mnt/nas20/zhening.yang/wod/scenario_pkl_v1_2/sdc_raw_merge_tokens/training_token_list.txt \
    scenario_builder.validation_token_list_path=/mnt/nas20/siqi01.chai/wod_splits/validation_token_list_256.txt \
    scenario_builder.subsample_ratio=10000 \
    cache.force_feature_computation=true
