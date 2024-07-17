
# SAVE_DIR="/mnt/nas20/kun.li/workspace/trained_params/nuplan/simulation_save_dir"
# SAVE_DIR='/mnt/nas37/kun.li/workspace/trained_params/nuplan/simulation_save_dir'
SAVE_DIR='/mnt/nas37/kun.li/workspace/trained_params/nuplan/simulation_save_dir/nuplan_v1.1/raster_type50k'
# SAVE_DIR="/mnt/nas37/yihan01.hu/workspace/trained_params/nuplan/simulation_save_dir/"


# MODEL_PATH="/mnt/nas37/kun.li/workspace/policy_model/training/nuplan_v1.1/raster_35W/resnet18_unet_vcs_rearAxle_pertubation_11_trajectory_4s_epoch20/training_raster_model/2023.10.09.15.35.15/checkpoints/epoch\=19.ckpt"
MODEL_PATH="/mnt/nas37/kun.li/workspace/policy_model/training/nuplan_v1.1/raster_375k/resnet18_pertubation_11_trajectory_epoch20/training_raster_heatmap_occ_model/2023.08.13.23.21.18/checkpoints/epoch\=19.ckpt"


# EXPERIMENT="tmp"
# EXPERIMENT="nuplan_v1.1/raster_375k/resnet18_pertubation_11_trajectory_epoch20/world_model_ego_isLast"
# EXPERIMENT="nuplan_v1.1/pdm/"
EXPERIMENT="nuplan_v1.1/raster_375k/resnet18_pertubation_11_trajectory_epoch20/world_model_on_ego_reactivate"


# export NUPLAN_MAPS_ROOT="/mnt/nas20/zhuangzhuang.ding/nuplan/nuplan-maps-v1.0"
# export NUPLAN_MAPS_ROOT="/train-data/nuplanv1.1/maps"
export NUPLAN_MAPS_ROOT="/mnt/nas25/kun.li/nuplan/maps"
# export NUPLAN_DATA_ROOT="/mnt/nas20/zhuangzhuang.ding/nuplan/nuplan_mini"
# export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache/test/"
# export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache/trainval/"
# export NUPLAN_DATA_ROOT="/mnt/nas20/nuplan-v1.1/test/"
export NUPLAN_DATA_ROOT="/train-data/nuplanv1.1/data/cache/test/"
export NUPLAN_HYDRA_CONFIG_PATH=$PWD/nuplan_extent/planning/script/config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # #4,5 #,5,6,7,0,1,2,3

# export PYTHONPATH=/mnt/nas37/kun.li/hoplan_dev_2/nuplanv1.1:$PYTHONPATH
# export PYTHONPATH=$PWD:$PYTHONPATH


# conda env hoplan2
python nuplan_extent/planning/script/run_simulation.py \
    experiment_name=$EXPERIMENT \
    group=$SAVE_DIR \
    +simulation='closed_loop_reactive_agents' \
    planner=ml_planner \
    planner.ml_planner.model_config='${model}' \
    planner.ml_planner.checkpoint_path=$MODEL_PATH \
    common_cfg=policy_model \
    common_cfg.input_cfg.input_channel_indexes=[0,1,2,3,4,5] \
    common_cfg.output_cfg.trajectory_steps=16 \
    common_cfg.output_cfg.time_horizon=8. \
    scenario_builder=nuplan \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    scenario_builder.scenario_mapping.subsample_ratio_override=0.5 \
    model=policy_structured_plan_model_raster_naive \
    model.feature_builders.0.subsample_ratio_override=0.5 \
    scenario_filter.timestamp_threshold_s=15 \
    number_of_gpus_allocated_per_simulation=1.0 \
    scenario_builder.map_root=$NUPLAN_MAPS_ROOT \
    scenario_filter.num_scenarios_per_type=1 \
    ~callback.simulation_nuboard_video_callback \
    simulation_history_buffer_duration=6.0 \
    worker.threads_per_node=8 \
    max_callback_workers=8 \
    observation=world_model_agents_observation \
    callback.simulation_feature_video_callback.visualize_all_scenarios=True \
    scenario_filter.limit_total_scenarios=1 \
    # ~callback.simulation_feature_video_callback \
    # scenario_filter.limit_total_scenarios=10 \
    # planner=pdm_closed_planner \
    # observation=world_model_agents_observation \
    # worker=single_machine_thread_pool \
    # worker.max_workers=8 \
    # ~callback.simulation_log_callback \
    # scenario_filter.limit_total_scenarios=5 \
    # scenario_filter.limit_total_scenarios=0.1 \
    # open_loop_boxes / closed_loop_nonreactive_agents
    # worker.threads_per_node=4 \
    # max_callback_workers=4 \
    # planner=ml_planner \
    # planner.ml_planner.model_config='${model}' \
    # planner.ml_planner.checkpoint_path=$MODEL_PATH \
