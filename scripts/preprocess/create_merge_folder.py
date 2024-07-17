import os
import glob

nuplan_path = '/home/users/public/data/home/users/yihan01.hu/data/nuplan_cache/nuplan_type50k_2hz_104m/'
waymo_path = '/home/users/public/data/wod_cache_refactor_range100m_2HZ_with_z/'
target_folder = '/home/users/public/data/merge_cache'

nuplan_path_scenarios = glob.glob(nuplan_path + '**')
waymo_path_scenarios = glob.glob(waymo_path + '**')

all_scenarios = nuplan_path_scenarios + waymo_path_scenarios

for scenario in all_scenarios:
    # Extract the scenario name
    scenario_name = os.path.basename(scenario)

    # Create a full path for the link in the target folder
    link_path = os.path.join(target_folder, scenario_name)

    # Create the symbolic link
    os.symlink(scenario, link_path)
