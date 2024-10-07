import glob
import numpy as np
import gzip
import pickle



scenario_paths = glob.glob("/home/users/yihan01.hu/data/nuplan_cache_wm_v4/**/**/**/tokenized_data_32x64_v3.gz")

ade = []
for i in range(1000):
    with gzip.open(scenario_paths[i]) as f:
        tokenized_data = pickle.load(f)
    x, y, heading = tokenized_data[0, :, :, 2], tokenized_data[0, :, :, 3], tokenized_data[0, :, :, 4]
    dx, dy, dheading = tokenized_data[0, :, :, -5], tokenized_data[0, :, :, -4], tokenized_data[0, :, :, -3]
    valid_mask = tokenized_data[0, :, :, -1]
    tmp_ade = np.sqrt((x - dx)**2 + (y - dy)**2)[valid_mask==1]
    ade.append(tmp_ade)
print(np.concatenate(ade, axis=0).mean())   


ade = []
for i in range(1000):
    with gzip.open(scenario_paths[i]) as f:
        tokenized_data = pickle.load(f)
    x, y, heading = tokenized_data[0, :, :, 2], tokenized_data[0, :, :, 3], tokenized_data[0, :, :, 4]
    dx, dy, dheading = tokenized_data[0, :, :, -5], tokenized_data[0, :, :, -4], tokenized_data[0, :, :, -3]
    valid_mask = tokenized_data[0, :, :, -1]
    tmp_ade = np.sqrt((x - dx)**2 + (y - dy)**2 + (heading - dheading)**2)[valid_mask==1]
    ade.append(tmp_ade)
print(np.concatenate(ade, axis=0).mean())   