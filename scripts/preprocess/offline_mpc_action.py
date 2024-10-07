import os
import gzip
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from easydict import EasyDict as edict
import numpy as np
from copy import deepcopy
np.set_printoptions(precision=3, suppress=True)
import glob
from nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer import KineticsTokenizer
# from nuplan_extent.planning.training.modeling.models.renders.kinetics_render import KineticsRender
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils


# scenario_paths = glob.glob("/home/users/yihan01.hu/data/**/**/**/**")
# scenario_paths = glob.glob("/home/users/yihan01.hu/data/nuplan_cache_wm_v5/**/**/**")
scenario_paths = sorted(glob.glob("/mnt/nas26/yihan01.hu/wod_cache/cache_wod_data_2HZ_v5/**/**/**"))[1::2]
# only select without tokenized_data_32x64_v5
# scenario_paths = [path for path in scenario_paths if not os.path.exists(path + '/tokenized_data_32x64_v5.gz')]


# scenario_paths = scenario_paths[::2]

tokenizer = KineticsTokenizer(
  max_seq_len=416
)
# render = KineticsRender(
# )

import multiprocessing
from functools import partial
from easydict import EasyDict as edict
import gzip
import pickle
from tqdm import tqdm
import numpy as np
import os

def process_single_data(tokenizer, scenario_path):
    # Tokenize a single data item
    with gzip.open(scenario_path + '/vector.gz', 'rb') as f:
        vector = pickle.load(f)
    if os.path.exists(scenario_path + '/tokenized_data_32x64_v6.gz'):
        try:
            with gzip.open(scenario_path + '/tokenized_data_32x64_v6.gz', 'rb') as f:
                tokenized_data = pickle.load(f)
            return None
        except:
            print("Error in loading tokenized_data_32x64_v6.gz")

    tokenized_data = tokenizer.forward(edict({"data": [vector]}))
    with gzip.open(scenario_path + '/tokenized_data_32x64_v6.gz', 'wb') as f:
        pickle.dump(tokenized_data, f)  
    return None
        
def multiprocess_tokenization(tokenizer, scenario_paths, pool_size=None):
    # Create a pool of workers, specifying the pool size
    with multiprocessing.Pool(processes=pool_size) as pool:
        # Use tqdm to wrap imap_unordered for progress tracking
        results = list(tqdm(
            pool.imap_unordered(
                partial(process_single_data, tokenizer),
                scenario_paths
            ),
            total=len(scenario_paths)
        ))
    return results
# process_single_data(tokenizer, scenario_paths[0])
multiprocess_tokenization(tokenizer, scenario_paths, pool_size=256)